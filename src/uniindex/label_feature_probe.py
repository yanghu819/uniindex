from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .i2t_llm_decoder import _load_frozen_denoiser, extract_i2t_image_features
from .layout import TaskLayout
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables


class VQTokenLabelProbe(nn.Module):
    def __init__(self, codebook_size: int, seq_len: int, num_classes: int, d_model: int = 128) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(codebook_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(image_tokens.shape[1], device=image_tokens.device)
        hidden = self.token_embed(image_tokens) + self.pos_embed(positions).unsqueeze(0)
        return self.head(self.norm(hidden.mean(dim=1)))


class PooledFeatureLabelProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@dataclass(frozen=True)
class ProbeBatchResult:
    loss: torch.Tensor
    accuracy: float


def labels_to_class_indices(labels: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    label_values = label_values.to(labels.device)
    matches = labels[:, None].eq(label_values[None, :])
    if not bool(matches.any(dim=1).all().item()):
        missing = labels[~matches.any(dim=1)].detach().cpu().tolist()
        raise ValueError(f"labels contain values outside config.labels.values: {missing}")
    return matches.float().argmax(dim=1).long()


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float(logits.argmax(dim=1).eq(targets).float().mean().detach().cpu().item())


def _infinite(loader):
    while True:
        yield from loader


@torch.no_grad()
def _evaluate(
    *,
    config: ProjectConfig,
    layout: TaskLayout,
    denoiser,
    schedule_tables: dict,
    vq_probe: VQTokenLabelProbe,
    flm_probe: PooledFeatureLabelProbe,
    label_values: torch.Tensor,
    split: str,
) -> dict[str, float]:
    device = label_values.device
    loader = build_loader(
        split_path(config, split),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    totals = {
        "vq_correct": 0,
        "flm_correct": 0,
        "total": 0,
    }
    for batch in tqdm(loader, desc=f"probe-label-features/{split}"):
        image_tokens = batch["image_tokens"].to(device)
        labels = labels_to_class_indices(batch["label"].to(device), label_values)
        vq_logits = vq_probe(image_tokens)
        flm_features = extract_i2t_image_features(
            denoiser=denoiser,
            config=config,
            layout=layout,
            schedule_tables=schedule_tables,
            image_tokens=image_tokens,
        )
        flm_logits = flm_probe(flm_features)
        totals["vq_correct"] += int(vq_logits.argmax(dim=1).eq(labels).sum().item())
        totals["flm_correct"] += int(flm_logits.argmax(dim=1).eq(labels).sum().item())
        totals["total"] += int(labels.numel())
    total = max(totals["total"], 1)
    return {
        f"{split}_vq_token_accuracy": totals["vq_correct"] / total,
        f"{split}_flm_hidden_accuracy": totals["flm_correct"] / total,
        f"{split}_total": float(totals["total"]),
    }


def _train_step(
    *,
    config: ProjectConfig,
    layout: TaskLayout,
    denoiser,
    schedule_tables: dict,
    vq_probe: VQTokenLabelProbe,
    flm_probe: PooledFeatureLabelProbe,
    label_values: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = label_values.device
    image_tokens = batch["image_tokens"].to(device)
    labels = labels_to_class_indices(batch["label"].to(device), label_values)
    vq_logits = vq_probe(image_tokens)
    flm_features = extract_i2t_image_features(
        denoiser=denoiser,
        config=config,
        layout=layout,
        schedule_tables=schedule_tables,
        image_tokens=image_tokens,
    )
    flm_logits = flm_probe(flm_features)
    vq_loss = F.cross_entropy(vq_logits, labels)
    flm_loss = F.cross_entropy(flm_logits, labels)
    loss = vq_loss + flm_loss
    return loss, {
        "vq_loss": float(vq_loss.detach().cpu().item()),
        "flm_loss": float(flm_loss.detach().cpu().item()),
        "vq_batch_accuracy": _accuracy(vq_logits, labels),
        "flm_batch_accuracy": _accuracy(flm_logits, labels),
    }


def run_label_feature_probe(
    *,
    config: ProjectConfig,
    steps: int = 200,
    eval_every: int = 50,
    run_context: RunContext | None = None,
) -> dict:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    run_context = run_context or RunContext(config, "probe-label-features")
    run_context.set_device(device)

    denoiser, _, layout, _ = _load_frozen_denoiser(config, device)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    label_values = torch.tensor(config.labels.values, dtype=torch.long, device=device)
    num_classes = int(label_values.numel())
    vq_probe = VQTokenLabelProbe(
        codebook_size=layout.codebook_size,
        seq_len=layout.image_seq_len,
        num_classes=num_classes,
    ).to(device)
    flm_probe = PooledFeatureLabelProbe(input_dim=config.model.d_model, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(
        list(vq_probe.parameters()) + list(flm_probe.parameters()),
        lr=config.i2t_llm.lr,
        weight_decay=config.train.weight_decay,
    )
    train_loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(train_loader)
    train_log = run_context.log_path("label_feature_train.jsonl")
    eval_log = run_context.log_path("label_feature_eval.jsonl")
    eval_steps = {0, steps}
    eval_steps.update(range(eval_every, steps + 1, eval_every))
    snapshots: list[dict] = []
    summary = {
        "source": "probe-label-features",
        "steps": int(steps),
        "eval_every": int(eval_every),
        "source_checkpoint": str(config.paths.models_dir / "checkpoints" / "stage2_latest.pt"),
        "feature_progress": float(config.i2t_llm.feature_progress),
        "snapshots": snapshots,
    }

    for step in range(steps + 1):
        if step in eval_steps:
            vq_probe.eval()
            flm_probe.eval()
            metrics = {
                "step": step,
                **_evaluate(
                    config=config,
                    layout=layout,
                    denoiser=denoiser,
                    schedule_tables=schedule_tables,
                    vq_probe=vq_probe,
                    flm_probe=flm_probe,
                    label_values=label_values,
                    split="test",
                ),
            }
            snapshots.append(metrics)
            append_jsonl(eval_log, metrics)
            vq_probe.train()
            flm_probe.train()
        if step == steps:
            break

        batch = next(train_iter)
        loss, parts = _train_step(
            config=config,
            layout=layout,
            denoiser=denoiser,
            schedule_tables=schedule_tables,
            vq_probe=vq_probe,
            flm_probe=flm_probe,
            label_values=label_values,
            batch=batch,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(vq_probe.parameters()) + list(flm_probe.parameters()),
            max_norm=config.train.grad_clip_norm,
        )
        optimizer.step()
        if (step + 1) % config.train.log_every == 0 or step == 0:
            append_jsonl(
                train_log,
                {
                    "step": step + 1,
                    "loss": float(loss.detach().cpu().item()),
                    **parts,
                },
            )

    best_vq = max(snapshots, key=lambda record: record["test_vq_token_accuracy"]) if snapshots else {}
    best_flm = max(snapshots, key=lambda record: record["test_flm_hidden_accuracy"]) if snapshots else {}
    summary["best_vq"] = best_vq
    summary["best_flm"] = best_flm
    summary["final"] = snapshots[-1] if snapshots else {}
    summary_path = run_context.log_path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
