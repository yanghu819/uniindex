from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .i2t_sampler_state_ft import _build_model
from .label_feature_probe import VQTokenLabelProbe, labels_to_class_indices
from .layout import TaskLayout, mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps
from .t2i_token_guard import (
    _jsonable_counter,
    _labels_for_class_indices,
    _normalized_token_histogram,
    _per_class_accuracy_from_confusion,
    _real_test_token_histogram,
    _sample_t2i_tokens,
    _sample_unconditional,
    _train_probe,
    confusion_matrix,
)
from .text import label_values_from_text_tokens, metadata_from_state
from .train import _task_time_schedule, latest_checkpoint_path


DEFAULT_DISTRIBUTIONAL_EVAL_STEPS = (100, 300, 500)
DISTRIBUTIONAL_LOSSES = ("hard_ce", "set_ce_k16")
DISTRIBUTIONAL_STATES = ("simplex",)


def _source_checkpoint_path(config: ProjectConfig) -> Path:
    raw_path = config.train.stage2_init_checkpoint
    if raw_path is None:
        return latest_checkpoint_path(config, "stage2")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config.repo_root / path).resolve()


def _load_source_model(
    config: ProjectConfig,
    device: torch.device,
) -> tuple[UnifiedDenoiser, dict, TaskLayout]:
    source_path = _source_checkpoint_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"missing distributional t2i source checkpoint at {source_path}")
    payload = torch.load(source_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    text_metadata = metadata_from_state(tokenizer_state)
    layout = TaskLayout(
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(text_metadata.seq_len),
        codebook_size=int(tokenizer_state["codebook_size"]),
        text_vocab_size=int(text_metadata.vocab_size),
    )
    model = _build_model(config, layout, device)
    model.load_state_dict(payload["model"])
    return model, tokenizer_state, layout


def _record_distributional_eval_steps(total_steps: int, eval_steps: tuple[int, ...] | None = None) -> list[int]:
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    selected = {0, int(total_steps)}
    requested = DEFAULT_DISTRIBUTIONAL_EVAL_STEPS if eval_steps is None else eval_steps
    for step in requested:
        if step < 0:
            raise ValueError(f"eval steps must be >= 0, got {step}")
        if step <= total_steps:
            selected.add(int(step))
    return sorted(selected)


def _sample_endpoint_progress(batch_size: int, *, endpoint_prob: float, device: torch.device) -> torch.Tensor:
    if not 0.0 <= float(endpoint_prob) <= 1.0:
        raise ValueError(f"endpoint_prob must be in [0, 1], got {endpoint_prob}")
    progress = torch.rand(batch_size, device=device)
    endpoint_mask = torch.rand(batch_size, device=device).lt(float(endpoint_prob))
    return torch.where(endpoint_mask, torch.zeros_like(progress), progress)


def build_image_simplex_state(
    *,
    x1: torch.Tensor,
    image_tokens: torch.Tensor,
    layout: TaskLayout,
    t_pos: torch.Tensor,
) -> torch.Tensor:
    z_t = x1.clone()
    image_t = t_pos[:, layout.image_slice].unsqueeze(-1).to(dtype=x1.dtype)
    image_state = torch.zeros_like(z_t[:, layout.image_slice])
    image_state[:, :, : layout.codebook_size] = (1.0 - image_t) / float(layout.codebook_size)
    image_state.scatter_add_(-1, image_tokens.long().unsqueeze(-1), image_t)
    z_t[:, layout.image_slice] = image_state
    return z_t


def _image_hard_ce_loss(logits: torch.Tensor, image_tokens: torch.Tensor, layout: TaskLayout) -> torch.Tensor:
    image_logits = logits[:, layout.image_slice, : layout.codebook_size]
    return F.cross_entropy(image_logits.reshape(-1, layout.codebook_size), image_tokens.long().reshape(-1))


def _label_value_to_index_map(label_values: torch.Tensor) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(label_values.detach().cpu().long().tolist())}


def build_label_candidate_bank_from_batches(
    *,
    batches: list[dict[str, torch.Tensor]],
    label_values: torch.Tensor,
    set_size: int,
) -> torch.Tensor:
    if set_size < 1:
        raise ValueError(f"set_size must be >= 1, got {set_size}")
    label_map = _label_value_to_index_map(label_values)
    buckets: list[list[torch.Tensor]] = [[] for _ in range(int(label_values.numel()))]
    for batch in batches:
        image_tokens = batch["image_tokens"].detach().cpu().long()
        labels = batch["label"].detach().cpu().long()
        for image, label in zip(image_tokens, labels):
            class_index = label_map.get(int(label.item()))
            if class_index is None:
                continue
            buckets[class_index].append(image.clone())

    if any(not bucket for bucket in buckets):
        empty = [int(label_values[index].detach().cpu().item()) for index, bucket in enumerate(buckets) if not bucket]
        raise ValueError(f"candidate bank missing labels: {empty}")

    rows = []
    for bucket in buckets:
        selected = [bucket[index % len(bucket)] for index in range(set_size)]
        rows.append(torch.stack(selected, dim=0))
    return torch.stack(rows, dim=0)


def _build_label_candidate_bank(
    *,
    config: ProjectConfig,
    label_values: torch.Tensor,
    set_size: int,
) -> torch.Tensor:
    loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    return build_label_candidate_bank_from_batches(
        batches=list(loader),
        label_values=label_values.detach().cpu(),
        set_size=set_size,
    )


def set_ce_k_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    candidate_bank: torch.Tensor,
    layout: TaskLayout,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    class_indices = labels_to_class_indices(labels, label_values.to(labels.device))
    if bool(class_indices.lt(0).any().item()):
        raise ValueError("labels contain values that are not present in label_values")
    image_logits = logits[:, layout.image_slice, : layout.codebook_size]
    log_probs = F.log_softmax(image_logits, dim=-1)
    candidates = candidate_bank.to(logits.device).index_select(0, class_indices.long())
    gathered = log_probs.unsqueeze(1).expand(-1, candidates.shape[1], -1, -1).gather(
        -1,
        candidates.long().unsqueeze(-1),
    )
    sequence_nll = -gathered.squeeze(-1).mean(dim=-1)
    log_candidate_count = torch.log(torch.tensor(float(candidates.shape[1]), device=logits.device))
    return (-float(temperature) * (torch.logsumexp(-sequence_nll / float(temperature), dim=1) - log_candidate_count)).mean()


def _reset_sampling_seed(config: ProjectConfig, device: torch.device) -> None:
    seed = config.eval.sampling_seed if config.eval.sampling_seed is not None else config.train.seed
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


@torch.no_grad()
def _evaluate_distributional_checkpoint(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    token_probe: VQTokenLabelProbe,
    label_values: torch.Tensor,
    real_hist: torch.Tensor,
    samples_per_label: int,
    unconditional_count: int,
) -> dict:
    device = next(model.parameters()).device
    model.eval()
    _reset_sampling_seed(config, device)

    num_classes = int(label_values.numel())
    condition_class_indices = torch.arange(num_classes, device=device).repeat_interleave(samples_per_label)
    condition_text_tokens = text_metadata.label_text_tokens.to(device).index_select(0, condition_class_indices)
    generated_tokens = _sample_t2i_tokens(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        config=config,
        text_metadata=text_metadata,
        condition_text_tokens=condition_text_tokens,
        batch_size=config.train.eval_batch_size,
    )

    generated_logits = token_probe(generated_tokens.to(device))
    generated_pred_indices = generated_logits.argmax(dim=1).detach().cpu()
    condition_indices_cpu = condition_class_indices.detach().cpu()
    conditioned_confusion = confusion_matrix(
        generated_pred_indices,
        condition_indices_cpu,
        num_classes=num_classes,
    )
    conditioned_accuracy = float(generated_pred_indices.eq(condition_indices_cpu).float().mean().item())
    generated_hist = _normalized_token_histogram(generated_tokens, layout.codebook_size)
    token_hist_l1 = float((real_hist - generated_hist).abs().sum().item())
    avg_unique_per_sample = float(torch.tensor([row.unique().numel() for row in generated_tokens]).float().mean().item())
    total_unique = int(generated_tokens.unique().numel())
    label_values_cpu = label_values.detach().cpu()

    summary: dict[str, object] = {
        "conditioned_token_label_accuracy": conditioned_accuracy,
        "conditioned_total": int(generated_tokens.shape[0]),
        "conditioned_confusion": conditioned_confusion.tolist(),
        "conditioned_per_label_accuracy": _per_class_accuracy_from_confusion(conditioned_confusion),
        "conditioned_token_pred_histogram": _jsonable_counter(
            Counter(_labels_for_class_indices(generated_pred_indices, label_values_cpu).tolist())
        ),
        "generated_unique_token_count": total_unique,
        "generated_avg_unique_tokens_per_sample": avg_unique_per_sample,
        "generated_vs_real_test_token_histogram_l1": token_hist_l1,
    }

    if unconditional_count > 0:
        unconditional = _sample_unconditional(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            config=config,
            text_metadata=text_metadata,
            count=unconditional_count,
            batch_size=config.train.eval_batch_size,
        )
        uncond_image_tokens = unconditional[:, layout.image_slice]
        uncond_text_tokens = unconditional[:, layout.text_slice] - layout.text_offset
        uncond_pred_indices = token_probe(uncond_image_tokens.to(device)).argmax(dim=1).detach().cpu()
        uncond_pred_values = _labels_for_class_indices(uncond_pred_indices, label_values_cpu)
        uncond_text_values = label_values_from_text_tokens(uncond_text_tokens, text_metadata).detach().cpu()
        valid_text = uncond_text_values.ne(-1)
        valid_consistency = (
            uncond_pred_values[valid_text].eq(uncond_text_values[valid_text]).float().mean().item()
            if bool(valid_text.any().item())
            else 0.0
        )
        summary.update(
            {
                "unconditional_token_text_consistency": float(
                    uncond_pred_values.eq(uncond_text_values).float().mean().item()
                ),
                "unconditional_valid_text_consistency": float(valid_consistency),
                "unconditional_valid_text_fraction": float(valid_text.float().mean().item()),
                "unconditional_token_pred_histogram": _jsonable_counter(Counter(uncond_pred_values.tolist())),
                "unconditional_text_value_histogram": _jsonable_counter(Counter(uncond_text_values.tolist())),
            }
        )

    model.train()
    return summary


def _save_distributional_checkpoint(
    *,
    config: ProjectConfig,
    model: UnifiedDenoiser,
    optimizer: torch.optim.Optimizer,
    tokenizer_state: dict,
    step: int,
    loss_kind: str,
    state_kind: str,
    run_context: RunContext,
) -> Path:
    payload = {
        "stage": "stage2",
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer_state": tokenizer_state,
        "config_name": config.name,
        "probe": "probe-t2i-distributional-ft",
        "loss_kind": loss_kind,
        "state_kind": state_kind,
        "source_checkpoint": str(_source_checkpoint_path(config)),
    }
    path = run_context.log_path(f"checkpoints/distributional_step{step:06d}.pt")
    torch.save(payload, path)
    return path


def _checkpoint_score(snapshot: dict) -> tuple[float, int, float]:
    metrics = snapshot["metrics"]
    return (
        float(metrics["conditioned_token_label_accuracy"]),
        int(metrics["generated_unique_token_count"]),
        -float(metrics["generated_vs_real_test_token_histogram_l1"]),
    )


def _distributional_train_step(
    *,
    model: UnifiedDenoiser,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    batch: dict[str, torch.Tensor],
    label_values: torch.Tensor,
    loss_kind: str,
    state_kind: str,
    endpoint_prob: float,
    candidate_bank: torch.Tensor | None,
    set_temperature: float,
) -> dict[str, float]:
    if loss_kind not in DISTRIBUTIONAL_LOSSES:
        raise ValueError(f"unsupported distributional loss {loss_kind!r}")
    if state_kind not in DISTRIBUTIONAL_STATES:
        raise ValueError(f"unsupported distributional state {state_kind!r}")

    image_tokens = batch["image_tokens"]
    text_tokens = batch["text_tokens"]
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size).to(image_tokens.device)
    modality_ids = layout.position_modalities().to(image_tokens.device)
    progress = _sample_endpoint_progress(image_tokens.shape[0], endpoint_prob=endpoint_prob, device=image_tokens.device)
    t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "text_to_image")
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=False,
        condition_text=True,
    )
    z_t = build_image_simplex_state(
        x1=x1,
        image_tokens=image_tokens,
        layout=layout,
        t_pos=t_pos,
    )

    logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
    if loss_kind == "hard_ce":
        loss = _image_hard_ce_loss(logits, image_tokens, layout)
    else:
        if candidate_bank is None:
            raise ValueError("set_ce_k16 requires a candidate_bank")
        loss = set_ce_k_loss(
            logits=logits,
            labels=batch["label"],
            label_values=label_values,
            candidate_bank=candidate_bank,
            layout=layout,
            temperature=set_temperature,
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
    optimizer.step()
    image_logits = logits[:, layout.image_slice, : layout.codebook_size]
    image_token_accuracy = float(image_logits.argmax(dim=-1).eq(image_tokens).float().mean().detach().cpu().item())
    return {
        "loss": float(loss.detach().cpu().item()),
        "image_token_accuracy": image_token_accuracy,
        "progress_mean": float(progress.detach().cpu().mean().item()),
        "endpoint_fraction": float(progress.eq(0).float().detach().cpu().mean().item()),
        "image_t_mean": float(t_pos[:, layout.image_slice].detach().cpu().mean().item()),
    }


def _infinite(loader):
    while True:
        yield from loader


def run_t2i_distributional_ft_probe(
    *,
    config: ProjectConfig,
    steps: int = 500,
    loss_kind: str = "hard_ce",
    state_kind: str = "simplex",
    set_size: int = 16,
    set_temperature: float = 0.25,
    endpoint_prob: float = 0.9,
    lr: float | None = None,
    eval_steps: tuple[int, ...] | None = None,
    token_probe_steps: int = 200,
    token_probe_eval_every: int = 50,
    samples_per_label: int = 4,
    unconditional_count: int = 10,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if samples_per_label < 1:
        raise ValueError(f"samples_per_label must be >= 1, got {samples_per_label}")
    if unconditional_count < 0:
        raise ValueError(f"unconditional_count must be >= 0, got {unconditional_count}")
    if loss_kind not in DISTRIBUTIONAL_LOSSES:
        raise ValueError(f"unsupported distributional loss {loss_kind!r}")
    if state_kind not in DISTRIBUTIONAL_STATES:
        raise ValueError(f"unsupported distributional state {state_kind!r}")

    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "probe-t2i-distributional-ft")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_source_model(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=device)
    candidate_bank = None
    if loss_kind == "set_ce_k16":
        candidate_bank = _build_label_candidate_bank(config=config, label_values=label_values, set_size=set_size)

    token_probe, token_probe_snapshots = _train_probe(
        config=config,
        layout=layout,
        label_values=label_values,
        steps=token_probe_steps,
        eval_every=token_probe_eval_every,
        lr=float(config.i2t_llm.lr if lr is None else lr),
        run_context=run_context,
    )
    token_probe.eval()
    real_hist = _real_test_token_histogram(config, layout.codebook_size)

    train_loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(train_loader)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr if lr is None else float(lr),
        weight_decay=config.train.weight_decay,
    )
    eval_step_set = set(_record_distributional_eval_steps(steps, eval_steps))
    train_log = run_context.log_path("distributional_train.jsonl")
    eval_log = run_context.log_path("distributional_eval.jsonl")
    history: list[dict] = []

    try:
        model.train()
        for step in tqdm(range(0, steps + 1), desc=f"t2i-distributional/{loss_kind}"):
            if step in eval_step_set:
                metrics = _evaluate_distributional_checkpoint(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    text_metadata=text_metadata,
                    token_probe=token_probe,
                    label_values=label_values,
                    real_hist=real_hist,
                    samples_per_label=samples_per_label,
                    unconditional_count=unconditional_count,
                )
                checkpoint_path = _save_distributional_checkpoint(
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    tokenizer_state=tokenizer_state,
                    step=step,
                    loss_kind=loss_kind,
                    state_kind=state_kind,
                    run_context=run_context,
                )
                snapshot = {
                    "step": int(step),
                    "metrics": metrics,
                    "checkpoint": str(checkpoint_path),
                }
                history.append(snapshot)
                append_jsonl(eval_log, snapshot)
            if step == steps:
                break
            batch = next(train_iter)
            batch = {key: value.to(device) for key, value in batch.items()}
            parts = _distributional_train_step(
                model=model,
                optimizer=optimizer,
                config=config,
                layout=layout,
                schedule_tables=schedule_tables,
                batch=batch,
                label_values=label_values,
                loss_kind=loss_kind,
                state_kind=state_kind,
                endpoint_prob=endpoint_prob,
                candidate_bank=candidate_bank,
                set_temperature=set_temperature,
            )
            append_jsonl(train_log, {"step": int(step + 1), **parts})

        best = max(history, key=_checkpoint_score)
        latest_path = latest_checkpoint_path(config, "stage2")
        shutil.copyfile(best["checkpoint"], latest_path)
        summary = {
            "source": "probe-t2i-distributional-ft",
            "source_checkpoint": str(_source_checkpoint_path(config)),
            "best_checkpoint_copied_to": str(latest_path),
            "steps": int(steps),
            "loss_kind": loss_kind,
            "state_kind": state_kind,
            "set_size": int(set_size),
            "set_temperature": float(set_temperature),
            "endpoint_prob": float(endpoint_prob),
            "lr": float(config.train.lr if lr is None else lr),
            "eval_steps": [int(step) for step in sorted(eval_step_set)],
            "samples_per_label": int(samples_per_label),
            "unconditional_count": int(unconditional_count),
            "token_probe_steps": int(token_probe_steps),
            "token_probe_snapshots": token_probe_snapshots,
            "initial": history[0],
            "best": best,
            "final": history[-1],
            "history": history,
        }
        summary_path = run_context.log_path("summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if own_context:
            run_context.update_status("ok")
        return summary
    except Exception:
        if own_context:
            run_context.update_status("error")
        raise
