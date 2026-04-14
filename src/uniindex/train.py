from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import mask_logits, position_modalities, position_valid_token_mask, unified_targets, unified_vocab_size
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .state import apply_time_schedule, build_flm_clean_state, condition_clean_timesteps, mix_flm_noise
from .task_schedule import task_for_step


def latest_checkpoint_path(config: ProjectConfig, stage: str) -> Path:
    path = config.paths.models_dir / "checkpoints" / f"{stage}_latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _infinite(loader):
    while True:
        yield from loader


def _build_zt(
    x1: torch.Tensor,
    t_pos: torch.Tensor,
    image_seq_len: int,
    task: str,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    if task == "joint":
        return mix_flm_noise(x1, t_pos, valid_token_mask)
    z_t = x1.clone()
    if task == "label_to_image":
        z_t[:, :image_seq_len] = mix_flm_noise(
            x1[:, :image_seq_len],
            t_pos[:, :image_seq_len],
            valid_token_mask[:image_seq_len],
        )
    elif task == "image_to_label":
        z_t[:, image_seq_len:] = mix_flm_noise(
            x1[:, image_seq_len:],
            t_pos[:, image_seq_len:],
            valid_token_mask[image_seq_len:],
        )
    else:
        raise ValueError(f"unknown task {task}")
    return z_t


def _task_time_schedule(config: ProjectConfig, progress: torch.Tensor, modality_ids: torch.Tensor, task: str) -> torch.Tensor:
    label_time_power = config.train.label_time_power
    if task == "image_to_label" and config.train.image_to_label_label_time_power is not None:
        label_time_power = config.train.image_to_label_label_time_power
    return apply_time_schedule(
        progress=progress,
        modality_ids=modality_ids,
        image_time_power=config.train.image_time_power,
        label_time_power=label_time_power,
    )


def _loss_for_task(
    logits: torch.Tensor,
    targets: torch.Tensor,
    image_seq_len: int,
    codebook_size: int,
    num_labels: int,
    joint_weight: float,
    label_weight: float,
    task: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked = mask_logits(logits, image_seq_len, codebook_size, num_labels)
    img_logits = masked[:, :image_seq_len].reshape(-1, masked.shape[-1])
    img_targets = targets[:, :image_seq_len].reshape(-1)
    lbl_logits = masked[:, image_seq_len:].reshape(-1, masked.shape[-1])
    lbl_targets = targets[:, image_seq_len:].reshape(-1)

    image_loss = F.cross_entropy(img_logits, img_targets)
    label_loss = F.cross_entropy(lbl_logits, lbl_targets)

    if task == "joint":
        loss = joint_weight * image_loss + label_weight * label_loss
    elif task == "label_to_image":
        loss = image_loss
    else:
        loss = label_loss
    return loss, {"image_loss": float(image_loss.item()), "label_loss": float(label_loss.item())}


def _save_checkpoint(
    config: ProjectConfig,
    stage: str,
    model: UnifiedDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokenizer_state: dict,
    run_context: RunContext,
) -> Path:
    path = latest_checkpoint_path(config, stage)
    payload = {
        "stage": stage,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer_state": tokenizer_state,
        "config_name": config.name,
    }
    torch.save(payload, path)
    run_ckpt = run_context.log_path(f"checkpoints/{stage}_step{step:06d}.pt")
    torch.save(payload, run_ckpt)
    return path


def train_stage(config: ProjectConfig, stage: str, run_context: RunContext | None = None) -> Path:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, stage)
    run_context.set_device(device)

    tokenizer_state = load_tokenizer_state(config)
    codebook_size = int(tokenizer_state["codebook_size"])
    image_seq_len = int(torch.load(split_path(config, "train"))["image_seq_len"])
    num_labels = len(config.labels.values)
    vocab_size = unified_vocab_size(codebook_size, num_labels)

    loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(loader)

    model = UnifiedDenoiser(
        input_dim=vocab_size,
        seq_len=image_seq_len + 1,
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)

    if stage == "stage2":
        stage1_path = latest_checkpoint_path(config, "stage1")
        if not stage1_path.exists():
            raise FileNotFoundError(f"missing stage1 checkpoint at {stage1_path}")
        payload = torch.load(stage1_path, map_location=device)
        model.load_state_dict(payload["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)

    total_steps = config.train.stage1_steps if stage == "stage1" else config.train.stage2_steps
    modality_ids = position_modalities(image_seq_len).to(device)
    valid_token_mask = position_valid_token_mask(image_seq_len, codebook_size, num_labels).to(device)
    train_log = run_context.log_path("train.jsonl")

    model.train()
    for step in tqdm(range(1, total_steps + 1), desc=stage):
        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        targets = unified_targets(image_tokens, labels, codebook_size)
        x1 = build_flm_clean_state(targets, vocab_size).to(device)
        progress = torch.rand(image_tokens.shape[0], device=device)
        task = task_for_step(config, stage, step - 1)
        t_pos = _task_time_schedule(config, progress, modality_ids, task)
        t_pos = condition_clean_timesteps(
            t_pos,
            image_seq_len,
            condition_image=task == "image_to_label",
            condition_label=task == "label_to_image",
        )
        z_t = _build_zt(x1, t_pos, image_seq_len, task, valid_token_mask)
        logits = model(z_t, t_pos, modality_ids)
        loss, parts = _loss_for_task(
            logits=logits,
            targets=targets,
            image_seq_len=image_seq_len,
            codebook_size=codebook_size,
            num_labels=num_labels,
            joint_weight=config.train.joint_weight,
            label_weight=config.train.label_weight,
            task=task,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()

        if step % config.train.log_every == 0 or step == 1:
            append_jsonl(
                train_log,
                {
                    "step": step,
                    "task": task,
                    "loss": float(loss.item()),
                    "image_t_mean": float(t_pos[:, :image_seq_len].mean().item()),
                    "label_t_mean": float(t_pos[:, image_seq_len:].mean().item()),
                    **parts,
                },
            )

        if step % config.train.save_every == 0 or step == total_steps:
            _save_checkpoint(config, stage, model, optimizer, step, tokenizer_state, run_context)

    if own_context:
        run_context.update_status("ok")
    return latest_checkpoint_path(config, stage)
