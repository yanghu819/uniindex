from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import TaskLayout, mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise
from .task_schedule import task_for_step
from .text import metadata_from_state


def latest_checkpoint_path(config: ProjectConfig, stage: str) -> Path:
    path = config.paths.models_dir / "checkpoints" / f"{stage}_latest.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _stage2_init_checkpoint_path(config: ProjectConfig) -> Path:
    raw_path = config.train.stage2_init_checkpoint
    if raw_path is None:
        return latest_checkpoint_path(config, "stage1")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config.repo_root / path).resolve()


def _infinite(loader):
    while True:
        yield from loader


def _build_zt(
    x1: torch.Tensor,
    t_pos: torch.Tensor,
    layout: TaskLayout,
    task: str,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    if task == "joint":
        return mix_flm_noise(x1, t_pos, valid_token_mask)
    z_t = x1.clone()
    if task == "text_to_image":
        z_t[:, layout.image_slice] = mix_flm_noise(
            x1[:, layout.image_slice],
            t_pos[:, layout.image_slice],
            valid_token_mask[layout.image_slice],
        )
    elif task == "image_to_text":
        z_t[:, layout.text_slice] = mix_flm_noise(
            x1[:, layout.text_slice],
            t_pos[:, layout.text_slice],
            valid_token_mask[layout.text_slice],
        )
    else:
        raise ValueError(f"unknown task {task}")
    return z_t


def _task_time_schedule(
    config: ProjectConfig,
    progress: torch.Tensor,
    modality_ids: torch.Tensor,
    schedule_tables: dict,
    task: str,
) -> torch.Tensor:
    text_time = config.train.text_time_power
    if task == "image_to_text" and config.train.image_to_text_text_time_power is not None:
        text_time = config.train.image_to_text_text_time_power
    return apply_schedule(
        progress=progress,
        modality_ids=modality_ids,
        schedule_tables=schedule_tables,
        image_time_power=config.train.image_time_power,
        text_time_power=text_time,
    )


def _apply_image_to_text_noise_policy(
    t_pos: torch.Tensor,
    layout: TaskLayout,
    task: str,
    *,
    text_time_cap: float | None,
    noise_only_prob: float,
) -> torch.Tensor:
    if task != "image_to_text":
        return t_pos
    adjusted = t_pos.clone()
    if text_time_cap is not None:
        adjusted[:, layout.text_slice] = adjusted[:, layout.text_slice].clamp_max(float(text_time_cap))
    if noise_only_prob <= 0.0:
        return adjusted
    if noise_only_prob >= 1.0:
        adjusted[:, layout.text_slice] = 0.0
        return adjusted
    noise_only = torch.rand(adjusted.shape[0], device=adjusted.device).lt(float(noise_only_prob))
    adjusted[noise_only, layout.text_slice] = 0.0
    return adjusted


def _masked_text_loss(logits: torch.Tensor, targets: torch.Tensor, text_pad_token: int) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    valid = targets.ne(text_pad_token)
    if not torch.any(valid):
        return losses.mean()
    return losses[valid].mean()


def _loss_for_task(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layout: TaskLayout,
    joint_weight: float,
    text_weight: float,
    text_pad_id: int,
    task: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked = mask_logits(logits, layout=layout)
    image_loss = F.cross_entropy(
        masked[:, layout.image_slice].reshape(-1, masked.shape[-1]),
        targets[:, layout.image_slice].reshape(-1),
    )
    text_loss = _masked_text_loss(
        masked[:, layout.text_slice].reshape(-1, masked.shape[-1]),
        targets[:, layout.text_slice].reshape(-1),
        text_pad_token=layout.text_offset + text_pad_id,
    )
    if task == "joint":
        loss = joint_weight * image_loss + text_weight * text_loss
    elif task == "text_to_image":
        loss = image_loss
    else:
        loss = text_loss
    return loss, {"image_loss": float(image_loss.item()), "text_loss": float(text_loss.item())}


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
    torch.save(payload, run_context.log_path(f"checkpoints/{stage}_step{step:06d}.pt"))
    return path


def train_stage(config: ProjectConfig, stage: str, run_context: RunContext | None = None) -> Path:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, stage)
    run_context.set_device(device)

    tokenizer_state = load_tokenizer_state(config)
    text_metadata = metadata_from_state(tokenizer_state)
    layout = TaskLayout(
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(text_metadata.seq_len),
        codebook_size=int(tokenizer_state["codebook_size"]),
        text_vocab_size=int(text_metadata.vocab_size),
    )
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(loader)
    model = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)
    if stage == "stage2":
        init_path = _stage2_init_checkpoint_path(config)
        if not init_path.exists():
            raise FileNotFoundError(f"missing stage2 init checkpoint at {init_path}")
        payload = torch.load(init_path, map_location=device)
        model.load_state_dict(payload["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    total_steps = config.train.stage1_steps if stage == "stage1" else config.train.stage2_steps
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    train_log = run_context.log_path("train.jsonl")

    model.train()
    for step in tqdm(range(1, total_steps + 1), desc=stage):
        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
        x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
        progress = torch.rand(image_tokens.shape[0], device=device)
        task = task_for_step(config, stage, step - 1)
        t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, task)
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=task == "image_to_text",
            condition_text=task == "text_to_image",
        )
        t_pos = _apply_image_to_text_noise_policy(
            t_pos,
            layout,
            task,
            text_time_cap=config.train.image_to_text_text_time_cap,
            noise_only_prob=config.train.image_to_text_noise_only_prob,
        )
        z_t = _build_zt(x1, t_pos, layout, task, valid_token_mask)
        loss, parts = _loss_for_task(
            logits=model(z_t, t_pos, modality_ids),
            targets=targets,
            layout=layout,
            joint_weight=config.train.joint_weight,
            text_weight=config.train.text_weight,
            text_pad_id=text_metadata.pad_id,
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
                    "image_t_mean": float(t_pos[:, layout.image_slice].mean().item()),
                    "text_t_mean": float(t_pos[:, layout.text_slice].mean().item()),
                    **parts,
                },
            )

        if step % config.train.save_every == 0 or step == total_steps:
            _save_checkpoint(config, stage, model, optimizer, step, tokenizer_state, run_context)

    if own_context:
        run_context.update_status("ok")
    return latest_checkpoint_path(config, stage)
