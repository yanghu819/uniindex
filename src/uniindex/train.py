from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import TaskLayout, mask_logits, maybe_mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_state
from .task_schedule import task_for_step
from .text import metadata_from_state, sequence_candidate_scores, shifted_label_text_tokens


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
    valid_token_mask: torch.Tensor | None,
    state_path: str = "gaussian",
) -> torch.Tensor:
    if task == "joint":
        return mix_flm_state(x1, t_pos, path=state_path, valid_token_mask=valid_token_mask)
    z_t = x1.clone()
    if task == "text_to_image":
        image_valid_token_mask = None if valid_token_mask is None else valid_token_mask[layout.image_slice]
        z_t[:, layout.image_slice] = mix_flm_state(
            x1[:, layout.image_slice],
            t_pos[:, layout.image_slice],
            path=state_path,
            valid_token_mask=image_valid_token_mask,
        )
    elif task == "image_to_text":
        text_valid_token_mask = None if valid_token_mask is None else valid_token_mask[layout.text_slice]
        z_t[:, layout.text_slice] = mix_flm_state(
            x1[:, layout.text_slice],
            t_pos[:, layout.text_slice],
            path=state_path,
            valid_token_mask=text_valid_token_mask,
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


def _masked_text_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    text_pad_token: int,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, targets, reduction="none")
    valid = targets.ne(text_pad_token)
    if not torch.any(valid):
        return losses.mean()
    return losses[valid].mean()


def _sequence_text_loss(
    text_logits: torch.Tensor,
    text_targets: torch.Tensor,
    candidate_text_targets: torch.Tensor,
) -> torch.Tensor:
    scores = sequence_candidate_scores(text_logits, candidate_text_targets)
    matches = text_targets.unsqueeze(1).eq(candidate_text_targets.unsqueeze(0)).all(dim=-1)
    if not torch.all(matches.any(dim=1)):
        raise ValueError("each text target row must exactly match one canonical text candidate")
    target_indices = matches.float().argmax(dim=1)
    return F.cross_entropy(scores, target_indices)


def _labels_to_class_indices(labels: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    label_values = label_values.to(labels.device)
    matches = labels[:, None].eq(label_values[None, :])
    if not bool(matches.any(dim=1).all().item()):
        missing = labels[~matches.any(dim=1)].detach().cpu().tolist()
        raise ValueError(f"labels contain values outside config.labels.values: {missing}")
    return matches.float().argmax(dim=1).long()


def _image_to_text_semantic_label_loss(
    *,
    model: UnifiedDenoiser,
    label_head: nn.Module,
    x1: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    modality_ids: torch.Tensor,
    layout: TaskLayout,
    valid_token_mask: torch.Tensor | None,
    text_time: float,
    pool: str = "image",
    state_path: str = "gaussian",
) -> torch.Tensor:
    t_pos = torch.full(
        (x1.shape[0], layout.seq_len),
        float(text_time),
        device=x1.device,
        dtype=x1.dtype,
    )
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=True,
        condition_text=False,
    )
    z_t = _build_zt(x1, t_pos, layout, "image_to_text", valid_token_mask, state_path)
    hidden = model.forward_features(z_t, t_pos, modality_ids, include_extra_tokens=pool == "semantic")
    pooled = _pool_semantic_hidden(hidden, layout, pool)
    class_targets = _labels_to_class_indices(labels, label_values)
    return F.cross_entropy(label_head(pooled), class_targets)


def _pool_semantic_hidden(hidden: torch.Tensor, layout: TaskLayout, pool: str) -> torch.Tensor:
    if pool == "image":
        return hidden[:, layout.image_slice].mean(dim=1)
    if pool == "text":
        return hidden[:, layout.text_slice].mean(dim=1)
    if pool == "all":
        return hidden.mean(dim=1)
    if pool == "semantic":
        if hidden.shape[1] <= layout.seq_len:
            raise ValueError("semantic feature pool requires model.image_semantic_tokens > 0")
        return hidden[:, layout.seq_len :].mean(dim=1)
    raise ValueError(f"unsupported semantic feature pool: {pool}")


def _image_text_mismatch_loss(
    *,
    model: UnifiedDenoiser,
    x1: torch.Tensor,
    t_pos: torch.Tensor,
    modality_ids: torch.Tensor,
    layout: TaskLayout,
    valid_token_mask: torch.Tensor | None,
    text_targets: torch.Tensor,
    candidate_text_targets: torch.Tensor,
    margin: float,
    state_path: str = "gaussian",
) -> torch.Tensor:
    if x1.shape[0] < 2:
        return torch.zeros((), device=x1.device)
    shifted_x1 = x1.roll(shifts=1, dims=0)
    mismatched = x1.clone()
    mismatched[:, layout.image_slice] = shifted_x1[:, layout.image_slice]
    z_t = _build_zt(mismatched, t_pos, layout, "image_to_text", valid_token_mask, state_path)
    mismatch_logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
    candidate_targets = candidate_text_targets.to(mismatch_logits.device)
    scores = sequence_candidate_scores(mismatch_logits[:, layout.text_slice], candidate_targets)
    matches = text_targets.unsqueeze(1).eq(candidate_text_targets.to(text_targets.device).unsqueeze(0)).all(dim=-1)
    if not torch.all(matches.any(dim=1)):
        raise ValueError("each text target row must exactly match one canonical text candidate")
    shifted_text_targets = text_targets.roll(shifts=1, dims=0)
    shifted_matches = shifted_text_targets.unsqueeze(1).eq(
        candidate_text_targets.to(text_targets.device).unsqueeze(0)
    ).all(dim=-1)
    if not torch.all(shifted_matches.any(dim=1)):
        raise ValueError("each shifted text target row must exactly match one canonical text candidate")
    valid_mismatch = text_targets.ne(shifted_text_targets).any(dim=1)
    if not torch.any(valid_mismatch):
        return torch.zeros((), device=x1.device)

    wrong_indices = matches.float().argmax(dim=1)
    correct_indices = shifted_matches.float().argmax(dim=1)
    wrong_score = scores.gather(dim=1, index=wrong_indices[:, None].to(scores.device)).squeeze(1)
    correct_score = scores.gather(dim=1, index=correct_indices[:, None].to(scores.device)).squeeze(1)
    losses = F.relu(float(margin) + wrong_score - correct_score)
    return losses[valid_mismatch.to(losses.device)].mean()


def _image_to_text_label_loss(
    *,
    model: UnifiedDenoiser,
    x1: torch.Tensor,
    modality_ids: torch.Tensor,
    layout: TaskLayout,
    valid_token_mask: torch.Tensor | None,
    text_targets: torch.Tensor,
    candidate_text_targets: torch.Tensor,
    text_time: float,
    state_path: str = "gaussian",
) -> torch.Tensor:
    t_pos = torch.full(
        (x1.shape[0], layout.seq_len),
        float(text_time),
        device=x1.device,
        dtype=x1.dtype,
    )
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=True,
        condition_text=False,
    )
    z_t = _build_zt(x1, t_pos, layout, "image_to_text", valid_token_mask, state_path)
    label_logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
    return _sequence_text_loss(
        text_logits=label_logits[:, layout.text_slice],
        text_targets=text_targets,
        candidate_text_targets=candidate_text_targets.to(label_logits.device),
    )


def _loss_for_task(
    logits: torch.Tensor,
    targets: torch.Tensor,
    layout: TaskLayout,
    joint_weight: float,
    text_weight: float,
    text_pad_id: int,
    label_text_tokens: torch.Tensor,
    text_sequence_weight: float,
    task: str,
    model: UnifiedDenoiser | None = None,
    x1: torch.Tensor | None = None,
    t_pos: torch.Tensor | None = None,
    modality_ids: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    image_to_text_mismatch_weight: float = 0.0,
    image_to_text_mismatch_margin: float = 1.0,
    image_to_text_label_weight: float = 0.0,
    image_to_text_label_text_time: float = 0.0,
    logit_mask: str = "modality",
    state_path: str = "gaussian",
) -> tuple[torch.Tensor, dict[str, float]]:
    masked = maybe_mask_logits(logits, layout=layout, mode=logit_mask)
    img_logits = masked[:, layout.image_slice].reshape(-1, masked.shape[-1])
    img_targets = targets[:, layout.image_slice].reshape(-1)
    text_logits = masked[:, layout.text_slice]
    txt_logits = text_logits.reshape(-1, masked.shape[-1])
    text_targets = targets[:, layout.text_slice]
    txt_targets = text_targets.reshape(-1)

    image_loss = F.cross_entropy(img_logits, img_targets)
    text_loss = _masked_text_loss(txt_logits, txt_targets, text_pad_token=layout.text_offset + text_pad_id)
    sequence_loss = torch.zeros((), device=masked.device)
    if text_sequence_weight > 0.0 and task in {"joint", "image_to_text"}:
        sequence_loss = _sequence_text_loss(
            text_logits=text_logits,
            text_targets=text_targets,
            candidate_text_targets=label_text_tokens.to(masked.device),
        )
    mismatch_loss = torch.zeros((), device=masked.device)
    if image_to_text_mismatch_weight > 0.0 and task == "image_to_text":
        if model is None or x1 is None or t_pos is None or modality_ids is None or valid_token_mask is None:
            raise ValueError("image_to_text mismatch loss requires model, x1, t_pos, modality_ids, and valid_token_mask")
        mismatch_loss = _image_text_mismatch_loss(
            model=model,
            x1=x1,
            t_pos=t_pos,
            modality_ids=modality_ids,
            layout=layout,
            valid_token_mask=valid_token_mask,
            text_targets=text_targets,
            candidate_text_targets=label_text_tokens.to(masked.device),
            margin=image_to_text_mismatch_margin,
            state_path=state_path,
        )
    label_loss = torch.zeros((), device=masked.device)
    if image_to_text_label_weight > 0.0 and task == "image_to_text":
        if model is None or x1 is None or modality_ids is None or valid_token_mask is None:
            raise ValueError("image_to_text label loss requires model, x1, modality_ids, and valid_token_mask")
        label_loss = _image_to_text_label_loss(
            model=model,
            x1=x1,
            modality_ids=modality_ids,
            layout=layout,
            valid_token_mask=valid_token_mask,
            text_targets=text_targets,
            candidate_text_targets=label_text_tokens.to(masked.device),
            text_time=image_to_text_label_text_time,
            state_path=state_path,
        )

    if task == "joint":
        loss = joint_weight * image_loss + text_weight * text_loss
    elif task == "text_to_image":
        loss = image_loss
    else:
        loss = text_loss
    if text_sequence_weight > 0.0 and task in {"joint", "image_to_text"}:
        loss = loss + text_sequence_weight * sequence_loss
    if image_to_text_mismatch_weight > 0.0 and task == "image_to_text":
        loss = loss + image_to_text_mismatch_weight * mismatch_loss
    if image_to_text_label_weight > 0.0 and task == "image_to_text":
        loss = loss + image_to_text_label_weight * label_loss

    return loss, {
        "image_loss": float(image_loss.item()),
        "text_loss": float(text_loss.item()),
        "sequence_loss": float(sequence_loss.item()),
        "mismatch_loss": float(mismatch_loss.item()),
        "label_loss": float(label_loss.item()),
    }


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
    shifted_label_tokens = shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size)
    label_values = torch.tensor(config.labels.values, dtype=torch.long, device=device)

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
        image_summary_to_text=config.model.image_summary_to_text,
        image_semantic_tokens=config.model.image_semantic_tokens,
        image_semantic_source=config.model.image_semantic_source,
        image_vocab_size=layout.codebook_size,
    ).to(device)

    if stage == "stage2":
        init_path = _stage2_init_checkpoint_path(config)
        if not init_path.exists():
            raise FileNotFoundError(f"missing stage2 init checkpoint at {init_path}")
        payload = torch.load(init_path, map_location=device)
        model.load_state_dict(payload["model"])

    semantic_label_head: nn.Module | None = None
    optimizer_params: list[nn.Parameter] = list(model.parameters())
    if config.train.image_to_text_semantic_weight > 0.0:
        semantic_label_head = nn.Sequential(
            nn.LayerNorm(config.model.d_model),
            nn.Linear(config.model.d_model, len(config.labels.values)),
        ).to(device)
        optimizer_params += list(semantic_label_head.parameters())

    optimizer = torch.optim.AdamW(optimizer_params, lr=config.train.lr, weight_decay=config.train.weight_decay)

    total_steps = config.train.stage1_steps if stage == "stage1" else config.train.stage2_steps
    modality_ids = layout.position_modalities().to(device)
    modality_valid_token_mask = layout.position_valid_token_mask().to(device)
    noise_valid_token_mask = (
        None if config.state.noise_support == "full_vocab" else modality_valid_token_mask
    )
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
        z_t = _build_zt(x1, t_pos, layout, task, noise_valid_token_mask, config.state.path)
        logits = model(z_t, t_pos, modality_ids)
        loss, parts = _loss_for_task(
            logits=logits,
            targets=targets,
            layout=layout,
            joint_weight=config.train.joint_weight,
            text_weight=config.train.text_weight,
            text_pad_id=text_metadata.pad_id,
            label_text_tokens=shifted_label_tokens,
            text_sequence_weight=config.train.text_sequence_weight,
            task=task,
            model=model,
            x1=x1,
            t_pos=t_pos,
            modality_ids=modality_ids,
            valid_token_mask=noise_valid_token_mask,
            image_to_text_mismatch_weight=config.train.image_to_text_mismatch_weight,
            image_to_text_mismatch_margin=config.train.image_to_text_mismatch_margin,
            image_to_text_label_weight=config.train.image_to_text_label_weight,
            image_to_text_label_text_time=config.train.image_to_text_label_text_time,
            logit_mask=config.train.logit_mask,
            state_path=config.state.path,
        )
        semantic_label_loss = torch.zeros((), device=device)
        if semantic_label_head is not None and task == "image_to_text":
            semantic_label_loss = _image_to_text_semantic_label_loss(
                model=model,
                label_head=semantic_label_head,
                x1=x1,
                labels=batch["label"].to(device),
                label_values=label_values,
                modality_ids=modality_ids,
                layout=layout,
                valid_token_mask=noise_valid_token_mask,
                text_time=config.train.image_to_text_semantic_text_time,
                pool=config.train.image_to_text_semantic_pool,
                state_path=config.state.path,
            )
            loss = loss + config.train.image_to_text_semantic_weight * semantic_label_loss
        parts["semantic_label_loss"] = float(semantic_label_loss.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=config.train.grad_clip_norm)
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
