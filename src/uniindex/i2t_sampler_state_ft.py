from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import (
    _apply_i2t_text_time_schedule,
    _project_text_state,
    _projection_step_indices,
)
from .layout import TaskLayout, mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise, sample_masked_noise
from .text import metadata_from_state, sequence_candidate_scores, shifted_label_text_tokens
from .train import _build_zt, _masked_text_loss, _sequence_text_loss, _task_time_schedule, latest_checkpoint_path


DEFAULT_SAMPLER_STATE_PROGRESS = (0.5, 0.75, 0.9)
DEFAULT_ANCHOR_TASKS = ("joint", "text_to_image")
TRAINABLE_SCOPES = ("all", "head", "last_block", "last_two_blocks")


@dataclass(frozen=True)
class SamplerStateBatch:
    progress: float
    z_t: torch.Tensor
    t_pos: torch.Tensor


def _source_checkpoint_path(config: ProjectConfig) -> Path:
    raw_path = config.train.stage2_init_checkpoint
    if raw_path is None:
        return latest_checkpoint_path(config, "stage2")
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config.repo_root / path).resolve()


def _record_steps(total_steps: int, eval_every: int) -> list[int]:
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    steps = [0]
    steps.extend(step for step in range(eval_every, total_steps + 1, eval_every))
    if steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def _trace_step_requests(steps: int, progress_values: tuple[float, ...]) -> dict[int, list[float]]:
    if steps < 1:
        raise ValueError(f"sampling steps must be >= 1, got {steps}")
    requests: dict[int, list[float]] = {}
    for value in progress_values:
        progress = float(value)
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress values must be in [0, 1], got {progress}")
        step = min(range(steps), key=lambda index: abs((index / steps) - progress))
        requests.setdefault(step, []).append(progress)
    return requests


def _build_model(config: ProjectConfig, layout: TaskLayout, device: torch.device) -> UnifiedDenoiser:
    return UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        image_summary_to_text=config.model.image_summary_to_text,
    ).to(device)


def _load_source_model(
    config: ProjectConfig,
    device: torch.device,
) -> tuple[UnifiedDenoiser, dict, TaskLayout, dict]:
    source_path = _source_checkpoint_path(config)
    if not source_path.exists():
        raise FileNotFoundError(f"missing sampler-state source checkpoint at {source_path}")
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
    return model, tokenizer_state, layout, payload


def _set_trainable_scope(model: UnifiedDenoiser, scope: str) -> dict[str, int | str]:
    if scope not in TRAINABLE_SCOPES:
        raise ValueError(f"unsupported trainable scope {scope!r}; expected one of {TRAINABLE_SCOPES}")

    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")

    if scope in {"last_block", "last_two_blocks"}:
        layer_count = 1 if scope == "last_block" else 2
        for layer in list(model.transformer.layers)[-layer_count:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for module in (model.norm, model.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif scope == "head":
        for module in (model.norm, model.head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    trainable_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters == 0:
        raise ValueError(f"trainable scope {scope!r} leaves no trainable parameters")
    return {
        "trainable_scope": scope,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_tensors": trainable_tensors,
    }


@torch.inference_mode()
def _generate_i2t_sampler_states(
    *,
    teacher: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    image_tokens: torch.Tensor,
    progress_values: tuple[float, ...],
) -> list[SamplerStateBatch]:
    device = image_tokens.device
    batch_size = image_tokens.shape[0]
    steps = int(config.sampling.steps)
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    z_t = sample_masked_noise(
        torch.zeros(batch_size, layout.seq_len, layout.vocab_size, device=device),
        valid_token_mask,
    )
    clean_image_state = build_flm_clean_state(image_tokens, layout.vocab_size)
    z_t[:, layout.image_slice] = clean_image_state
    effective_text_time_power = config.sampling.text_time_power
    if config.sampling.image_to_text_text_time_power is not None:
        effective_text_time_power = config.sampling.image_to_text_text_time_power

    projection_progresses = (
        config.sampling.image_to_text_projection_progresses
        if config.sampling.image_to_text_projection_progresses is not None
        else [config.sampling.image_to_text_projection_progress]
    )
    projection_steps = (
        _projection_step_indices(steps, projection_progresses)
        if config.sampling.image_to_text_projection != "none"
        else set()
    )
    trace_requests = _trace_step_requests(steps, progress_values)
    states: list[SamplerStateBatch] = []

    for step in range(steps):
        progress = torch.full((batch_size,), step / steps, device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=config.sampling.image_time_power,
            text_time_power=effective_text_time_power,
        )
        t_pos = _apply_i2t_text_time_schedule(
            t_pos,
            progress,
            layout=layout,
            schedule=config.sampling.image_to_text_text_time_schedule,
            logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
            logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
        )
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=True,
            condition_text=False,
        )
        next_progress = torch.full((batch_size,), (step + 1) / steps, device=device)
        next_t_pos = apply_schedule(
            progress=next_progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=config.sampling.image_time_power,
            text_time_power=effective_text_time_power,
        )
        next_t_pos = _apply_i2t_text_time_schedule(
            next_t_pos,
            next_progress,
            layout=layout,
            schedule=config.sampling.image_to_text_text_time_schedule,
            logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
            logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
        )
        next_t_pos = condition_clean_timesteps(
            next_t_pos,
            layout.image_seq_len,
            condition_image=True,
            condition_text=False,
        )
        if config.sampling.integrator == "scheduled_euler":
            dt_pos = next_t_pos - t_pos
        else:
            dt_pos = torch.full_like(t_pos, 1.0 / steps)

        if step in trace_requests:
            for requested_progress in trace_requests[step]:
                states.append(
                    SamplerStateBatch(
                        progress=float(requested_progress),
                        z_t=z_t.detach().clone(),
                        t_pos=t_pos.detach().clone(),
                    )
                )

        logits = mask_logits(teacher(z_t, t_pos, modality_ids), layout=layout)
        probs = torch.softmax(logits / max(config.sampling.temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + dt_pos.unsqueeze(-1) * v_t
        z_t[:, layout.image_slice] = clean_image_state
        if config.sampling.image_to_text_projection != "none" and step in projection_steps:
            projected_targets = _project_text_state(
                logits[:, layout.text_slice],
                layout=layout,
                projection=config.sampling.image_to_text_projection,
                text_metadata=text_metadata,
                model=teacher,
                schedule_tables=schedule_tables,
                image_tokens=image_tokens,
                image_time_power=config.sampling.image_time_power,
                text_time_power=config.sampling.text_time_power,
                image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
                image_to_text_text_time_schedule=config.sampling.image_to_text_text_time_schedule,
                image_to_text_logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
                image_to_text_logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
                candidate_score_progress=config.sampling.image_to_text_candidate_score_progress,
                candidate_score_num_noise=config.sampling.image_to_text_candidate_score_num_noise,
                candidate_score_blend_weight=config.sampling.image_to_text_candidate_score_blend_weight,
            )
            projected_clean = build_flm_clean_state(projected_targets, layout.vocab_size)
            z_t[:, layout.text_slice] = mix_flm_noise(
                projected_clean,
                next_t_pos[:, layout.text_slice],
                valid_token_mask[layout.text_slice],
            )

    if len(states) != len(progress_values):
        raise RuntimeError(f"expected {len(progress_values)} sampler states, got {len(states)}")
    return states


def _masked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError(f"anchor temperature must be > 0, got {temperature}")
    valid_token_mask = valid_token_mask.to(device=student_logits.device, dtype=torch.bool)
    losses = []
    for position in range(student_logits.shape[1]):
        valid = valid_token_mask[position]
        student_position = student_logits[:, position, valid] / float(temperature)
        teacher_position = teacher_logits[:, position, valid] / float(temperature)
        losses.append(
            F.kl_div(
                F.log_softmax(student_position, dim=-1),
                F.softmax(teacher_position, dim=-1),
                reduction="batchmean",
            )
            * float(temperature) ** 2
        )
    return torch.stack(losses).mean()


def _anchor_kl_loss(
    *,
    teacher: UnifiedDenoiser,
    student: UnifiedDenoiser,
    config: ProjectConfig,
    x1: torch.Tensor,
    progress: torch.Tensor,
    layout: TaskLayout,
    modality_ids: torch.Tensor,
    schedule_tables: dict,
    valid_token_mask: torch.Tensor,
    anchor_tasks: tuple[str, ...],
    temperature: float,
) -> torch.Tensor:
    if not anchor_tasks:
        return torch.zeros((), device=x1.device)
    losses = []
    for task in anchor_tasks:
        if task not in {"joint", "text_to_image", "image_to_text"}:
            raise ValueError(f"unsupported anchor task {task!r}")
        t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, task)
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=task == "image_to_text",
            condition_text=task == "text_to_image",
        )
        z_t = _build_zt(x1, t_pos, layout, task, valid_token_mask)
        with torch.no_grad():
            teacher_logits = teacher(z_t, t_pos, modality_ids)
        student_logits = student(z_t, t_pos, modality_ids)
        losses.append(
            _masked_kl_divergence(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                valid_token_mask=valid_token_mask,
                temperature=temperature,
            )
        )
    return torch.stack(losses).mean()


def _true_label_sequence_scores(
    text_logits: torch.Tensor,
    text_targets: torch.Tensor,
    candidate_text_targets: torch.Tensor,
) -> torch.Tensor:
    scores = sequence_candidate_scores(text_logits, candidate_text_targets)
    matches = text_targets.unsqueeze(1).eq(candidate_text_targets.to(text_targets.device).unsqueeze(0)).all(dim=-1)
    if not torch.all(matches.any(dim=1)):
        raise ValueError("each text target row must exactly match one canonical text candidate")
    target_indices = matches.float().argmax(dim=1)
    return scores.gather(dim=1, index=target_indices[:, None].to(scores.device)).squeeze(1)


def _shuffled_image_contrast_loss(
    *,
    true_text_logits: torch.Tensor,
    shuffled_text_logits: torch.Tensor,
    text_targets: torch.Tensor,
    labels: torch.Tensor,
    shifted_labels: torch.Tensor,
    candidate_text_targets: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    valid = labels.ne(shifted_labels).to(true_text_logits.device)
    if not torch.any(valid):
        return torch.zeros((), device=true_text_logits.device)
    true_scores = _true_label_sequence_scores(true_text_logits, text_targets, candidate_text_targets)
    shuffled_scores = _true_label_sequence_scores(shuffled_text_logits, text_targets, candidate_text_targets)
    losses = F.relu(float(margin) + shuffled_scores - true_scores)
    return losses[valid].mean()


def _sampler_state_loss(
    *,
    student: UnifiedDenoiser,
    state: SamplerStateBatch,
    layout: TaskLayout,
    modality_ids: torch.Tensor,
    text_targets: torch.Tensor,
    text_pad_id: int,
    candidate_text_targets: torch.Tensor,
    sequence_weight: float,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    # Sampler traces are produced under inference_mode by the frozen teacher.
    # Clone them here so autograd can save the student input for backward.
    z_t = state.z_t.clone()
    t_pos = state.t_pos.clone()
    logits = mask_logits(student(z_t, t_pos, modality_ids), layout=layout)
    text_logits = logits[:, layout.text_slice]
    text_loss = _masked_text_loss(
        text_logits.reshape(-1, logits.shape[-1]),
        text_targets.reshape(-1),
        text_pad_token=layout.text_offset + text_pad_id,
    )
    sequence_loss = _sequence_text_loss(
        text_logits=text_logits,
        text_targets=text_targets,
        candidate_text_targets=candidate_text_targets,
    )
    loss = text_loss + float(sequence_weight) * sequence_loss
    return (
        loss,
        {
            "text_loss": float(text_loss.item()),
            "sequence_loss": float(sequence_loss.item()),
        },
        text_logits,
    )


def _save_sampler_state_checkpoint(
    *,
    config: ProjectConfig,
    student: UnifiedDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokenizer_state: dict,
    run_context: RunContext,
) -> Path:
    path = latest_checkpoint_path(config, "stage2")
    payload = {
        "stage": "stage2",
        "step": step,
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokenizer_state": tokenizer_state,
        "config_name": config.name,
        "source": "probe-i2t-sampler-state-ft",
    }
    torch.save(payload, path)
    torch.save(payload, run_context.log_path(f"checkpoints/stage2_step{step:06d}.pt"))
    return path


def run_i2t_sampler_state_ft(
    *,
    config: ProjectConfig,
    steps: int = 150,
    lr: float | None = None,
    progress_values: tuple[float, ...] | None = DEFAULT_SAMPLER_STATE_PROGRESS,
    contrast_weight: float = 0.0,
    contrast_margin: float = 1.0,
    sequence_weight: float | None = None,
    anchor_weight: float = 0.0,
    anchor_tasks: tuple[str, ...] | None = None,
    anchor_temperature: float = 1.0,
    trainable_scope: str = "all",
    save_every: int | None = None,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if contrast_weight < 0.0:
        raise ValueError(f"contrast_weight must be >= 0, got {contrast_weight}")
    if contrast_margin <= 0.0:
        raise ValueError(f"contrast_margin must be > 0, got {contrast_margin}")
    if anchor_weight < 0.0:
        raise ValueError(f"anchor_weight must be >= 0, got {anchor_weight}")
    if anchor_temperature <= 0.0:
        raise ValueError(f"anchor_temperature must be > 0, got {anchor_temperature}")
    effective_anchor_tasks = DEFAULT_ANCHOR_TASKS if anchor_tasks is None else anchor_tasks
    for task in effective_anchor_tasks:
        if task not in {"joint", "text_to_image", "image_to_text"}:
            raise ValueError(f"unsupported anchor task {task!r}")
    effective_progress_values = DEFAULT_SAMPLER_STATE_PROGRESS if progress_values is None else progress_values
    if not effective_progress_values:
        raise ValueError("progress_values must contain at least one value")
    for progress in effective_progress_values:
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError(f"progress values must be in [0, 1], got {progress}")

    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "probe-i2t-sampler-state-ft")
    run_context.set_device(device)
    teacher, tokenizer_state, layout, source_payload = _load_source_model(config, device)
    student = _build_model(config, layout, device)
    student.load_state_dict(source_payload["model"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    trainable_stats = _set_trainable_scope(student, trainable_scope)
    text_metadata = metadata_from_state(tokenizer_state)
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
    train_iter = iter(loader)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=config.train.lr if lr is None else float(lr),
        weight_decay=config.train.weight_decay,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    shifted_label_tokens = shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size).to(device)
    effective_sequence_weight = config.train.text_sequence_weight if sequence_weight is None else float(sequence_weight)
    effective_save_every = config.train.save_every if save_every is None else int(save_every)
    if effective_sequence_weight < 0.0:
        raise ValueError(f"sequence_weight must be >= 0, got {effective_sequence_weight}")
    if effective_save_every < 1:
        raise ValueError(f"save_every must be >= 1, got {effective_save_every}")

    train_log = run_context.log_path("sampler_state_train.jsonl")
    try:
        for step in tqdm(range(1, steps + 1), desc="i2t-sampler-state-ft"):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            image_tokens = batch["image_tokens"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)
            targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
            text_targets = targets[:, layout.text_slice]
            states = _generate_i2t_sampler_states(
                teacher=teacher,
                config=config,
                layout=layout,
                schedule_tables=schedule_tables,
                text_metadata=text_metadata,
                image_tokens=image_tokens,
                progress_values=tuple(float(value) for value in effective_progress_values),
            )

            loss = torch.zeros((), device=device)
            part_sums = {"text_loss": 0.0, "sequence_loss": 0.0, "contrast_loss": 0.0, "anchor_loss": 0.0}
            for state in states:
                state_loss, parts, true_text_logits = _sampler_state_loss(
                    student=student,
                    state=state,
                    layout=layout,
                    modality_ids=modality_ids,
                    text_targets=text_targets,
                    text_pad_id=text_metadata.pad_id,
                    candidate_text_targets=shifted_label_tokens,
                    sequence_weight=effective_sequence_weight,
                )
                loss = loss + state_loss
                part_sums["text_loss"] += parts["text_loss"]
                part_sums["sequence_loss"] += parts["sequence_loss"]
                if contrast_weight > 0.0:
                    order = torch.arange(image_tokens.shape[0], device=device).roll(1)
                    shuffled_state = state.z_t.clone()
                    shuffled_image_state = build_flm_clean_state(
                        image_tokens.index_select(0, order),
                        layout.vocab_size,
                    )
                    shuffled_state[:, layout.image_slice] = shuffled_image_state
                    shuffled_logits = mask_logits(
                        student(shuffled_state, state.t_pos.clone(), modality_ids),
                        layout=layout,
                    )
                    contrast_loss = _shuffled_image_contrast_loss(
                        true_text_logits=true_text_logits,
                        shuffled_text_logits=shuffled_logits[:, layout.text_slice],
                        text_targets=text_targets,
                        labels=labels,
                        shifted_labels=labels.index_select(0, order),
                        candidate_text_targets=shifted_label_tokens,
                        margin=contrast_margin,
                    )
                    loss = loss + float(contrast_weight) * contrast_loss
                    part_sums["contrast_loss"] += float(contrast_loss.item())
            loss = loss / float(len(states))
            if anchor_weight > 0.0:
                anchor_progress = torch.rand(image_tokens.shape[0], device=device)
                x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
                anchor_loss = _anchor_kl_loss(
                    teacher=teacher,
                    student=student,
                    config=config,
                    x1=x1,
                    progress=anchor_progress,
                    layout=layout,
                    modality_ids=modality_ids,
                    schedule_tables=schedule_tables,
                    valid_token_mask=valid_token_mask,
                    anchor_tasks=tuple(effective_anchor_tasks),
                    temperature=float(anchor_temperature),
                )
                loss = loss + float(anchor_weight) * anchor_loss
                part_sums["anchor_loss"] = float(anchor_loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=config.train.grad_clip_norm)
            optimizer.step()

            if step % config.train.log_every == 0 or step == 1:
                append_jsonl(
                    train_log,
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "progress_values": [float(value) for value in effective_progress_values],
                        "contrast_weight": float(contrast_weight),
                        "contrast_margin": float(contrast_margin),
                        "anchor_weight": float(anchor_weight),
                        "anchor_tasks": list(effective_anchor_tasks),
                        "anchor_temperature": float(anchor_temperature),
                        "trainable_scope": trainable_scope,
                        "text_loss": part_sums["text_loss"] / float(len(states)),
                        "sequence_loss": part_sums["sequence_loss"] / float(len(states)),
                        "contrast_loss": part_sums["contrast_loss"] / float(len(states)),
                        "anchor_loss": part_sums["anchor_loss"],
                    },
                )
            if step % effective_save_every == 0 or step == steps:
                _save_sampler_state_checkpoint(
                    config=config,
                    student=student,
                    optimizer=optimizer,
                    step=step,
                    tokenizer_state=tokenizer_state,
                    run_context=run_context,
                )

        summary = {
            "steps": steps,
            "lr": config.train.lr if lr is None else float(lr),
            "progress_values": [float(value) for value in effective_progress_values],
            "contrast_weight": float(contrast_weight),
            "contrast_margin": float(contrast_margin),
            "sequence_weight": float(effective_sequence_weight),
            "anchor_weight": float(anchor_weight),
            "anchor_tasks": list(effective_anchor_tasks),
            "anchor_temperature": float(anchor_temperature),
            **trainable_stats,
            "source_checkpoint": str(_source_checkpoint_path(config)),
            "final_checkpoint": str(latest_checkpoint_path(config, "stage2")),
        }
        run_context.log_path("sampler_state_ft_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        if own_context:
            run_context.update_status("ok")
        return summary
    except Exception:
        if own_context:
            run_context.update_status("error")
        raise
