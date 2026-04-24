from __future__ import annotations

import json

import torch
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .diagnostics import _prediction_records
from .eval import _load_stage2, _sample_unified_with_logits
from .layout import TaskLayout, mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps
from .text import metadata_from_state, shifted_label_text_tokens
from .train import (
    _apply_image_to_text_noise_policy,
    _build_zt,
    _loss_for_task,
    _task_time_schedule,
)


DEFAULT_OVERFIT_PROGRESS = (0.5, 0.75, 0.9, 0.95)


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


def _first_examples(config: ProjectConfig, split: str, sample_count: int) -> dict[str, torch.Tensor]:
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}")
    loader = build_loader(
        split_path(config, split),
        batch_size=min(sample_count, max(config.train.eval_batch_size, 1)),
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    chunks: dict[str, list[torch.Tensor]] = {"image_tokens": [], "text_tokens": [], "label": []}
    total = 0
    for batch in loader:
        take = min(sample_count - total, batch["label"].shape[0])
        for key in chunks:
            chunks[key].append(batch[key][:take].clone())
        total += take
        if total >= sample_count:
            break
    if total == 0:
        raise ValueError(f"split {split!r} is empty")
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _summarize_prediction_records(records: list[dict]) -> dict[str, float]:
    total = max(len(records), 1)
    return {
        "image_to_text_exact_match": sum(bool(record["free_exact"]) for record in records) / total,
        "image_to_text_token_accuracy": sum(float(record["token_accuracy"]) for record in records) / total,
        "image_to_text_label_accuracy_constrained": sum(bool(record["label_correct"]) for record in records) / total,
        "mean_true_label_margin": sum(float(record["true_label_margin"]) for record in records) / total,
    }


@torch.inference_mode()
def _direct_metrics(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    tokenizer_state: dict,
    batch: dict[str, torch.Tensor],
    progress_value: float,
    mode: str,
) -> dict:
    device = next(model.parameters()).device
    text_metadata = metadata_from_state(tokenizer_state)
    image_tokens = _image_condition(batch["image_tokens"], mode=mode, codebook_size=layout.codebook_size)
    text_tokens = batch["text_tokens"]
    labels = batch["label"]
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    progress = torch.full((labels.shape[0],), float(progress_value), device=device)
    t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=True,
        condition_text=False,
    )
    t_pos = _apply_image_to_text_noise_policy(
        t_pos,
        layout,
        "image_to_text",
        text_time_cap=config.train.image_to_text_text_time_cap,
        noise_only_prob=config.train.image_to_text_noise_only_prob,
    )
    z_t = _build_zt(x1, t_pos, layout, "image_to_text", valid_token_mask)
    logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
    records, summary = _prediction_records(
        text_logits=logits[:, layout.text_slice],
        text_tokens=text_tokens,
        labels=labels,
        text_metadata=text_metadata,
        layout=layout,
    )
    return {
        "style": "direct",
        "mode": mode,
        "progress": float(progress_value),
        "summary": summary,
        "records": records,
    }


def _image_condition(image_tokens: torch.Tensor, *, mode: str, codebook_size: int) -> torch.Tensor:
    if mode == "true_image":
        return image_tokens
    if mode == "shuffled_image":
        order = torch.arange(image_tokens.shape[0], device=image_tokens.device).roll(1)
        return image_tokens.index_select(0, order)
    if mode == "random_image_tokens":
        return torch.randint(
            low=0,
            high=codebook_size,
            size=image_tokens.shape,
            device=image_tokens.device,
        )
    raise ValueError(f"unsupported image condition mode: {mode}")


@torch.inference_mode()
def _sampler_metrics(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    tokenizer_state: dict,
    batch: dict[str, torch.Tensor],
    mode: str,
) -> dict:
    text_metadata = metadata_from_state(tokenizer_state)
    image_tokens = _image_condition(batch["image_tokens"], mode=mode, codebook_size=layout.codebook_size)
    _, logits = _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        image_to_text_text_time_schedule=config.sampling.image_to_text_text_time_schedule,
        image_to_text_logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
        image_to_text_logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
        integrator=config.sampling.integrator,
        final_decode=config.sampling.final_decode,
        final_model_progress=config.sampling.final_model_progress,
        image_to_text_projection=config.sampling.image_to_text_projection,
        image_to_text_projection_progress=config.sampling.image_to_text_projection_progress,
        image_to_text_projection_progresses=config.sampling.image_to_text_projection_progresses,
        image_to_text_candidate_score_progress=config.sampling.image_to_text_candidate_score_progress,
        image_to_text_candidate_score_num_noise=config.sampling.image_to_text_candidate_score_num_noise,
        image_to_text_candidate_score_blend_weight=config.sampling.image_to_text_candidate_score_blend_weight,
        text_metadata=text_metadata,
        condition_image_tokens=image_tokens,
    )
    records, summary = _prediction_records(
        text_logits=logits[:, layout.text_slice],
        text_tokens=batch["text_tokens"],
        labels=batch["label"],
        text_metadata=text_metadata,
        layout=layout,
    )
    return {
        "style": "sampler",
        "mode": mode,
        "progress": "final",
        "summary": summary,
        "records": records,
    }


def _evaluate_probe(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    tokenizer_state: dict,
    train_batch: dict[str, torch.Tensor],
    test_batch: dict[str, torch.Tensor],
    progress_values: tuple[float, ...],
) -> dict:
    model.eval()
    result: dict[str, list[dict]] = {"train_fixed": [], "test_fixed": []}
    for split_name, batch in (("train_fixed", train_batch), ("test_fixed", test_batch)):
        for mode in ("true_image", "shuffled_image"):
            result[split_name].append(
                _sampler_metrics(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    tokenizer_state=tokenizer_state,
                    batch=batch,
                    mode=mode,
                )
            )
        for progress in progress_values:
            for mode in ("true_image", "shuffled_image"):
                result[split_name].append(
                    _direct_metrics(
                        model=model,
                        config=config,
                        layout=layout,
                        schedule_tables=schedule_tables,
                        tokenizer_state=tokenizer_state,
                        batch=batch,
                        progress_value=progress,
                        mode=mode,
                    )
                )
    model.train()
    return result


def _train_step(
    *,
    model: UnifiedDenoiser,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    batch: dict[str, torch.Tensor],
    text_metadata,
    mismatch_weight: float,
    label_weight: float,
    label_text_time: float,
) -> dict[str, float]:
    image_tokens = batch["image_tokens"]
    text_tokens = batch["text_tokens"]
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size).to(image_tokens.device)
    modality_ids = layout.position_modalities().to(image_tokens.device)
    valid_token_mask = layout.position_valid_token_mask().to(image_tokens.device)
    progress = torch.rand(image_tokens.shape[0], device=image_tokens.device)
    t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=True,
        condition_text=False,
    )
    t_pos = _apply_image_to_text_noise_policy(
        t_pos,
        layout,
        "image_to_text",
        text_time_cap=config.train.image_to_text_text_time_cap,
        noise_only_prob=config.train.image_to_text_noise_only_prob,
    )
    z_t = _build_zt(x1, t_pos, layout, "image_to_text", valid_token_mask)
    logits = model(z_t, t_pos, modality_ids)
    loss, parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=config.train.joint_weight,
        text_weight=config.train.text_weight,
        text_pad_id=text_metadata.pad_id,
        label_text_tokens=shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size),
        text_sequence_weight=config.train.text_sequence_weight,
        task="image_to_text",
        model=model,
        x1=x1,
        t_pos=t_pos,
        modality_ids=modality_ids,
        valid_token_mask=valid_token_mask,
        image_to_text_mismatch_weight=mismatch_weight,
        image_to_text_mismatch_margin=config.train.image_to_text_mismatch_margin,
        image_to_text_label_weight=label_weight,
        image_to_text_label_text_time=label_text_time,
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.item()),
        "image_t_mean": float(t_pos[:, layout.image_slice].mean().item()),
        "text_t_mean": float(t_pos[:, layout.text_slice].mean().item()),
        **parts,
    }


def run_i2t_overfit_probe(
    *,
    config: ProjectConfig,
    steps: int = 100,
    sample_count: int = 16,
    test_sample_count: int = 16,
    lr: float | None = None,
    eval_every: int = 25,
    progress_values: tuple[float, ...] | None = DEFAULT_OVERFIT_PROGRESS,
    mismatch_weight: float | None = None,
    label_weight: float | None = None,
    label_text_time: float | None = None,
    save_model: bool = False,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if sample_count < 2:
        raise ValueError("sample_count must be >= 2 so shuffled-image controls are meaningful")
    if test_sample_count < 2:
        raise ValueError("test_sample_count must be >= 2 so shuffled-image controls are meaningful")
    effective_progress_values = DEFAULT_OVERFIT_PROGRESS if progress_values is None else progress_values
    for progress in effective_progress_values:
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError(f"progress values must be in [0, 1], got {progress}")

    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "probe-i2t-overfit")
    run_context.set_device(device)
    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    train_batch = _move_batch(_first_examples(config, "train", sample_count), device)
    test_batch = _move_batch(_first_examples(config, "test", test_sample_count), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr if lr is None else float(lr),
        weight_decay=config.train.weight_decay,
    )
    effective_mismatch_weight = (
        config.train.image_to_text_mismatch_weight if mismatch_weight is None else float(mismatch_weight)
    )
    effective_label_weight = config.train.image_to_text_label_weight if label_weight is None else float(label_weight)
    effective_label_text_time = (
        config.train.image_to_text_label_text_time if label_text_time is None else float(label_text_time)
    )
    if effective_mismatch_weight < 0.0:
        raise ValueError(f"mismatch_weight must be >= 0, got {effective_mismatch_weight}")
    if effective_label_weight < 0.0:
        raise ValueError(f"label_weight must be >= 0, got {effective_label_weight}")
    if not 0.0 <= effective_label_text_time <= 1.0:
        raise ValueError(f"label_text_time must be in [0, 1], got {effective_label_text_time}")

    train_log = run_context.log_path("overfit_train.jsonl")
    eval_log = run_context.log_path("overfit_eval.jsonl")
    eval_steps = set(_record_steps(steps, eval_every))
    history = []
    try:
        for step in tqdm(range(0, steps + 1), desc="i2t-overfit"):
            if step in eval_steps:
                snapshot = {
                    "step": step,
                    "metrics": _evaluate_probe(
                        model=model,
                        config=config,
                        layout=layout,
                        schedule_tables=schedule_tables,
                        tokenizer_state=tokenizer_state,
                        train_batch=train_batch,
                        test_batch=test_batch,
                        progress_values=tuple(float(value) for value in effective_progress_values),
                    ),
                }
                history.append(snapshot)
                append_jsonl(eval_log, snapshot)
            if step == steps:
                break
            parts = _train_step(
                model=model,
                optimizer=optimizer,
                config=config,
                layout=layout,
                schedule_tables=schedule_tables,
                batch=train_batch,
                text_metadata=text_metadata,
                mismatch_weight=effective_mismatch_weight,
                label_weight=effective_label_weight,
                label_text_time=effective_label_text_time,
            )
            append_jsonl(train_log, {"step": step + 1, **parts})

        summary = {
            "steps": steps,
            "sample_count": sample_count,
            "test_sample_count": test_sample_count,
            "lr": config.train.lr if lr is None else float(lr),
            "mismatch_weight": effective_mismatch_weight,
            "label_weight": effective_label_weight,
            "label_text_time": effective_label_text_time,
            "progress_values": [float(value) for value in effective_progress_values],
            "initial": history[0],
            "final": history[-1],
            "history": history,
        }
        summary_path = run_context.log_path("overfit_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if save_model:
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer_state": tokenizer_state,
                    "source_config": config.name,
                    "steps": steps,
                },
                run_context.log_path("overfit_final.pt"),
            )
        if own_context:
            run_context.update_status("ok")
        return summary
    except Exception:
        if own_context:
            run_context.update_status("error")
        raise
