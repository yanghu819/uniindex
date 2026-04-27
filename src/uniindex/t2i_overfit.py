from __future__ import annotations

import json
from collections import Counter

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _load_stage2, sample_unified
from .i2t_overfit import _first_examples, _move_batch, _record_steps
from .label_feature_probe import VQTokenLabelProbe, labels_to_class_indices
from .layout import TaskLayout, mask_logits, unified_targets
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise
from .t2i_token_guard import confusion_matrix
from .text import label_values_from_text_tokens, metadata_from_state
from .train import _loss_for_task, _task_time_schedule


DEFAULT_T2I_OVERFIT_PROGRESS = (0.25, 0.5, 0.75, 0.9)


def _jsonable_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _label_values_from_class_indices(indices: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    return label_values.to(indices.device).index_select(0, indices.long())


def _train_token_probe(
    *,
    config: ProjectConfig,
    layout: TaskLayout,
    label_values: torch.Tensor,
    steps: int,
) -> tuple[VQTokenLabelProbe, float]:
    if steps < 1:
        raise ValueError(f"probe_steps must be >= 1, got {steps}")
    device = label_values.device
    probe = VQTokenLabelProbe(
        codebook_size=layout.codebook_size,
        seq_len=layout.image_seq_len,
        num_classes=int(label_values.numel()),
    ).to(device)
    probe.train()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=config.i2t_llm.lr, weight_decay=config.train.weight_decay)
    loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    batches = list(loader)
    if not batches:
        raise ValueError("training split is empty")
    for step in range(steps):
        batch = batches[step % len(batches)]
        image_tokens = batch["image_tokens"].to(device)
        targets = labels_to_class_indices(batch["label"].to(device), label_values)
        logits = probe(image_tokens)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()

    probe.eval()
    correct = 0
    total = 0
    eval_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    for batch in eval_loader:
        image_tokens = batch["image_tokens"].to(device)
        targets = labels_to_class_indices(batch["label"].to(device), label_values)
        pred = probe(image_tokens).argmax(dim=1)
        correct += int(pred.eq(targets).sum().item())
        total += int(targets.numel())
    return probe, correct / max(total, 1)


def _image_ce_loss(logits: torch.Tensor, targets: torch.Tensor, layout: TaskLayout) -> torch.Tensor:
    image_logits = logits[:, layout.image_slice, : layout.codebook_size]
    image_targets = targets[:, layout.image_slice].long()
    return F.cross_entropy(image_logits.reshape(-1, layout.codebook_size), image_targets.reshape(-1))


def _image_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, layout: TaskLayout) -> float:
    pred = logits[:, layout.image_slice, : layout.codebook_size].argmax(dim=-1)
    actual = targets[:, layout.image_slice].long()
    return float(pred.eq(actual).float().mean().detach().cpu().item())


def _direct_t2i_metrics(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    batch: dict[str, torch.Tensor],
    progress_values: tuple[float, ...],
) -> list[dict]:
    device = next(model.parameters()).device
    image_tokens = batch["image_tokens"]
    text_tokens = batch["text_tokens"]
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    records = []
    for progress_value in progress_values:
        progress = torch.full((image_tokens.shape[0],), float(progress_value), device=device)
        t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "text_to_image")
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=False,
            condition_text=True,
        )
        z_t = x1.clone()
        z_t[:, layout.image_slice] = mix_flm_noise(
            x1[:, layout.image_slice],
            t_pos[:, layout.image_slice],
            valid_token_mask[layout.image_slice],
        )
        logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
        records.append(
            {
                "progress": float(progress_value),
                "image_ce": float(_image_ce_loss(logits, targets, layout).detach().cpu().item()),
                "image_token_accuracy": _image_token_accuracy(logits, targets, layout),
            }
        )
    return records


def _sample_conditioned_images(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    text_tokens: torch.Tensor,
) -> torch.Tensor:
    return sample_unified(
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
        condition_image_tokens=None,
        condition_text_tokens=text_tokens,
    )[:, layout.image_slice]


@torch.no_grad()
def _sampler_t2i_metrics(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    batch: dict[str, torch.Tensor],
    token_probe: VQTokenLabelProbe,
    label_values: torch.Tensor,
) -> dict:
    sampled_images = _sample_conditioned_images(
        model=model,
        config=config,
        layout=layout,
        schedule_tables=schedule_tables,
        text_metadata=text_metadata,
        text_tokens=batch["text_tokens"],
    )
    generated_pred_indices = token_probe(sampled_images).argmax(dim=1).detach().cpu()
    target_indices = labels_to_class_indices(batch["label"], label_values.detach().cpu()).detach().cpu()
    target_values = batch["label"].detach().cpu().long()
    pred_values = _label_values_from_class_indices(generated_pred_indices, label_values.detach().cpu())
    matrix = confusion_matrix(generated_pred_indices, target_indices, num_classes=int(label_values.numel()))
    return {
        "token_label_accuracy": float(generated_pred_indices.eq(target_indices).float().mean().item()),
        "token_pred_histogram": _jsonable_counter(Counter(pred_values.tolist())),
        "target_histogram": _jsonable_counter(Counter(target_values.tolist())),
        "confusion": matrix.tolist(),
        "generated_unique_token_count": int(sampled_images.unique().numel()),
        "generated_avg_unique_tokens_per_sample": float(
            torch.tensor([row.unique().numel() for row in sampled_images.detach().cpu()]).float().mean().item()
        ),
    }


@torch.no_grad()
def _unconditional_metrics(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    token_probe: VQTokenLabelProbe,
    label_values: torch.Tensor,
    count: int,
) -> dict:
    if count <= 0:
        return {}
    sampled = sample_unified(
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
        batch_size=count,
        condition_image_tokens=None,
        condition_text_tokens=None,
    )
    image_tokens = sampled[:, layout.image_slice]
    text_tokens = sampled[:, layout.text_slice] - layout.text_offset
    pred_indices = token_probe(image_tokens).argmax(dim=1).detach().cpu()
    pred_values = _label_values_from_class_indices(pred_indices, label_values.detach().cpu())
    text_values = label_values_from_text_tokens(text_tokens, text_metadata).detach().cpu()
    return {
        "unconditional_token_text_consistency": float(pred_values.eq(text_values).float().mean().item()),
        "unconditional_token_pred_histogram": _jsonable_counter(Counter(pred_values.tolist())),
        "unconditional_text_value_histogram": _jsonable_counter(Counter(text_values.tolist())),
        "unconditional_unique_token_count": int(image_tokens.unique().numel()),
    }


def _evaluate_probe(
    *,
    model: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    text_metadata,
    train_batch: dict[str, torch.Tensor],
    test_batch: dict[str, torch.Tensor],
    token_probe: VQTokenLabelProbe,
    label_values: torch.Tensor,
    progress_values: tuple[float, ...],
    unconditional_count: int,
) -> dict:
    model.eval()
    with torch.no_grad():
        result = {
            "train_fixed": {
                "direct": _direct_t2i_metrics(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    batch=train_batch,
                    progress_values=progress_values,
                ),
                "sampler": _sampler_t2i_metrics(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    text_metadata=text_metadata,
                    batch=train_batch,
                    token_probe=token_probe,
                    label_values=label_values,
                ),
            },
            "test_fixed": {
                "direct": _direct_t2i_metrics(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    batch=test_batch,
                    progress_values=progress_values,
                ),
                "sampler": _sampler_t2i_metrics(
                    model=model,
                    config=config,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    text_metadata=text_metadata,
                    batch=test_batch,
                    token_probe=token_probe,
                    label_values=label_values,
                ),
            },
            "unconditional": _unconditional_metrics(
                model=model,
                config=config,
                layout=layout,
                schedule_tables=schedule_tables,
                text_metadata=text_metadata,
                token_probe=token_probe,
                label_values=label_values,
                count=unconditional_count,
            ),
        }
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
) -> dict[str, float]:
    image_tokens = batch["image_tokens"]
    text_tokens = batch["text_tokens"]
    targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
    x1 = build_flm_clean_state(targets, layout.vocab_size).to(image_tokens.device)
    modality_ids = layout.position_modalities().to(image_tokens.device)
    valid_token_mask = layout.position_valid_token_mask().to(image_tokens.device)
    progress = torch.rand(image_tokens.shape[0], device=image_tokens.device)
    t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "text_to_image")
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=False,
        condition_text=True,
    )
    z_t = x1.clone()
    z_t[:, layout.image_slice] = mix_flm_noise(
        x1[:, layout.image_slice],
        t_pos[:, layout.image_slice],
        valid_token_mask[layout.image_slice],
    )
    logits = model(z_t, t_pos, modality_ids)
    loss, parts = _loss_for_task(
        logits=logits,
        targets=targets,
        layout=layout,
        joint_weight=config.train.joint_weight,
        text_weight=config.train.text_weight,
        text_pad_id=text_metadata.pad_id,
        label_text_tokens=torch.empty(0, dtype=torch.long, device=image_tokens.device),
        text_sequence_weight=0.0,
        task="text_to_image",
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_clip_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu().item()),
        "image_t_mean": float(t_pos[:, layout.image_slice].mean().detach().cpu().item()),
        "text_t_mean": float(t_pos[:, layout.text_slice].mean().detach().cpu().item()),
        "image_token_accuracy": _image_token_accuracy(mask_logits(logits, layout=layout), targets, layout),
        **parts,
    }


def run_t2i_overfit_probe(
    *,
    config: ProjectConfig,
    steps: int = 200,
    sample_count: int = 16,
    test_sample_count: int = 16,
    lr: float | None = None,
    eval_every: int = 50,
    progress_values: tuple[float, ...] | None = DEFAULT_T2I_OVERFIT_PROGRESS,
    token_probe_steps: int = 100,
    unconditional_count: int = 8,
    save_model: bool = False,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}")
    if test_sample_count < 1:
        raise ValueError(f"test_sample_count must be >= 1, got {test_sample_count}")
    if unconditional_count < 0:
        raise ValueError(f"unconditional_count must be >= 0, got {unconditional_count}")
    effective_progress_values = DEFAULT_T2I_OVERFIT_PROGRESS if progress_values is None else progress_values
    for progress in effective_progress_values:
        if not 0.0 <= float(progress) <= 1.0:
            raise ValueError(f"progress values must be in [0, 1], got {progress}")

    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "probe-t2i-overfit")
    run_context.set_device(device)
    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=device)
    token_probe, token_probe_accuracy = _train_token_probe(
        config=config,
        layout=layout,
        label_values=label_values,
        steps=token_probe_steps,
    )
    train_batch = _move_batch(_first_examples(config, "train", sample_count), device)
    test_batch = _move_batch(_first_examples(config, "test", test_sample_count), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr if lr is None else float(lr),
        weight_decay=config.train.weight_decay,
    )

    train_log = run_context.log_path("t2i_overfit_train.jsonl")
    eval_log = run_context.log_path("t2i_overfit_eval.jsonl")
    eval_steps = set(_record_steps(steps, eval_every))
    history = []
    try:
        for step in tqdm(range(0, steps + 1), desc="t2i-overfit"):
            if step in eval_steps:
                snapshot = {
                    "step": step,
                    "metrics": _evaluate_probe(
                        model=model,
                        config=config,
                        layout=layout,
                        schedule_tables=schedule_tables,
                        text_metadata=text_metadata,
                        train_batch=train_batch,
                        test_batch=test_batch,
                        token_probe=token_probe,
                        label_values=label_values,
                        progress_values=tuple(float(value) for value in effective_progress_values),
                        unconditional_count=unconditional_count,
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
            )
            append_jsonl(train_log, {"step": step + 1, **parts})

        summary = {
            "steps": steps,
            "sample_count": sample_count,
            "test_sample_count": test_sample_count,
            "lr": config.train.lr if lr is None else float(lr),
            "progress_values": [float(value) for value in effective_progress_values],
            "token_probe_steps": int(token_probe_steps),
            "token_probe_test_real_accuracy": float(token_probe_accuracy),
            "unconditional_count": int(unconditional_count),
            "initial": history[0],
            "final": history[-1],
            "history": history,
        }
        summary_path = run_context.log_path("t2i_overfit_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if save_model:
            torch.save(
                {
                    "model": model.state_dict(),
                    "tokenizer_state": tokenizer_state,
                    "source_config": config.name,
                    "steps": steps,
                },
                run_context.log_path("t2i_overfit_final.pt"),
            )
        if own_context:
            run_context.update_status("ok")
        return summary
    except Exception:
        if own_context:
            run_context.update_status("error")
        raise
