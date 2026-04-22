from __future__ import annotations

import json
from collections import Counter

import torch
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _load_stage2, constrained_text_label_values
from .layout import mask_logits, unified_targets
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise, sample_masked_noise
from .text import decode_text_tokens, metadata_from_state, text_scoring_mask
from .train import _task_time_schedule


DEFAULT_I2T_PROGRESS = (0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_I2T_TRAJECTORY_PROGRESS = (0.5, 0.75, 0.9, 0.95)
IMAGE_DEPENDENCE_MODES = ("true_image", "shuffled_image", "random_image_tokens")


def _image_condition_tokens(
    image_tokens: torch.Tensor,
    *,
    mode: str,
    codebook_size: int,
) -> torch.Tensor:
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
    raise ValueError(f"unsupported image-dependence mode: {mode}")


def _image_dependence_margins(results: list[dict]) -> list[dict]:
    by_progress = {}
    for result in results:
        by_progress.setdefault(float(result["progress"]), {})[result["mode"]] = result

    margins = []
    for progress, modes in sorted(by_progress.items()):
        true_result = modes.get("true_image")
        shuffled = modes.get("shuffled_image")
        random_tokens = modes.get("random_image_tokens")
        if true_result is None:
            continue
        record = {"progress": progress}
        if shuffled is not None:
            record["true_minus_shuffled_label"] = (
                true_result["image_to_text_label_accuracy_constrained"]
                - shuffled["image_to_text_label_accuracy_constrained"]
            )
            record["true_minus_shuffled_token"] = (
                true_result["image_to_text_token_accuracy"] - shuffled["image_to_text_token_accuracy"]
            )
            record["true_minus_shuffled_exact"] = (
                true_result["image_to_text_exact_match"] - shuffled["image_to_text_exact_match"]
            )
        if random_tokens is not None:
            record["true_minus_random_label"] = (
                true_result["image_to_text_label_accuracy_constrained"]
                - random_tokens["image_to_text_label_accuracy_constrained"]
            )
            record["true_minus_random_token"] = (
                true_result["image_to_text_token_accuracy"] - random_tokens["image_to_text_token_accuracy"]
            )
            record["true_minus_random_exact"] = (
                true_result["image_to_text_exact_match"] - random_tokens["image_to_text_exact_match"]
            )
        margins.append(record)
    return margins


def _trace_step_requests(steps: int, progress_values: tuple[float, ...]) -> dict[int, list[float]]:
    if steps < 1:
        raise ValueError(f"sampling steps must be >= 1, got {steps}")
    requests: dict[int, list[float]] = {}
    for value in progress_values:
        progress = float(value)
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"trace progress values must be in [0, 1], got {progress}")
        step = min(range(steps), key=lambda index: abs((index / steps) - progress))
        requests.setdefault(step, []).append(progress)
    return requests


def _new_text_metric_accumulator() -> dict:
    return {
        "exact": 0,
        "token_correct": 0,
        "token_total": 0,
        "constrained_correct": 0,
        "total": 0,
    }


def _update_text_metric_accumulator(
    accumulator: dict,
    *,
    text_logits: torch.Tensor,
    text_tokens: torch.Tensor,
    labels: torch.Tensor,
    text_metadata,
    layout,
) -> None:
    sampled_text = text_logits.argmax(dim=-1) - layout.text_offset
    sampled_strings = decode_text_tokens(sampled_text, text_metadata)
    target_strings = decode_text_tokens(text_tokens, text_metadata)
    valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
    constrained_values = constrained_text_label_values(
        text_logits,
        text_metadata,
        codebook_size=layout.codebook_size,
    )
    accumulator["exact"] += sum(pred == target for pred, target in zip(sampled_strings, target_strings))
    accumulator["token_correct"] += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
    accumulator["token_total"] += valid_text.sum().item()
    accumulator["constrained_correct"] += (constrained_values == labels).sum().item()
    accumulator["total"] += labels.shape[0]


def _finalize_text_metric_accumulator(accumulator: dict) -> dict:
    total = max(accumulator["total"], 1)
    token_total = max(accumulator["token_total"], 1)
    return {
        "image_to_text_exact_match": accumulator["exact"] / total,
        "image_to_text_token_accuracy": accumulator["token_correct"] / token_total,
        "image_to_text_label_accuracy_constrained": accumulator["constrained_correct"] / total,
    }


@torch.inference_mode()
def diagnose_i2t_denoiser(
    config: ProjectConfig,
    progress_values: tuple[float, ...] = DEFAULT_I2T_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)

    results = []
    for progress_value in progress_values:
        exact = 0
        token_correct = 0
        token_total = 0
        constrained_correct = 0
        position_correct = torch.zeros(layout.text_seq_len, dtype=torch.long)
        position_total = torch.zeros(layout.text_seq_len, dtype=torch.long)
        generated_counter: Counter[str] = Counter()
        total = 0
        image_t_sum = 0.0
        text_t_sum = 0.0

        for batch in tqdm(test_loader, desc=f"diagnose-i2t@{progress_value:g}"):
            image_tokens = batch["image_tokens"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)
            batch_size = labels.shape[0]

            targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
            x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
            progress = torch.full((batch_size,), float(progress_value), device=device)
            t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
            t_pos = condition_clean_timesteps(
                t_pos,
                layout.image_seq_len,
                condition_image=True,
                condition_text=False,
            )

            z_t = x1.clone()
            z_t[:, layout.text_slice] = mix_flm_noise(
                x1[:, layout.text_slice],
                t_pos[:, layout.text_slice],
                valid_token_mask[layout.text_slice],
            )
            logits = model(z_t, t_pos, modality_ids)
            logits = mask_logits(logits, layout=layout)
            text_logits = logits[:, layout.text_slice]
            sampled_text = text_logits.argmax(dim=-1) - layout.text_offset

            sampled_strings = decode_text_tokens(sampled_text, text_metadata)
            target_strings = decode_text_tokens(text_tokens, text_metadata)
            exact += sum(pred == target for pred, target in zip(sampled_strings, target_strings))
            valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
            token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
            token_total += valid_text.sum().item()
            position_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum(dim=0).cpu()
            position_total += valid_text.sum(dim=0).cpu()
            generated_counter.update(sampled_strings)
            constrained_values = constrained_text_label_values(
                text_logits,
                text_metadata,
                codebook_size=layout.codebook_size,
            )
            constrained_correct += (constrained_values == labels).sum().item()
            total += batch_size
            image_t_sum += float(t_pos[:, layout.image_slice].mean().item()) * batch_size
            text_t_sum += float(t_pos[:, layout.text_slice].mean().item()) * batch_size

        results.append(
            {
                "progress": float(progress_value),
                "image_t_mean": image_t_sum / max(total, 1),
                "text_t_mean": text_t_sum / max(total, 1),
                "image_to_text_exact_match": exact / max(total, 1),
                "image_to_text_token_accuracy": token_correct / max(token_total, 1),
                "image_to_text_label_accuracy_constrained": constrained_correct / max(total, 1),
                "image_to_text_position_accuracy": [
                    correct / max(total_count, 1)
                    for correct, total_count in zip(position_correct.tolist(), position_total.tolist())
                ],
                "image_to_text_generated_text_counts": dict(generated_counter.most_common(32)),
            }
        )

    summary = {
        "progress_values": [float(value) for value in progress_values],
        "results": results,
        "label_strings": list(text_metadata.label_strings),
    }
    path = run_context.log_path("i2t_diagnostics.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary


@torch.inference_mode()
def diagnose_i2t_image_dependence(
    config: ProjectConfig,
    progress_values: tuple[float, ...] = DEFAULT_I2T_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t-image-dependence")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)

    results = []
    for progress_value in progress_values:
        for mode in IMAGE_DEPENDENCE_MODES:
            exact = 0
            token_correct = 0
            token_total = 0
            constrained_correct = 0
            total = 0
            image_t_sum = 0.0
            text_t_sum = 0.0

            for batch in tqdm(test_loader, desc=f"diagnose-i2t-image-dependence/{mode}@{progress_value:g}"):
                image_tokens = batch["image_tokens"].to(device)
                text_tokens = batch["text_tokens"].to(device)
                labels = batch["label"].to(device)
                batch_size = labels.shape[0]

                condition_image_tokens = _image_condition_tokens(
                    image_tokens,
                    mode=mode,
                    codebook_size=layout.codebook_size,
                )
                targets = unified_targets(condition_image_tokens, text_tokens, layout.codebook_size)
                x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
                progress = torch.full((batch_size,), float(progress_value), device=device)
                t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
                t_pos = condition_clean_timesteps(
                    t_pos,
                    layout.image_seq_len,
                    condition_image=True,
                    condition_text=False,
                )

                z_t = x1.clone()
                z_t[:, layout.text_slice] = mix_flm_noise(
                    x1[:, layout.text_slice],
                    t_pos[:, layout.text_slice],
                    valid_token_mask[layout.text_slice],
                )
                logits = model(z_t, t_pos, modality_ids)
                logits = mask_logits(logits, layout=layout)
                text_logits = logits[:, layout.text_slice]
                sampled_text = text_logits.argmax(dim=-1) - layout.text_offset

                sampled_strings = decode_text_tokens(sampled_text, text_metadata)
                target_strings = decode_text_tokens(text_tokens, text_metadata)
                exact += sum(pred == target for pred, target in zip(sampled_strings, target_strings))
                valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
                token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
                token_total += valid_text.sum().item()
                constrained_values = constrained_text_label_values(
                    text_logits,
                    text_metadata,
                    codebook_size=layout.codebook_size,
                )
                constrained_correct += (constrained_values == labels).sum().item()
                total += batch_size
                image_t_sum += float(t_pos[:, layout.image_slice].mean().item()) * batch_size
                text_t_sum += float(t_pos[:, layout.text_slice].mean().item()) * batch_size

            results.append(
                {
                    "progress": float(progress_value),
                    "mode": mode,
                    "image_t_mean": image_t_sum / max(total, 1),
                    "text_t_mean": text_t_sum / max(total, 1),
                    "image_to_text_exact_match": exact / max(total, 1),
                    "image_to_text_token_accuracy": token_correct / max(token_total, 1),
                    "image_to_text_label_accuracy_constrained": constrained_correct / max(total, 1),
                }
            )

    summary = {
        "progress_values": [float(value) for value in progress_values],
        "modes": list(IMAGE_DEPENDENCE_MODES),
        "results": results,
        "margins": _image_dependence_margins(results),
        "label_strings": list(text_metadata.label_strings),
    }
    path = run_context.log_path("i2t_image_dependence.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary


@torch.inference_mode()
def diagnose_i2t_sampler_trajectory(
    config: ProjectConfig,
    progress_values: tuple[float, ...] = DEFAULT_I2T_TRAJECTORY_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t-sampler-trajectory")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    steps = int(config.sampling.steps)
    trace_requests = _trace_step_requests(steps, progress_values)
    effective_text_time_power = config.sampling.text_time_power
    if config.sampling.image_to_text_text_time_power is not None:
        effective_text_time_power = config.sampling.image_to_text_text_time_power

    accumulators: dict[tuple[str, float, float, str], dict] = {}

    def update_point(
        *,
        trace_kind: str,
        requested_progress: float,
        sampler_progress: float,
        mode: str,
        text_logits: torch.Tensor,
        text_tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        key = (trace_kind, requested_progress, sampler_progress, mode)
        accumulator = accumulators.setdefault(key, _new_text_metric_accumulator())
        _update_text_metric_accumulator(
            accumulator,
            text_logits=text_logits,
            text_tokens=text_tokens,
            labels=labels,
            text_metadata=text_metadata,
            layout=layout,
        )

    for batch in tqdm(test_loader, desc="diagnose-i2t-sampler-trajectory"):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)
        batch_size = labels.shape[0]
        z_t = sample_masked_noise(
            torch.zeros(batch_size, layout.seq_len, layout.vocab_size, device=device),
            valid_token_mask,
        )
        true_image_state = build_flm_clean_state(image_tokens, layout.vocab_size)
        z_t[:, layout.image_slice] = true_image_state

        last_logits = None
        last_progress = 0.0
        last_t_pos = None
        for step in range(steps):
            sampler_progress = step / steps
            progress = torch.full((batch_size,), sampler_progress, device=device)
            t_pos = apply_schedule(
                progress=progress,
                modality_ids=modality_ids,
                schedule_tables=schedule_tables,
                image_time_power=config.sampling.image_time_power,
                text_time_power=effective_text_time_power,
            )
            t_pos = condition_clean_timesteps(
                t_pos,
                layout.image_seq_len,
                condition_image=True,
                condition_text=False,
            )
            logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
            last_logits = logits
            last_progress = sampler_progress
            last_t_pos = t_pos

            if step in trace_requests:
                for requested_progress in trace_requests[step]:
                    for mode in IMAGE_DEPENDENCE_MODES:
                        if mode == "true_image":
                            mode_logits = logits
                        else:
                            condition_image_tokens = _image_condition_tokens(
                                image_tokens,
                                mode=mode,
                                codebook_size=layout.codebook_size,
                            )
                            control_z_t = z_t.clone()
                            control_z_t[:, layout.image_slice] = build_flm_clean_state(
                                condition_image_tokens,
                                layout.vocab_size,
                            )
                            mode_logits = mask_logits(model(control_z_t, t_pos, modality_ids), layout=layout)
                        update_point(
                            trace_kind="sampler_step",
                            requested_progress=requested_progress,
                            sampler_progress=sampler_progress,
                            mode=mode,
                            text_logits=mode_logits[:, layout.text_slice],
                            text_tokens=text_tokens,
                            labels=labels,
                        )

            if config.sampling.integrator == "scheduled_euler":
                next_progress = torch.full((batch_size,), (step + 1) / steps, device=device)
                next_t_pos = apply_schedule(
                    progress=next_progress,
                    modality_ids=modality_ids,
                    schedule_tables=schedule_tables,
                    image_time_power=config.sampling.image_time_power,
                    text_time_power=effective_text_time_power,
                )
                next_t_pos = condition_clean_timesteps(
                    next_t_pos,
                    layout.image_seq_len,
                    condition_image=True,
                    condition_text=False,
                )
                dt_pos = next_t_pos - t_pos
            else:
                dt_pos = torch.full_like(t_pos, 1.0 / steps)

            probs = torch.softmax(logits / max(config.sampling.temperature, 1e-4), dim=-1)
            v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
            z_t = z_t + dt_pos.unsqueeze(-1) * v_t
            z_t[:, layout.image_slice] = true_image_state

        if config.sampling.final_decode == "final_model_call":
            final_progress = float(config.sampling.final_model_progress)
            progress = torch.full((batch_size,), final_progress, device=device)
            final_t_pos = apply_schedule(
                progress=progress,
                modality_ids=modality_ids,
                schedule_tables=schedule_tables,
                image_time_power=config.sampling.image_time_power,
                text_time_power=effective_text_time_power,
            )
            final_t_pos = condition_clean_timesteps(
                final_t_pos,
                layout.image_seq_len,
                condition_image=True,
                condition_text=False,
            )
            true_final_logits = mask_logits(model(z_t, final_t_pos, modality_ids), layout=layout)
            for mode in IMAGE_DEPENDENCE_MODES:
                if mode == "true_image":
                    mode_logits = true_final_logits
                else:
                    condition_image_tokens = _image_condition_tokens(
                        image_tokens,
                        mode=mode,
                        codebook_size=layout.codebook_size,
                    )
                    control_z_t = z_t.clone()
                    control_z_t[:, layout.image_slice] = build_flm_clean_state(
                        condition_image_tokens,
                        layout.vocab_size,
                    )
                    mode_logits = mask_logits(model(control_z_t, final_t_pos, modality_ids), layout=layout)
                update_point(
                    trace_kind="final_model_call",
                    requested_progress=final_progress,
                    sampler_progress=final_progress,
                    mode=mode,
                    text_logits=mode_logits[:, layout.text_slice],
                    text_tokens=text_tokens,
                    labels=labels,
                )
        elif last_logits is not None and last_t_pos is not None:
            for mode in IMAGE_DEPENDENCE_MODES:
                update_point(
                    trace_kind="last_endpoint",
                    requested_progress=last_progress,
                    sampler_progress=last_progress,
                    mode=mode,
                    text_logits=last_logits[:, layout.text_slice],
                    text_tokens=text_tokens,
                    labels=labels,
                )

    results = []
    for (trace_kind, requested_progress, sampler_progress, mode), accumulator in sorted(accumulators.items()):
        record = {
            "trace_kind": trace_kind,
            "requested_progress": requested_progress,
            "sampler_progress": sampler_progress,
            "mode": mode,
        }
        record.update(_finalize_text_metric_accumulator(accumulator))
        results.append(record)
    margins = _image_dependence_margins(
        [
            {
                "progress": result["requested_progress"],
                "mode": result["mode"],
                "image_to_text_exact_match": result["image_to_text_exact_match"],
                "image_to_text_token_accuracy": result["image_to_text_token_accuracy"],
                "image_to_text_label_accuracy_constrained": result["image_to_text_label_accuracy_constrained"],
            }
            for result in results
            if result["trace_kind"] == "sampler_step"
        ]
    )
    summary = {
        "progress_values": [float(value) for value in progress_values],
        "trace_step_indices": {
            str(index): values for index, values in sorted(trace_requests.items())
        },
        "modes": list(IMAGE_DEPENDENCE_MODES),
        "results": results,
        "sampler_step_margins": margins,
        "label_strings": list(text_metadata.label_strings),
    }
    path = run_context.log_path("i2t_sampler_trajectory.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary
