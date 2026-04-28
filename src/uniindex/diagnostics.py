from __future__ import annotations

import json
from collections import Counter

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import (
    _apply_i2t_text_time_schedule,
    _decode_image_tokens,
    _load_stage2,
    _project_text_state,
    _projection_step_indices,
    constrained_text_label_values,
    safe_text_tokens,
    text_token_invalid_mask,
)
from .layout import mask_logits, maybe_mask_logits, unified_targets
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise, sample_masked_noise
from .text import decode_text_tokens, metadata_from_state, sequence_candidate_scores, shifted_label_text_tokens, text_scoring_mask
from .tokenizer import build_tokenizer
from .train import _task_time_schedule


DEFAULT_I2T_PROGRESS = (0.25, 0.5, 0.75, 0.9, 0.95)
DEFAULT_I2T_TRAJECTORY_PROGRESS = (0.5, 0.75, 0.9, 0.95)
DEFAULT_I2T_UNDERSTANDING_PROGRESS = (0.5, 0.75, 0.9, 0.95)
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
        "invalid_text_count": 0,
        "invalid_text_total": 0,
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
    shifted_text = text_logits.argmax(dim=-1)
    invalid_text = text_token_invalid_mask(shifted_text, layout)
    sampled_text = safe_text_tokens(shifted_text, layout, text_metadata)
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
    accumulator["invalid_text_count"] += invalid_text.sum().item()
    accumulator["invalid_text_total"] += invalid_text.numel()
    accumulator["total"] += labels.shape[0]


def _finalize_text_metric_accumulator(accumulator: dict) -> dict:
    total = max(accumulator["total"], 1)
    token_total = max(accumulator["token_total"], 1)
    return {
        "image_to_text_exact_match": accumulator["exact"] / total,
        "image_to_text_token_accuracy": accumulator["token_correct"] / token_total,
        "image_to_text_label_accuracy_constrained": accumulator["constrained_correct"] / total,
        "invalid_text_token_rate": accumulator["invalid_text_count"] / max(accumulator["invalid_text_total"], 1),
    }


def _confusion_matrix(true_values: list[int], pred_values: list[int], label_values: list[int]) -> list[list[int]]:
    label_to_index = {int(value): index for index, value in enumerate(label_values)}
    matrix = [[0 for _ in label_values] for _ in label_values]
    for true_value, pred_value in zip(true_values, pred_values):
        true_index = label_to_index.get(int(true_value))
        pred_index = label_to_index.get(int(pred_value))
        if true_index is not None and pred_index is not None:
            matrix[true_index][pred_index] += 1
    return matrix


def _label_score_summary(
    text_logits: torch.Tensor,
    text_metadata,
    *,
    codebook_size: int,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_tokens = shifted_label_text_tokens(text_metadata, token_offset=codebook_size).to(text_logits.device)
    scores = sequence_candidate_scores(text_logits, candidate_tokens)
    top_scores, top_indices = scores.topk(k=min(top_k, scores.shape[1]), dim=1)
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=text_logits.device)
    top_labels = label_values.index_select(0, top_indices.reshape(-1)).reshape_as(top_indices)
    predicted_labels = label_values.index_select(0, scores.argmax(dim=1))
    return scores, predicted_labels, top_labels


def _top_label_records(
    scores: torch.Tensor,
    top_labels: torch.Tensor,
    text_metadata,
) -> list[list[dict]]:
    label_to_string = {
        int(value): string for value, string in zip(text_metadata.label_values, text_metadata.label_strings)
    }
    top_scores = scores.gather(
        dim=1,
        index=torch.stack(
            [
                torch.tensor(
                    [text_metadata.label_values.index(int(label)) for label in row.tolist()],
                    dtype=torch.long,
                    device=scores.device,
                )
                for row in top_labels
            ],
            dim=0,
        ),
    )
    probabilities = torch.softmax(scores, dim=1)
    top_probabilities = probabilities.gather(
        dim=1,
        index=torch.stack(
            [
                torch.tensor(
                    [text_metadata.label_values.index(int(label)) for label in row.tolist()],
                    dtype=torch.long,
                    device=scores.device,
                )
                for row in top_labels
            ],
            dim=0,
        ),
    )
    records = []
    for label_row, score_row, probability_row in zip(top_labels.cpu(), top_scores.cpu(), top_probabilities.cpu()):
        records.append(
            [
                {
                    "label": int(label),
                    "text": label_to_string.get(int(label), str(int(label))),
                    "score": float(score),
                    "probability": float(probability),
                }
                for label, score, probability in zip(label_row.tolist(), score_row.tolist(), probability_row.tolist())
            ]
        )
    return records


def _true_label_margin(scores: torch.Tensor, labels: torch.Tensor, label_values: tuple[int, ...]) -> torch.Tensor:
    label_to_index = {int(value): index for index, value in enumerate(label_values)}
    true_indices = torch.tensor(
        [label_to_index[int(label)] for label in labels.detach().cpu().tolist()],
        dtype=torch.long,
        device=scores.device,
    )
    true_scores = scores.gather(dim=1, index=true_indices[:, None]).squeeze(1)
    wrong_scores = scores.clone()
    wrong_scores.scatter_(dim=1, index=true_indices[:, None], value=-torch.inf)
    return true_scores - wrong_scores.max(dim=1).values


def _tensor_to_pil(image: torch.Tensor, size: int = 128) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    pil = Image.fromarray(array)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BICUBIC)
    return pil


def _draw_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    *,
    cell_size: int = 36,
    margin: int = 72,
) -> Image.Image:
    size = margin + cell_size * len(labels)
    canvas = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    max_value = max([value for row in matrix for value in row] or [1])
    max_value = max(max_value, 1)
    draw.text((margin, 8), "pred", fill=(0, 0, 0), font=font)
    draw.text((8, margin - 24), "true", fill=(0, 0, 0), font=font)
    for index, label in enumerate(labels):
        x = margin + index * cell_size + 8
        y = margin - 18
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        draw.text((margin - 28, margin + index * cell_size + 12), label, fill=(0, 0, 0), font=font)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            intensity = int(255 - 210 * (value / max_value))
            x0 = margin + col_index * cell_size
            y0 = margin + row_index * cell_size
            fill = (255, intensity, intensity) if row_index != col_index else (intensity, 255, intensity)
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), fill=fill, outline=(220, 220, 220))
            draw.text((x0 + 10, y0 + 12), str(value), fill=(0, 0, 0), font=font)
    return canvas


def _draw_margin_chart(points: list[dict], *, width: int = 640, height: int = 360) -> Image.Image:
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    left, right, top, bottom = 58, width - 24, 28, height - 44
    draw.rectangle((left, top, right, bottom), outline=(210, 210, 210))
    draw.text((left, 8), "true - shuffled margin", fill=(0, 0, 0), font=font)
    if not points:
        return canvas

    progress_values = [float(point["progress"]) for point in points]
    min_progress = min(progress_values)
    max_progress = max(progress_values)
    if min_progress == max_progress:
        min_progress -= 0.01
        max_progress += 0.01
    series = [
        ("direct_label_margin", (40, 110, 220)),
        ("sampler_label_margin", (220, 90, 40)),
        ("direct_exact_margin", (70, 170, 90)),
        ("sampler_exact_margin", (160, 90, 190)),
    ]
    all_values = [float(point[key]) for point in points for key, _ in series if key in point]
    y_min = min(0.0, min(all_values))
    y_max = max(0.05, max(all_values))
    if y_min == y_max:
        y_max += 0.01

    def xy(progress: float, value: float) -> tuple[int, int]:
        x = left + int((progress - min_progress) / (max_progress - min_progress) * (right - left))
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        return x, y

    zero_y = xy(min_progress, 0.0)[1]
    draw.line((left, zero_y, right, zero_y), fill=(230, 230, 230))
    for offset, (key, color) in enumerate(series):
        series_points = [xy(float(point["progress"]), float(point[key])) for point in points if key in point]
        if len(series_points) >= 2:
            draw.line(series_points, fill=color, width=2)
        for point in series_points:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.rectangle((left + offset * 142, bottom + 18, left + offset * 142 + 10, bottom + 28), fill=color)
        draw.text((left + offset * 142 + 14, bottom + 16), key.replace("_", " "), fill=(0, 0, 0), font=font)
    for progress in progress_values:
        x, _ = xy(progress, y_min)
        draw.text((x - 10, bottom + 4), f"{progress:g}", fill=(0, 0, 0), font=font)
    return canvas


def _draw_sample_grid(samples: list[dict], images: list[Image.Image], *, cols: int = 4) -> Image.Image:
    if not samples:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    cell_size = 128
    caption_height = 86
    rows = (len(samples) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + caption_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, sample in enumerate(samples):
        row = index // cols
        col = index % cols
        x = col * cell_size
        y = row * (cell_size + caption_height)
        canvas.paste(images[index], (x, y))
        correct = bool(sample["free_label_correct"])
        outline = (44, 160, 72) if correct else (210, 70, 60)
        draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1), outline=outline, width=3)
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + caption_height), fill=(248, 248, 248))
        top = sample["free_top3"][0] if sample["free_top3"] else {"text": "?", "probability": 0.0}
        caption = (
            f"gt={sample['target_text']} free={sample['free_text']}\n"
            f"cls={sample['free_label_text']} top={top['text']} {top['probability']:.2f}\n"
            f"direct@.5={sample['direct_true_progress_0.5_label_text']}\n"
            f"sampler@.5={sample['sampler_true_progress_0.5_label_text']}"
        )
        draw.multiline_text((x + 4, y + cell_size + 4), caption, fill=(0, 0, 0), font=font, spacing=2)
    return canvas


def _prediction_records(
    *,
    text_logits: torch.Tensor,
    text_tokens: torch.Tensor,
    labels: torch.Tensor,
    text_metadata,
    layout,
    top_k: int = 3,
) -> tuple[list[dict], dict]:
    sampled_text = text_logits.argmax(dim=-1) - layout.text_offset
    sampled_strings = decode_text_tokens(sampled_text, text_metadata)
    target_strings = decode_text_tokens(text_tokens, text_metadata)
    scores, predicted_labels, top_labels = _label_score_summary(
        text_logits,
        text_metadata,
        codebook_size=layout.codebook_size,
        top_k=top_k,
    )
    margins = _true_label_margin(scores, labels, text_metadata.label_values)
    top_records = _top_label_records(scores, top_labels, text_metadata)
    label_to_string = {
        int(value): string for value, string in zip(text_metadata.label_values, text_metadata.label_strings)
    }
    valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
    exact = [pred == target for pred, target in zip(sampled_strings, target_strings)]
    token_correct = sampled_text.eq(text_tokens).logical_and(valid_text).sum(dim=1)
    token_total = valid_text.sum(dim=1).clamp_min(1)
    records = []
    for index in range(labels.shape[0]):
        predicted_label = int(predicted_labels[index].item())
        true_label = int(labels[index].item())
        records.append(
            {
                "target_text": target_strings[index],
                "free_text": sampled_strings[index],
                "free_exact": bool(exact[index]),
                "true_label": true_label,
                "true_label_text": label_to_string.get(true_label, str(true_label)),
                "predicted_label": predicted_label,
                "predicted_label_text": label_to_string.get(predicted_label, str(predicted_label)),
                "label_correct": predicted_label == true_label,
                "token_accuracy": float(token_correct[index].item() / token_total[index].item()),
                "true_label_margin": float(margins[index].item()),
                "top3": top_records[index],
            }
        )
    total = max(labels.shape[0], 1)
    summary = {
        "image_to_text_exact_match": sum(exact) / total,
        "image_to_text_token_accuracy": float(token_correct.sum().item() / token_total.sum().clamp_min(1).item()),
        "image_to_text_label_accuracy_constrained": float((predicted_labels == labels).float().mean().item()),
        "mean_true_label_margin": float(margins.mean().item()),
    }
    return records, summary


def _summarize_records_by_progress(
    records: list[dict],
    *,
    style: str,
    mode: str,
) -> list[dict]:
    return [
        {
            "progress": record["progress"],
            "style": style,
            "mode": mode,
            "image_to_text_exact_match": record["summary"]["image_to_text_exact_match"],
            "image_to_text_token_accuracy": record["summary"]["image_to_text_token_accuracy"],
            "image_to_text_label_accuracy_constrained": record["summary"][
                "image_to_text_label_accuracy_constrained"
            ],
            "mean_true_label_margin": record["summary"]["mean_true_label_margin"],
        }
        for record in records
        if record["progress"] != "final"
        if record["mode"] == mode
    ]


def _sample_i2t_trace_logits(
    *,
    model,
    layout,
    schedule_tables: dict,
    image_tokens: torch.Tensor,
    text_metadata,
    config: ProjectConfig,
    progress_values: tuple[float, ...],
) -> dict[float | str, torch.Tensor]:
    device = image_tokens.device
    batch_size = image_tokens.shape[0]
    steps = int(config.sampling.steps)
    modality_ids = layout.position_modalities().to(device)
    modality_valid_token_mask = layout.position_valid_token_mask().to(device)
    noise_valid_token_mask = None if config.state.noise_support == "full_vocab" else modality_valid_token_mask
    z_t = sample_masked_noise(
        torch.zeros(batch_size, layout.seq_len, layout.vocab_size, device=device),
        noise_valid_token_mask,
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
    traced: dict[float | str, torch.Tensor] = {}
    last_logits = None

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
        if config.sampling.integrator == "scheduled_euler":
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
            dt_pos = next_t_pos - t_pos
        else:
            dt_pos = torch.full_like(t_pos, 1.0 / steps)
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

        logits = maybe_mask_logits(model(z_t, t_pos, modality_ids), layout=layout, mode=config.sampling.logit_mask)
        last_logits = logits
        if step in trace_requests:
            for requested_progress in trace_requests[step]:
                traced[float(requested_progress)] = logits[:, layout.text_slice].detach().clone()
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
                model=model,
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
                None if noise_valid_token_mask is None else noise_valid_token_mask[layout.text_slice],
            )

    if config.sampling.final_decode == "final_model_call":
        final_progress = torch.full((batch_size,), float(config.sampling.final_model_progress), device=device)
        final_t_pos = apply_schedule(
            progress=final_progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=config.sampling.image_time_power,
            text_time_power=effective_text_time_power,
        )
        final_t_pos = _apply_i2t_text_time_schedule(
            final_t_pos,
            final_progress,
            layout=layout,
            schedule=config.sampling.image_to_text_text_time_schedule,
            logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
            logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
        )
        final_t_pos = condition_clean_timesteps(
            final_t_pos,
            layout.image_seq_len,
            condition_image=True,
            condition_text=False,
        )
        final_logits = maybe_mask_logits(
            model(z_t, final_t_pos, modality_ids),
            layout=layout,
            mode=config.sampling.logit_mask,
        )
    elif last_logits is not None:
        final_logits = last_logits
    else:
        raise RuntimeError("i2t sampler trace requires at least one sampling step")
    traced["final"] = final_logits[:, layout.text_slice].detach().clone()
    return traced


@torch.inference_mode()
def diagnose_i2t_understanding(
    config: ProjectConfig,
    sample_count: int = 32,
    progress_values: tuple[float, ...] = DEFAULT_I2T_UNDERSTANDING_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}")
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t-understanding")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    loader = build_loader(
        split_path(config, "test"),
        batch_size=int(sample_count),
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    batch = next(iter(loader))
    image_tokens = batch["image_tokens"].to(device)
    text_tokens = batch["text_tokens"].to(device)
    labels = batch["label"].to(device)
    sample_count = int(labels.shape[0])
    modality_ids = layout.position_modalities().to(device)
    modality_valid_token_mask = layout.position_valid_token_mask().to(device)
    noise_valid_token_mask = None if config.state.noise_support == "full_vocab" else modality_valid_token_mask

    direct_records = []
    direct_by_key: dict[tuple[float, str], list[dict]] = {}
    for progress_value in progress_values:
        for mode in IMAGE_DEPENDENCE_MODES:
            condition_image_tokens = _image_condition_tokens(
                image_tokens,
                mode=mode,
                codebook_size=layout.codebook_size,
            )
            targets = unified_targets(condition_image_tokens, text_tokens, layout.codebook_size)
            x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
            progress = torch.full((sample_count,), float(progress_value), device=device)
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
                None if noise_valid_token_mask is None else noise_valid_token_mask[layout.text_slice],
            )
            logits = maybe_mask_logits(model(z_t, t_pos, modality_ids), layout=layout, mode=config.train.logit_mask)
            samples, summary = _prediction_records(
                text_logits=logits[:, layout.text_slice],
                text_tokens=text_tokens,
                labels=labels,
                text_metadata=text_metadata,
                layout=layout,
            )
            direct_by_key[(float(progress_value), mode)] = samples
            direct_records.append(
                {
                    "progress": float(progress_value),
                    "mode": mode,
                    "summary": summary,
                }
            )

    sampler_records = []
    sampler_by_key: dict[tuple[float | str, str], list[dict]] = {}
    for mode in IMAGE_DEPENDENCE_MODES:
        condition_image_tokens = _image_condition_tokens(
            image_tokens,
            mode=mode,
            codebook_size=layout.codebook_size,
        )
        trace_logits = _sample_i2t_trace_logits(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            image_tokens=condition_image_tokens,
            text_metadata=text_metadata,
            config=config,
            progress_values=progress_values,
        )
        for progress_value in [*progress_values, "final"]:
            samples, summary = _prediction_records(
                text_logits=trace_logits[progress_value],
                text_tokens=text_tokens,
                labels=labels,
                text_metadata=text_metadata,
                layout=layout,
            )
            sampler_by_key[(progress_value, mode)] = samples
            sampler_records.append(
                {
                    "progress": progress_value if progress_value == "final" else float(progress_value),
                    "mode": mode,
                    "summary": summary,
                }
            )

    final_true_samples = sampler_by_key[("final", "true_image")]
    final_shuffled_samples = sampler_by_key[("final", "shuffled_image")]
    final_random_samples = sampler_by_key[("final", "random_image_tokens")]
    progress_anchor = float(progress_values[0])
    sample_records = []
    for index, final_record in enumerate(final_true_samples):
        direct_anchor = direct_by_key[(progress_anchor, "true_image")][index]
        sampler_anchor = sampler_by_key[(progress_anchor, "true_image")][index]
        shuffled_final = final_shuffled_samples[index]
        random_final = final_random_samples[index]
        sample_records.append(
            {
                "sample_index": index,
                "true_label": final_record["true_label"],
                "target_text": final_record["target_text"],
                "free_text": final_record["free_text"],
                "free_exact": final_record["free_exact"],
                "free_label": final_record["predicted_label"],
                "free_label_text": final_record["predicted_label_text"],
                "free_label_correct": final_record["label_correct"],
                "free_top3": final_record["top3"],
                "free_true_label_margin": final_record["true_label_margin"],
                "shuffled_final_label": shuffled_final["predicted_label"],
                "shuffled_final_label_text": shuffled_final["predicted_label_text"],
                "shuffled_final_label_correct": shuffled_final["label_correct"],
                "random_final_label": random_final["predicted_label"],
                "random_final_label_text": random_final["predicted_label_text"],
                "random_final_label_correct": random_final["label_correct"],
                "direct_true_progress_0.5_label": direct_anchor["predicted_label"],
                "direct_true_progress_0.5_label_text": direct_anchor["predicted_label_text"],
                "direct_true_progress_0.5_margin": direct_anchor["true_label_margin"],
                "sampler_true_progress_0.5_label": sampler_anchor["predicted_label"],
                "sampler_true_progress_0.5_label_text": sampler_anchor["predicted_label_text"],
                "sampler_true_progress_0.5_margin": sampler_anchor["true_label_margin"],
                "failure_type": (
                    "exact_correct"
                    if final_record["free_exact"]
                    else "label_correct_text_wrong"
                    if final_record["label_correct"]
                    else "image_insensitive"
                    if shuffled_final["label_correct"] or random_final["label_correct"]
                    else "label_wrong"
                ),
            }
        )

    label_values = [int(value) for value in text_metadata.label_values]
    label_strings = list(text_metadata.label_strings)
    true_labels = [sample["true_label"] for sample in sample_records]
    free_labels = [sample["free_label"] for sample in sample_records]
    confusion = _confusion_matrix(true_labels, free_labels, label_values)
    failure_counts = Counter(sample["failure_type"] for sample in sample_records)

    def record_map(records: list[dict], mode: str) -> dict[float, dict]:
        return {
            float(record["progress"]): record["summary"]
            for record in records
            if record["mode"] == mode and record["progress"] != "final"
        }

    direct_true = record_map(direct_records, "true_image")
    direct_shuffled = record_map(direct_records, "shuffled_image")
    sampler_true = record_map(sampler_records, "true_image")
    sampler_shuffled = record_map(sampler_records, "shuffled_image")
    margin_points = []
    for progress_value in progress_values:
        progress = float(progress_value)
        margin_points.append(
            {
                "progress": progress,
                "direct_label_margin": direct_true[progress]["image_to_text_label_accuracy_constrained"]
                - direct_shuffled[progress]["image_to_text_label_accuracy_constrained"],
                "direct_exact_margin": direct_true[progress]["image_to_text_exact_match"]
                - direct_shuffled[progress]["image_to_text_exact_match"],
                "sampler_label_margin": sampler_true[progress]["image_to_text_label_accuracy_constrained"]
                - sampler_shuffled[progress]["image_to_text_label_accuracy_constrained"],
                "sampler_exact_margin": sampler_true[progress]["image_to_text_exact_match"]
                - sampler_shuffled[progress]["image_to_text_exact_match"],
            }
        )

    tokenizer = build_tokenizer(config, device=device)
    decoded_images = _decode_image_tokens(
        tokenizer,
        image_tokens,
        tokenizer_state,
        tuple(tokenizer_state["grid_shape"]),
        device,
    )
    pil_images = [_tensor_to_pil(image) for image in decoded_images]
    grid_path = run_context.log_path("i2t_understanding/sample_cards.png")
    confusion_path = run_context.log_path("i2t_understanding/confusion_matrix.png")
    margin_chart_path = run_context.log_path("i2t_understanding/train_vs_sampler_margins.png")
    samples_path = run_context.log_path("i2t_understanding/samples.json")
    summary_path = run_context.log_path("i2t_understanding/summary.json")
    _draw_sample_grid(sample_records, pil_images).save(grid_path)
    _draw_confusion_matrix(confusion, label_strings).save(confusion_path)
    _draw_margin_chart(margin_points).save(margin_chart_path)

    summary = {
        "sample_count": sample_count,
        "progress_values": [float(value) for value in progress_values],
        "paths": {
            "sample_cards": str(grid_path),
            "confusion_matrix": str(confusion_path),
            "train_vs_sampler_margins": str(margin_chart_path),
            "samples": str(samples_path),
            "summary": str(summary_path),
        },
        "label_strings": label_strings,
        "final_true_summary": next(
            record["summary"]
            for record in sampler_records
            if record["mode"] == "true_image" and record["progress"] == "final"
        ),
        "final_shuffled_summary": next(
            record["summary"]
            for record in sampler_records
            if record["mode"] == "shuffled_image" and record["progress"] == "final"
        ),
        "final_random_summary": next(
            record["summary"]
            for record in sampler_records
            if record["mode"] == "random_image_tokens" and record["progress"] == "final"
        ),
        "failure_counts": dict(failure_counts),
        "confusion_matrix": {
            "labels": label_values,
            "label_strings": label_strings,
            "matrix": confusion,
        },
        "direct_records": _summarize_records_by_progress(direct_records, style="train_style", mode="true_image")
        + _summarize_records_by_progress(direct_records, style="train_style", mode="shuffled_image")
        + _summarize_records_by_progress(direct_records, style="train_style", mode="random_image_tokens"),
        "sampler_records": _summarize_records_by_progress(sampler_records, style="sampler_style", mode="true_image")
        + _summarize_records_by_progress(sampler_records, style="sampler_style", mode="shuffled_image")
        + _summarize_records_by_progress(sampler_records, style="sampler_style", mode="random_image_tokens"),
        "train_vs_sampler_margins": margin_points,
    }
    with samples_path.open("w", encoding="utf-8") as handle:
        json.dump(sample_records, handle, indent=2)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary


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
    modality_valid_token_mask = layout.position_valid_token_mask().to(device)
    noise_valid_token_mask = None if config.state.noise_support == "full_vocab" else modality_valid_token_mask

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
        invalid_text_count = 0
        invalid_text_total = 0
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
                None if noise_valid_token_mask is None else noise_valid_token_mask[layout.text_slice],
            )
            logits = model(z_t, t_pos, modality_ids)
            logits = maybe_mask_logits(logits, layout=layout, mode=config.train.logit_mask)
            text_logits = logits[:, layout.text_slice]
            shifted_text = text_logits.argmax(dim=-1)
            invalid_text = text_token_invalid_mask(shifted_text, layout)
            sampled_text = safe_text_tokens(shifted_text, layout, text_metadata)
            invalid_text_count += int(invalid_text.sum().item())
            invalid_text_total += int(invalid_text.numel())

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
                "invalid_text_token_rate": invalid_text_count / max(invalid_text_total, 1),
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
