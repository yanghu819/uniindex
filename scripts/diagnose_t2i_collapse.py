from __future__ import annotations

import argparse
import html
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from uniindex.config import load_config
from uniindex.data import TokenizedImageDataset, split_path
from uniindex.datasets import build_image_dataset
from uniindex.eval import _decode_image_tokens, _load_stage2
from uniindex.layout import TaskLayout, mask_logits
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import apply_schedule, build_schedule_tables
from uniindex.state import build_flm_clean_state, condition_clean_timesteps
from uniindex.text import TextMetadata, metadata_from_state
from uniindex.tokenizer import build_tokenizer


PALETTE = {
    0: (31, 119, 180),
    1: (255, 127, 14),
    2: (44, 160, 44),
    3: (214, 39, 40),
    4: (148, 103, 189),
    5: (140, 86, 75),
    6: (227, 119, 194),
    7: (127, 127, 127),
    8: (188, 189, 34),
    9: (23, 190, 207),
}


@dataclass(frozen=True)
class ModeBatch:
    mode: str
    condition_text: torch.Tensor
    target_labels: torch.Tensor
    prompt_labels: torch.Tensor
    sample_seeds: torch.Tensor
    prompt_texts: list[str]


def _round_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def _counter_dict(values: torch.Tensor) -> dict[str, int]:
    counter = Counter(int(value) for value in values.detach().cpu().tolist())
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _safe_label(label: int, label_to_string: dict[int, str]) -> str:
    if label < 0:
        return "none"
    return label_to_string.get(int(label), str(int(label)))


def _tensor_to_pil(image: torch.Tensor, size: int) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = np.repeat(array, repeats=3, axis=-1)
    pil = Image.fromarray(array, mode="RGB")
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BILINEAR)
    return pil


def _raw_image(raw_dataset, index: int, size: int) -> Image.Image:
    image, _ = raw_dataset[index]
    image = image.convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return image


def _add_caption(image: Image.Image, caption: str, height: int) -> Image.Image:
    out = Image.new("RGB", (image.width, image.height + height), color=(248, 248, 248))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.multiline_text((3, image.height + 3), caption, fill=(0, 0, 0), font=font, spacing=2)
    return out


def _make_grid(images: list[Image.Image], *, cols: int) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_width, rows * cell_height), color=(255, 255, 255))
    for index, image in enumerate(images):
        row = index // cols
        col = index % cols
        canvas.paste(image, (col * cell_width, row * cell_height))
    return canvas


def _empty_prompt(metadata: TextMetadata) -> torch.Tensor:
    row = torch.full((metadata.seq_len,), metadata.pad_id, dtype=torch.long)
    row[0] = metadata.bos_id
    if metadata.seq_len > 1:
        row[1] = metadata.eos_id
    return row


def _random_prompt(metadata: TextMetadata, seed: int) -> tuple[torch.Tensor, str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    row = torch.full((metadata.seq_len,), metadata.pad_id, dtype=torch.long)
    row[0] = metadata.bos_id
    if metadata.seq_len <= 2:
        return row, ""
    content_len = int(torch.randint(1, metadata.seq_len - 1, (1,), generator=generator).item())
    char_ids = [
        index
        for index, token in enumerate(metadata.vocab_tokens)
        if index not in {metadata.pad_id, metadata.bos_id, metadata.eos_id}
    ]
    if not char_ids:
        char_ids = list(range(metadata.vocab_size))
    sampled = torch.randint(0, len(char_ids), (content_len,), generator=generator)
    for pos, sampled_index in enumerate(sampled.tolist(), start=1):
        row[pos] = int(char_ids[sampled_index])
    eos_pos = min(content_len + 1, metadata.seq_len - 1)
    row[eos_pos] = metadata.eos_id
    text = "".join(metadata.vocab_tokens[int(token)] for token in row.tolist()[1:eos_pos])
    return row, text


def _make_mode_batch(metadata: TextMetadata, mode: str, seeds_per_label: int, base_seed: int, device: torch.device) -> ModeBatch:
    label_values = [int(value) for value in metadata.label_values]
    rows: list[torch.Tensor] = []
    target_labels: list[int] = []
    prompt_labels: list[int] = []
    sample_seeds: list[int] = []
    prompt_texts: list[str] = []
    label_text_rows = metadata.label_text_tokens
    for label_index, target_label in enumerate(label_values):
        for repeat in range(seeds_per_label):
            sample_seed = int(base_seed + label_index * 100_000 + repeat)
            if mode == "correct":
                prompt_index = label_index
                row = label_text_rows[prompt_index]
                prompt_label = label_values[prompt_index]
                prompt_text = metadata.label_strings[prompt_index]
            elif mode == "shuffled":
                prompt_index = (label_index + 1) % len(label_values)
                row = label_text_rows[prompt_index]
                prompt_label = label_values[prompt_index]
                prompt_text = metadata.label_strings[prompt_index]
            elif mode == "empty":
                row = _empty_prompt(metadata)
                prompt_label = -1
                prompt_text = ""
            elif mode == "random":
                row, prompt_text = _random_prompt(metadata, sample_seed + 13_579)
                prompt_label = -1
            else:
                raise ValueError(f"unsupported mode: {mode}")
            rows.append(row.clone())
            target_labels.append(target_label)
            prompt_labels.append(prompt_label)
            sample_seeds.append(sample_seed)
            prompt_texts.append(prompt_text)
    return ModeBatch(
        mode=mode,
        condition_text=torch.stack(rows, dim=0).to(device),
        target_labels=torch.tensor(target_labels, dtype=torch.long, device=device),
        prompt_labels=torch.tensor(prompt_labels, dtype=torch.long, device=device),
        sample_seeds=torch.tensor(sample_seeds, dtype=torch.long, device=device),
        prompt_texts=prompt_texts,
    )


def _initial_noise(
    *,
    layout: TaskLayout,
    sample_seeds: torch.Tensor,
    valid_token_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    shape = (layout.seq_len, layout.vocab_size)
    mask = valid_token_mask.to(device=device, dtype=torch.float32)
    for seed in sample_seeds.detach().cpu().tolist():
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        rows.append(torch.randn(shape, device=device, generator=generator) * mask)
    return torch.stack(rows, dim=0)


def _nearest_matches(generated: torch.Tensor, bank_tokens: torch.Tensor, *, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    best_dist = torch.full((generated.shape[0],), generated.shape[1] + 1, dtype=torch.long, device=generated.device)
    best_index = torch.full((generated.shape[0],), -1, dtype=torch.long, device=generated.device)
    for start in range(0, bank_tokens.shape[0], chunk_size):
        chunk = bank_tokens[start : start + chunk_size]
        distances = generated[:, None, :].ne(chunk[None, :, :]).sum(dim=-1)
        chunk_best_dist, chunk_best_index = distances.min(dim=1)
        update = chunk_best_dist < best_dist
        best_dist = torch.where(update, chunk_best_dist, best_dist)
        best_index = torch.where(update, chunk_best_index + start, best_index)
    return best_index, best_dist


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    clipped = probs.clamp_min(1e-12)
    return -(clipped * clipped.log()).sum(dim=-1)


def _consensus_by_position(tokens: torch.Tensor) -> tuple[list[float], list[int]]:
    tokens_cpu = tokens.detach().cpu()
    fractions: list[float] = []
    top_tokens: list[int] = []
    total = max(int(tokens_cpu.shape[0]), 1)
    for pos in range(tokens_cpu.shape[1]):
        counts = torch.bincount(tokens_cpu[:, pos].long())
        value, index = counts.max(dim=0)
        fractions.append(float(value.item()) / total)
        top_tokens.append(int(index.item()))
    return fractions, top_tokens


def _prob_position_stats(probs: torch.Tensor) -> dict[str, Any]:
    entropy = _entropy(probs)
    top_prob = probs.max(dim=-1).values
    argmax_tokens = probs.argmax(dim=-1)
    consensus, consensus_tokens = _consensus_by_position(argmax_tokens)
    return {
        "entropy_mean": _round_float(float(entropy.mean().item())),
        "entropy_by_position": [_round_float(float(value)) for value in entropy.mean(dim=0).detach().cpu().tolist()],
        "top_prob_mean": _round_float(float(top_prob.mean().item())),
        "top_prob_by_position": [_round_float(float(value)) for value in top_prob.mean(dim=0).detach().cpu().tolist()],
        "argmax_consensus_mean": _round_float(float(sum(consensus) / max(len(consensus), 1))),
        "argmax_consensus_by_position": [_round_float(value) for value in consensus],
        "argmax_consensus_token_by_position": consensus_tokens,
    }


def _state_position_stats(tokens: torch.Tensor) -> dict[str, Any]:
    consensus, consensus_tokens = _consensus_by_position(tokens)
    return {
        "state_argmax_consensus_mean": _round_float(float(sum(consensus) / max(len(consensus), 1))),
        "state_argmax_consensus_by_position": [_round_float(value) for value in consensus],
        "state_argmax_consensus_token_by_position": consensus_tokens,
    }


def _label_group_summary(
    *,
    nearest_labels: torch.Tensor,
    nearest_distances: torch.Tensor,
    condition_labels: torch.Tensor,
    group_labels: torch.Tensor,
    probs: torch.Tensor | None,
    state_tokens: torch.Tensor,
    valid_group_values: list[int],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_value in valid_group_values:
        value = int(raw_value)
        mask = group_labels.eq(value)
        count = int(mask.sum().item())
        if count == 0:
            continue
        group_nearest = nearest_labels[mask]
        nearest_counts = _counter_dict(group_nearest)
        item: dict[str, Any] = {
            "total": count,
            "nearest_label_counts": nearest_counts,
            "nearest_label_fraction": {key: _round_float(val / count) for key, val in nearest_counts.items()},
            "mean_nearest_distance": _round_float(float(nearest_distances[mask].float().mean().item())),
            "target_match_fraction": _round_float(float(group_nearest.eq(condition_labels[mask]).float().mean().item())),
        }
        if probs is not None:
            group_probs = probs[mask]
            group_entropy = _entropy(group_probs)
            item["entropy_mean"] = _round_float(float(group_entropy.mean().item()))
            item["top_prob_mean"] = _round_float(float(group_probs.max(dim=-1).values.mean().item()))
            prob_consensus, _ = _consensus_by_position(group_probs.argmax(dim=-1))
            item["argmax_consensus_mean"] = _round_float(float(sum(prob_consensus) / max(len(prob_consensus), 1)))
        state_consensus, _ = _consensus_by_position(state_tokens[mask])
        item["state_argmax_consensus_mean"] = _round_float(float(sum(state_consensus) / max(len(state_consensus), 1)))
        output[str(value)] = item
    return output


def _step_summary(
    *,
    mode_batch: ModeBatch,
    step: int,
    phase: str,
    progress: float,
    state_tokens: torch.Tensor,
    probs: torch.Tensor | None,
    bank_tokens: torch.Tensor,
    bank_labels: torch.Tensor,
    bank_chunk_size: int,
    label_values: list[int],
    label_to_string: dict[int, str],
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    nearest_indices, nearest_distances = _nearest_matches(state_tokens, bank_tokens, chunk_size=bank_chunk_size)
    nearest_labels = bank_labels.index_select(0, nearest_indices)
    total = int(nearest_labels.shape[0])
    prompt_mask = mode_batch.prompt_labels.ge(0)
    prompt_total = int(prompt_mask.sum().item())
    nearest_counts = _counter_dict(nearest_labels)
    item: dict[str, Any] = {
        "mode": mode_batch.mode,
        "step": int(step),
        "phase": phase,
        "progress": _round_float(progress),
        "total": total,
        "nearest_label_counts": nearest_counts,
        "nearest_label_fraction": {key: _round_float(value / total) for key, value in nearest_counts.items()},
        "target_match_fraction": _round_float(float(nearest_labels.eq(mode_batch.target_labels).float().mean().item())),
        "prompt_match_fraction": None
        if prompt_total == 0
        else _round_float(float(nearest_labels[prompt_mask].eq(mode_batch.prompt_labels[prompt_mask]).float().mean().item())),
        "mean_nearest_distance": _round_float(float(nearest_distances.float().mean().item())),
        "per_target_label": _label_group_summary(
            nearest_labels=nearest_labels,
            nearest_distances=nearest_distances,
            condition_labels=mode_batch.target_labels,
            group_labels=mode_batch.target_labels,
            probs=probs,
            state_tokens=state_tokens,
            valid_group_values=label_values,
        ),
        "per_prompt_label": {}
        if prompt_total == 0
        else _label_group_summary(
            nearest_labels=nearest_labels[prompt_mask],
            nearest_distances=nearest_distances[prompt_mask],
            condition_labels=mode_batch.prompt_labels[prompt_mask],
            group_labels=mode_batch.prompt_labels[prompt_mask],
            probs=None if probs is None else probs[prompt_mask],
            state_tokens=state_tokens[prompt_mask],
            valid_group_values=label_values,
        ),
        "label_names": {str(label): label_to_string[label] for label in label_values},
    }
    item.update(_state_position_stats(state_tokens))
    if probs is not None:
        item["model_prob_stats"] = _prob_position_stats(probs)
    else:
        item["model_prob_stats"] = None
    return item, nearest_indices, nearest_distances, nearest_labels


def _update_position_collapse(
    collapse: dict[str, Any],
    *,
    mode: str,
    group_kind: str,
    group_value: str,
    step: int,
    phase: str,
    probs: torch.Tensor | None,
    state_tokens: torch.Tensor,
    top_prob_threshold: float,
    consensus_threshold: float,
) -> None:
    key = f"{mode}/{group_kind}/{group_value}"
    entry = collapse.setdefault(
        key,
        {
            "mode": mode,
            "group_kind": group_kind,
            "group_value": group_value,
            "top_prob_threshold": top_prob_threshold,
            "consensus_threshold": consensus_threshold,
            "positions": {},
        },
    )
    positions = entry["positions"]
    if probs is not None:
        prob_stats = _prob_position_stats(probs)
        top_probs = prob_stats["top_prob_by_position"]
        argmax_consensus = prob_stats["argmax_consensus_by_position"]
    else:
        top_probs = [None] * state_tokens.shape[1]
        argmax_consensus = [None] * state_tokens.shape[1]
    state_consensus, state_tokens_top = _consensus_by_position(state_tokens)
    for pos, state_value in enumerate(state_consensus):
        top_prob = top_probs[pos]
        argmax_value = argmax_consensus[pos]
        collapsed_by_top_prob = top_prob is not None and top_prob >= top_prob_threshold
        collapsed_by_argmax = argmax_value is not None and argmax_value >= consensus_threshold
        collapsed_by_state = state_value >= consensus_threshold
        if not (collapsed_by_top_prob or collapsed_by_argmax or collapsed_by_state):
            continue
        pos_key = str(pos)
        if pos_key in positions:
            continue
        positions[pos_key] = {
            "first_step": int(step),
            "phase": phase,
            "top_prob": top_prob,
            "argmax_consensus": argmax_value,
            "state_argmax_consensus": _round_float(float(state_value)),
            "state_top_token": int(state_tokens_top[pos]),
            "reason": {
                "top_prob": bool(collapsed_by_top_prob),
                "model_argmax_consensus": bool(collapsed_by_argmax),
                "state_argmax_consensus": bool(collapsed_by_state),
            },
        }


def _draw_line_chart(
    *,
    title: str,
    x_values: list[int],
    series: list[tuple[str, list[float | None], tuple[int, int, int]]],
    out_path: Path,
    y_min: float = 0.0,
    y_max: float | None = None,
    width: int = 960,
    height: int = 520,
) -> None:
    margin_left, margin_right, margin_top, margin_bottom = 70, 24, 44, 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if y_max is None:
        observed = [value for _, values, _ in series for value in values if value is not None]
        y_max = max(observed) if observed else 1.0
    y_max = max(y_max, y_min + 1e-6)
    min_x, max_x = min(x_values), max(x_values)
    if min_x == max_x:
        max_x = min_x + 1

    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((margin_left, 14), title, fill=(0, 0, 0), font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + plot_height), fill=(80, 80, 80), width=1)
    draw.line(
        (margin_left, margin_top + plot_height, margin_left + plot_width, margin_top + plot_height),
        fill=(80, 80, 80),
        width=1,
    )
    for tick in range(6):
        frac = tick / 5
        y = margin_top + plot_height - frac * plot_height
        value = y_min + frac * (y_max - y_min)
        draw.line((margin_left - 4, y, margin_left + plot_width, y), fill=(230, 230, 230), width=1)
        draw.text((8, y - 6), f"{value:.2f}", fill=(80, 80, 80), font=font)
    for tick in range(6):
        frac = tick / 5
        x = margin_left + frac * plot_width
        value = int(round(min_x + frac * (max_x - min_x)))
        draw.line((x, margin_top + plot_height, x, margin_top + plot_height + 4), fill=(80, 80, 80), width=1)
        draw.text((x - 12, margin_top + plot_height + 10), str(value), fill=(80, 80, 80), font=font)

    def xy(index: int, value: float) -> tuple[int, int]:
        x = margin_left + (x_values[index] - min_x) / (max_x - min_x) * plot_width
        y = margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height
        return int(round(x)), int(round(y))

    legend_x = margin_left + 8
    legend_y = margin_top + 8
    for name, values, color in series:
        points = [(idx, value) for idx, value in enumerate(values) if value is not None]
        if len(points) >= 2:
            draw.line([xy(idx, float(value)) for idx, value in points], fill=color, width=2)
        elif len(points) == 1:
            x, y = xy(points[0][0], float(points[0][1]))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.rectangle((legend_x, legend_y, legend_x + 10, legend_y + 10), fill=color)
        draw.text((legend_x + 14, legend_y - 1), name, fill=(0, 0, 0), font=font)
        legend_y += 14
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def _write_curves(out_dir: Path, per_step: list[dict[str, Any]], label_values: list[int]) -> dict[str, Any]:
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Any] = {"nearest_label_distribution": {}, "entropy": None, "top_prob": None}

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_step:
        by_mode[str(item["mode"])].append(item)

    for mode, items in sorted(by_mode.items()):
        items = sorted(items, key=lambda item: int(item["step"]))
        x_values = [int(item["step"]) for item in items]
        series = []
        for label in label_values:
            values = [
                (item["nearest_label_fraction"].get(str(label), 0.0) if item["nearest_label_fraction"] else 0.0)
                for item in items
            ]
            series.append((str(label), values, PALETTE.get(label, (0, 0, 0))))
        path = curves_dir / f"nearest_label_distribution_{mode}.png"
        _draw_line_chart(
            title=f"nearest-label distribution: {mode}",
            x_values=x_values,
            series=series,
            out_path=path,
            y_min=0.0,
            y_max=1.0,
        )
        paths["nearest_label_distribution"][mode] = str(path)

    modes = sorted(by_mode)
    common_x = sorted({int(item["step"]) for item in per_step})
    entropy_series = []
    top_prob_series = []
    for mode_index, mode in enumerate(modes):
        item_by_step = {int(item["step"]): item for item in by_mode[mode]}
        entropy_values = []
        top_prob_values = []
        for step in common_x:
            stats = (item_by_step.get(step) or {}).get("model_prob_stats")
            entropy_values.append(None if stats is None else stats.get("entropy_mean"))
            top_prob_values.append(None if stats is None else stats.get("top_prob_mean"))
        color = PALETTE.get(mode_index, (40, 40, 40))
        entropy_series.append((mode, entropy_values, color))
        top_prob_series.append((mode, top_prob_values, color))
    entropy_path = curves_dir / "entropy_by_mode.png"
    _draw_line_chart(
        title="image-token model entropy by mode",
        x_values=common_x,
        series=entropy_series,
        out_path=entropy_path,
        y_min=0.0,
    )
    paths["entropy"] = str(entropy_path)
    top_prob_path = curves_dir / "top_token_concentration_by_mode.png"
    _draw_line_chart(
        title="image-token top-token concentration by mode",
        x_values=common_x,
        series=top_prob_series,
        out_path=top_prob_path,
        y_min=0.0,
        y_max=1.0,
    )
    paths["top_prob"] = str(top_prob_path)
    return paths


def _write_decoded_grids(
    *,
    out_dir: Path,
    mode_batch: ModeBatch,
    final_tokens: torch.Tensor,
    nearest_indices: torch.Tensor,
    nearest_distances: torch.Tensor,
    nearest_labels: torch.Tensor,
    tokenizer,
    tokenizer_state: dict,
    raw_dataset,
    device: torch.device,
    label_to_string: dict[int, str],
    grid_cols: int,
    cell_size: int,
    decode_batch_size: int,
) -> dict[str, str]:
    mode_dir = out_dir / "decoded_grids" / mode_batch.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    decoded_chunks = []
    for start in range(0, final_tokens.shape[0], decode_batch_size):
        decoded_chunks.append(
            _decode_image_tokens(
                tokenizer,
                final_tokens[start : start + decode_batch_size],
                tokenizer_state,
                tuple(int(value) for value in tokenizer_state["grid_shape"]),
                device,
            ).cpu()
        )
    decoded = torch.cat(decoded_chunks, dim=0)

    output: dict[str, str] = {}
    for target_label in sorted(set(int(value) for value in mode_batch.target_labels.detach().cpu().tolist())):
        mask = mode_batch.target_labels.eq(target_label).detach().cpu()
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        cell_images = []
        for row in indices:
            nearest_index = int(nearest_indices[row].item())
            nearest_label = int(nearest_labels[row].item())
            generated_image = _tensor_to_pil(decoded[row], size=cell_size)
            nearest_image = _raw_image(raw_dataset, nearest_index, size=cell_size)
            pair = Image.new("RGB", (cell_size * 2, cell_size), color=(255, 255, 255))
            pair.paste(generated_image, (0, 0))
            pair.paste(nearest_image, (cell_size, 0))
            prompt_label = int(mode_batch.prompt_labels[row].item())
            prompt_text = mode_batch.prompt_texts[row]
            caption = "\n".join(
                [
                    f"target={_safe_label(target_label, label_to_string)}",
                    f"prompt={_safe_label(prompt_label, label_to_string) if prompt_label >= 0 else mode_batch.mode}:{prompt_text}",
                    f"seed={int(mode_batch.sample_seeds[row].item())}",
                    f"nn={_safe_label(nearest_label, label_to_string)} d={int(nearest_distances[row].item())}",
                ]
            )
            cell_images.append(_add_caption(pair, caption, height=45))
        path = mode_dir / f"target_{target_label}_{_safe_label(target_label, label_to_string)}.png"
        _make_grid(cell_images, cols=grid_cols).save(path)
        output[str(target_label)] = str(path)
    return output


def _distribution_vector(item: dict[str, Any], label_values: list[int]) -> list[float]:
    fractions = item.get("nearest_label_fraction") or {}
    return [float(fractions.get(str(label), 0.0)) for label in label_values]


def _l1(a: list[float], b: list[float]) -> float:
    return float(sum(abs(x - y) for x, y in zip(a, b)))


def _collapse_fraction(item: dict[str, Any], collapse_labels: list[int]) -> float:
    fractions = item.get("nearest_label_fraction") or {}
    return float(sum(float(fractions.get(str(label), 0.0)) for label in collapse_labels))


def _diagnose_phase(per_step: list[dict[str, Any]], *, mode: str, steps: int, collapse_labels: list[int]) -> dict[str, Any]:
    items = {int(item["step"]): item for item in per_step if item["mode"] == mode}
    checkpoints = {
        "init": 0,
        "early_10pct": max(1, int(round(steps * 0.1))),
        "middle_50pct": max(1, int(round(steps * 0.5))),
        "late_90pct": max(1, int(round(steps * 0.9))),
        "after_last_update": steps,
        "final_argmax": steps + 1,
    }
    fractions = {
        name: _round_float(_collapse_fraction(items[step], collapse_labels)) if step in items else None
        for name, step in checkpoints.items()
    }
    threshold = 0.8
    crossed_step = None
    for step in sorted(items):
        if _collapse_fraction(items[step], collapse_labels) >= threshold:
            crossed_step = step
            break
    final_fraction = fractions.get("final_argmax") or 0.0
    last_fraction = fractions.get("after_last_update") or 0.0
    if (fractions.get("init") or 0.0) >= threshold:
        phase = "initialization"
    elif crossed_step is not None and crossed_step <= max(1, int(round(steps * 0.1))):
        phase = "early_denoise"
    elif crossed_step is not None and crossed_step <= max(1, int(round(steps * 0.6))):
        phase = "middle_drift"
    elif final_fraction - last_fraction >= 0.25:
        phase = "final_argmax"
    elif crossed_step is not None:
        phase = "late_drift"
    else:
        phase = "not_thresholded_or_unclear"
    return {
        "mode": mode,
        "collapse_labels": collapse_labels,
        "threshold": threshold,
        "checkpoint_fractions": fractions,
        "first_threshold_step": crossed_step,
        "diagnosed_phase": phase,
    }


def _condition_control(per_step: list[dict[str, Any]], *, steps: int, label_values: list[int]) -> dict[str, Any]:
    final_step = steps + 1
    finals = {item["mode"]: item for item in per_step if int(item["step"]) == final_step}
    correct = finals.get("correct")
    shuffled = finals.get("shuffled")
    empty = finals.get("empty")
    random = finals.get("random")
    comparisons = {}
    if correct and shuffled:
        comparisons["correct_vs_shuffled_l1"] = _round_float(_l1(_distribution_vector(correct, label_values), _distribution_vector(shuffled, label_values)))
    if correct and empty:
        comparisons["correct_vs_empty_l1"] = _round_float(_l1(_distribution_vector(correct, label_values), _distribution_vector(empty, label_values)))
    if correct and random:
        comparisons["correct_vs_random_l1"] = _round_float(_l1(_distribution_vector(correct, label_values), _distribution_vector(random, label_values)))
    return {
        "final_prompt_match_correct": None if correct is None else correct.get("prompt_match_fraction"),
        "final_prompt_match_shuffled": None if shuffled is None else shuffled.get("prompt_match_fraction"),
        "final_target_match_empty": None if empty is None else empty.get("target_match_fraction"),
        "final_target_match_random": None if random is None else random.get("target_match_fraction"),
        "final_distribution_l1": comparisons,
        "interpretation": (
            "Text condition is controlling the trajectory only if correct and shuffled prompts follow their input prompt "
            "and differ clearly from empty/random. Similar final distributions across modes indicate weak conditioning."
        ),
    }


def _write_index(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = ["<h1>t2i collapse diagnostic</h1>"]
    rows.append("<h2>Curves</h2>")
    curves = summary.get("curves", {})
    for path in curves.get("nearest_label_distribution", {}).values():
        rows.append(f"<img src=\"{html.escape(str(Path(path).relative_to(out_dir)))}\" alt=\"curve\">")
    for key in ("entropy", "top_prob"):
        if curves.get(key):
            rows.append(f"<img src=\"{html.escape(str(Path(curves[key]).relative_to(out_dir)))}\" alt=\"{key}\">")
    rows.append("<h2>Decoded grids</h2>")
    for mode, grids in sorted((summary.get("decoded_grids") or {}).items()):
        rows.append(f"<h3>{html.escape(mode)}</h3>")
        for label, path in sorted(grids.items(), key=lambda item: int(item[0])):
            rows.append(f"<h4>target {html.escape(label)}</h4>")
            rows.append(f"<img src=\"{html.escape(str(Path(path).relative_to(out_dir)))}\" alt=\"grid\">")
    (out_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>t2i collapse diagnostic</title>",
                "<style>body{font-family:system-ui;margin:24px}img{max-width:100%;border:1px solid #ddd;margin:8px 0}</style>",
                "</head><body>",
                *rows,
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_markdown(out_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# t2i collapse diagnostic",
        "",
        f"- Config: `{summary['config']}`",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Steps: `{summary['steps']}`",
        f"- Temperature: `{summary['temperature']}`",
        f"- Seeds per label: `{summary['seeds_per_label']}`",
        f"- Modes: `{', '.join(summary['modes'])}`",
        "",
        "## Main Conclusion",
        "",
        f"- Correct-mode collapse phase: `{summary['collapse_phase']['correct']['diagnosed_phase']}`",
        f"- Collapse labels: `{summary['collapse_phase']['correct']['collapse_labels']}`",
        f"- Condition-control note: {summary['condition_control']['interpretation']}",
        "",
        "## Final Mode Metrics",
        "",
    ]
    for mode, metrics in sorted(summary["final_mode_metrics"].items()):
        lines.extend(
            [
                f"### {mode}",
                "",
                f"- target match: `{metrics.get('target_match_fraction')}`",
                f"- prompt match: `{metrics.get('prompt_match_fraction')}`",
                f"- nearest-label distribution: `{metrics.get('nearest_label_fraction')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Files",
            "",
            f"- per-step JSONL: `{summary['per_step_jsonl']}`",
            f"- per-step summary JSON: `{summary['per_step_summary_json']}`",
            f"- sample outcomes JSONL: `{summary['sample_outcomes_jsonl']}`",
            f"- position collapse JSON: `{summary['position_collapse_json']}`",
            f"- HTML index: `{out_dir / 'index.html'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def _run_mode(
    *,
    mode_batch: ModeBatch,
    model,
    layout: TaskLayout,
    schedule_tables: dict[str, Any],
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    temperature: float,
    steps: int,
    record_every: int,
    bank_tokens: torch.Tensor,
    bank_labels: torch.Tensor,
    bank_chunk_size: int,
    label_values: list[int],
    label_to_string: dict[int, str],
    position_collapse: dict[str, Any],
    top_prob_threshold: float,
    consensus_threshold: float,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    batch = mode_batch.condition_text.shape[0]
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    z_t = _initial_noise(layout=layout, sample_seeds=mode_batch.sample_seeds, valid_token_mask=valid_token_mask, device=device)

    text_targets = mode_batch.condition_text + layout.text_offset
    z_t[:, layout.text_slice] = build_flm_clean_state(text_targets, layout.vocab_size)
    effective_text_time_power = text_time_power
    if image_to_text_text_time_power is not None:
        effective_text_time_power = text_time_power

    records: list[dict[str, Any]] = []

    def record(step: int, phase: str, progress: float, probs: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tokens = z_t[:, layout.image_slice, : layout.codebook_size].argmax(dim=-1)
        item, nearest_indices, nearest_distances, nearest_labels = _step_summary(
            mode_batch=mode_batch,
            step=step,
            phase=phase,
            progress=progress,
            state_tokens=state_tokens,
            probs=None if probs is None else probs[:, layout.image_slice, : layout.codebook_size],
            bank_tokens=bank_tokens,
            bank_labels=bank_labels,
            bank_chunk_size=bank_chunk_size,
            label_values=label_values,
            label_to_string=label_to_string,
        )
        records.append(item)
        image_probs = None if probs is None else probs[:, layout.image_slice, : layout.codebook_size]
        _update_position_collapse(
            position_collapse,
            mode=mode_batch.mode,
            group_kind="all",
            group_value="all",
            step=step,
            phase=phase,
            probs=image_probs,
            state_tokens=state_tokens,
            top_prob_threshold=top_prob_threshold,
            consensus_threshold=consensus_threshold,
        )
        for label in label_values:
            mask = mode_batch.target_labels.eq(label)
            if int(mask.sum().item()) == 0:
                continue
            _update_position_collapse(
                position_collapse,
                mode=mode_batch.mode,
                group_kind="target",
                group_value=str(label),
                step=step,
                phase=phase,
                probs=None if image_probs is None else image_probs[mask],
                state_tokens=state_tokens[mask],
                top_prob_threshold=top_prob_threshold,
                consensus_threshold=consensus_threshold,
            )
        return nearest_indices, nearest_distances, nearest_labels

    record(0, "initial_noise_argmax", 0.0, None)
    final_probs = None
    for step in range(steps):
        progress = torch.full((batch,), step / max(steps, 1), device=device)
        next_progress = torch.full((batch,), (step + 1) / max(steps, 1), device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        next_t_pos = apply_schedule(
            progress=next_progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        t_pos = condition_clean_timesteps(t_pos, layout.image_seq_len, condition_image=False, condition_text=True)
        next_t_pos = condition_clean_timesteps(next_t_pos, layout.image_seq_len, condition_image=False, condition_text=True)
        logits = model(z_t, t_pos, modality_ids)
        logits = mask_logits(logits, layout=layout)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + (next_t_pos - t_pos).unsqueeze(-1).clamp_min(0.0) * v_t
        z_t[:, layout.text_slice] = build_flm_clean_state(text_targets, layout.vocab_size)
        if (step + 1) % record_every == 0 or (step + 1) == steps:
            record(step + 1, "after_update", float((step + 1) / max(steps, 1)), probs)

    final_t_pos = apply_schedule(
        progress=torch.ones(batch, device=device),
        modality_ids=modality_ids,
        schedule_tables=schedule_tables,
        image_time_power=image_time_power,
        text_time_power=effective_text_time_power,
    )
    final_t_pos = condition_clean_timesteps(final_t_pos, layout.image_seq_len, condition_image=False, condition_text=True)
    final_logits = mask_logits(model(z_t, final_t_pos, modality_ids), layout=layout)
    final_probs = torch.softmax(final_logits / max(temperature, 1e-4), dim=-1)
    final_tokens_all = final_logits.argmax(dim=-1)
    final_image_tokens = final_tokens_all[:, layout.image_slice]
    item, nearest_indices, nearest_distances, nearest_labels = _step_summary(
        mode_batch=mode_batch,
        step=steps + 1,
        phase="final_argmax",
        progress=1.0,
        state_tokens=final_image_tokens,
        probs=final_probs[:, layout.image_slice, : layout.codebook_size],
        bank_tokens=bank_tokens,
        bank_labels=bank_labels,
        bank_chunk_size=bank_chunk_size,
        label_values=label_values,
        label_to_string=label_to_string,
    )
    records.append(item)
    final_image_probs = final_probs[:, layout.image_slice, : layout.codebook_size]
    _update_position_collapse(
        position_collapse,
        mode=mode_batch.mode,
        group_kind="all",
        group_value="all",
        step=steps + 1,
        phase="final_argmax",
        probs=final_image_probs,
        state_tokens=final_image_tokens,
        top_prob_threshold=top_prob_threshold,
        consensus_threshold=consensus_threshold,
    )
    for label in label_values:
        mask = mode_batch.target_labels.eq(label)
        if int(mask.sum().item()) == 0:
            continue
        _update_position_collapse(
            position_collapse,
            mode=mode_batch.mode,
            group_kind="target",
            group_value=str(label),
            step=steps + 1,
            phase="final_argmax",
            probs=final_image_probs[mask],
            state_tokens=final_image_tokens[mask],
            top_prob_threshold=top_prob_threshold,
            consensus_threshold=consensus_threshold,
        )
    return records, final_image_tokens, nearest_indices, nearest_distances, nearest_labels


def _sample_outcomes(
    *,
    mode_batch: ModeBatch,
    nearest_indices: torch.Tensor,
    nearest_distances: torch.Tensor,
    nearest_labels: torch.Tensor,
    label_to_string: dict[int, str],
) -> list[dict[str, Any]]:
    outcomes = []
    for row in range(nearest_labels.shape[0]):
        target_label = int(mode_batch.target_labels[row].item())
        prompt_label = int(mode_batch.prompt_labels[row].item())
        nearest_label = int(nearest_labels[row].item())
        outcomes.append(
            {
                "mode": mode_batch.mode,
                "sample_index": row,
                "seed": int(mode_batch.sample_seeds[row].item()),
                "target_label": target_label,
                "target_text": _safe_label(target_label, label_to_string),
                "prompt_label": None if prompt_label < 0 else prompt_label,
                "prompt_text": mode_batch.prompt_texts[row],
                "nearest_index": int(nearest_indices[row].item()),
                "nearest_label": nearest_label,
                "nearest_text": _safe_label(nearest_label, label_to_string),
                "nearest_distance": int(nearest_distances[row].item()),
                "target_match": bool(nearest_label == target_label),
                "prompt_match": None if prompt_label < 0 else bool(nearest_label == prompt_label),
            }
        )
    return outcomes


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose text-to-image collapse without changing training or sampling algorithms.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seeds-per-label", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["correct", "shuffled", "empty", "random"])
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--bank-samples", type=int, default=5000)
    parser.add_argument("--bank-chunk-size", type=int, default=512)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=72)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--top-prob-collapse-threshold", type=float, default=0.5)
    parser.add_argument("--consensus-collapse-threshold", type=float, default=0.75)
    parser.add_argument("--collapse-labels", nargs="+", type=int, default=[9, 1, 7])
    args = parser.parse_args()

    if args.record_every < 1:
        raise ValueError("--record-every must be >= 1")
    if args.seeds_per_label < 32:
        raise ValueError("--seeds-per-label must be at least 32 for this diagnostic")

    config = load_config(args.config)
    checkpoint_config = load_config(args.checkpoint_config or args.config)
    steps = int(args.steps or config.sampling.steps)
    temperature = float(args.temperature if args.temperature is not None else config.sampling.temperature)
    base_seed = int(args.base_seed if args.base_seed is not None else config.train.seed)
    set_seed(base_seed)
    device = resolve_device(config.train.device, config.train.gpu_index)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = config.repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    model, tokenizer_state, layout = _load_stage2(
        config,
        device,
        checkpoint_config=checkpoint_config,
        checkpoint_path=Path(args.checkpoint_path),
    )
    text_metadata = metadata_from_state(tokenizer_state)
    label_values = [int(value) for value in text_metadata.label_values]
    label_to_string = {
        int(value): str(text)
        for value, text in zip(text_metadata.label_values, text_metadata.label_strings)
    }
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    dataset = TokenizedImageDataset(split_path(config, "test"))
    raw_dataset = build_image_dataset(config.dataset.name, config.paths.data_dir, train=False)
    bank_count = min(int(args.bank_samples), int(dataset.labels.shape[0]))
    if bank_count <= 0:
        raise ValueError("empty token bank")
    bank_tokens = dataset.image_tokens[:bank_count].to(device)
    bank_labels = dataset.labels[:bank_count].to(device)
    tokenizer = build_tokenizer(config, device=device)

    per_step: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    decoded_grids: dict[str, dict[str, str]] = {}
    final_mode_metrics: dict[str, Any] = {}
    position_collapse: dict[str, Any] = {}

    for mode in args.modes:
        mode_batch = _make_mode_batch(text_metadata, mode, args.seeds_per_label, base_seed, device)
        mode_records, final_tokens, nearest_indices, nearest_distances, nearest_labels = _run_mode(
            mode_batch=mode_batch,
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            image_time_power=config.sampling.image_time_power,
            text_time_power=config.sampling.text_time_power,
            image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
            temperature=temperature,
            steps=steps,
            record_every=args.record_every,
            bank_tokens=bank_tokens,
            bank_labels=bank_labels,
            bank_chunk_size=args.bank_chunk_size,
            label_values=label_values,
            label_to_string=label_to_string,
            position_collapse=position_collapse,
            top_prob_threshold=args.top_prob_collapse_threshold,
            consensus_threshold=args.consensus_collapse_threshold,
        )
        per_step.extend(mode_records)
        final_mode_metrics[mode] = mode_records[-1]
        all_outcomes.extend(
            _sample_outcomes(
                mode_batch=mode_batch,
                nearest_indices=nearest_indices,
                nearest_distances=nearest_distances,
                nearest_labels=nearest_labels,
                label_to_string=label_to_string,
            )
        )
        decoded_grids[mode] = _write_decoded_grids(
            out_dir=out_dir,
            mode_batch=mode_batch,
            final_tokens=final_tokens,
            nearest_indices=nearest_indices,
            nearest_distances=nearest_distances,
            nearest_labels=nearest_labels,
            tokenizer=tokenizer,
            tokenizer_state=tokenizer_state,
            raw_dataset=raw_dataset,
            device=device,
            label_to_string=label_to_string,
            grid_cols=args.grid_cols,
            cell_size=args.cell_size,
            decode_batch_size=args.decode_batch_size,
        )

    per_step_jsonl = out_dir / "per_step.jsonl"
    with per_step_jsonl.open("w", encoding="utf-8") as handle:
        for item in per_step:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    per_step_summary_json = out_dir / "per_step_summary.json"
    per_step_summary_json.write_text(json.dumps(per_step, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sample_outcomes_jsonl = out_dir / "sample_outcomes.jsonl"
    with sample_outcomes_jsonl.open("w", encoding="utf-8") as handle:
        for item in all_outcomes:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    position_collapse_json = out_dir / "position_collapse.json"
    position_collapse_json.write_text(json.dumps(position_collapse, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    curves = _write_curves(out_dir, per_step, label_values)

    collapse_phase = {
        mode: _diagnose_phase(per_step, mode=mode, steps=steps, collapse_labels=args.collapse_labels)
        for mode in args.modes
    }
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "out_dir": str(out_dir),
        "elapsed_sec": _round_float(time.time() - started, 3),
        "steps": steps,
        "temperature": temperature,
        "seeds_per_label": args.seeds_per_label,
        "modes": args.modes,
        "record_every": args.record_every,
        "bank_samples": bank_count,
        "label_values": label_values,
        "label_to_string": {str(key): value for key, value in label_to_string.items()},
        "collapse_labels": args.collapse_labels,
        "per_step_jsonl": str(per_step_jsonl),
        "per_step_summary_json": str(per_step_summary_json),
        "sample_outcomes_jsonl": str(sample_outcomes_jsonl),
        "position_collapse_json": str(position_collapse_json),
        "curves": curves,
        "decoded_grids": decoded_grids,
        "collapse_phase": collapse_phase,
        "condition_control": _condition_control(per_step, steps=steps, label_values=label_values),
        "final_mode_metrics": final_mode_metrics,
        "notes": [
            "This diagnostic does not train or change model parameters.",
            "Nearest-label metrics are token-space diagnostics; decoded grids remain the required visual check.",
            "Step 0 is image-token argmax from masked Gaussian initialization; step N is after the last Euler update; step N+1 is final logits argmax.",
        ],
    }
    summary_json = out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_index(out_dir, summary)
    _write_markdown(out_dir, summary)
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "report_md": str(out_dir / "report.md"),
                "index_html": str(out_dir / "index.html"),
                "elapsed_sec": summary["elapsed_sec"],
                "collapse_phase_correct": collapse_phase.get("correct", {}).get("diagnosed_phase"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
