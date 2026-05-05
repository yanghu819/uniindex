from __future__ import annotations

import argparse
import html
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from uniindex.config import load_config
from uniindex.data import TokenizedImageDataset, split_path
from uniindex.datasets import build_image_dataset
from uniindex.eval import (
    _load_stage2,
    _sample_unified_with_logits,
    constrained_text_label_values,
    sample_unified,
)
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.text import decode_text_tokens, metadata_from_state, text_scoring_mask


def _prepare_pil(image: Image.Image, image_size: int) -> Image.Image:
    prepared = image.convert("RGB")
    if prepared.size != (image_size, image_size):
        prepared = prepared.resize((image_size, image_size), Image.BILINEAR)
    return prepared


def _safe_default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _wrap_line(text: str, width: int) -> list[str]:
    if len(text) <= width:
        return [text]
    return [text[start : start + width] for start in range(0, len(text), width)]


def _wrap_caption(caption: str, width: int) -> str:
    lines: list[str] = []
    for raw_line in caption.splitlines():
        if not raw_line:
            lines.append("")
            continue
        lines.extend(_wrap_line(raw_line, width))
    return "\n".join(lines)


def _resize_square(image: Image.Image, size: int) -> Image.Image:
    if image.size != (size, size):
        return image.resize((size, size), Image.NEAREST)
    return image


def _token_grid_image(
    tokens: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    codebook_size: int,
    size: int = 112,
) -> Image.Image:
    grid = tokens.detach().cpu().long().reshape(grid_shape)
    denom = max(codebook_size - 1, 1)
    values = grid.float().div(float(denom)).clamp(0.0, 1.0).numpy()
    red = (values * 255.0).round().astype(np.uint8)
    green = ((1.0 - np.abs(values * 2.0 - 1.0)) * 255.0).round().astype(np.uint8)
    blue = ((1.0 - values) * 255.0).round().astype(np.uint8)
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(rgb, mode="RGB").resize((size, size), Image.NEAREST)


def _make_grid(
    images: list[Image.Image],
    captions: list[str],
    *,
    cols: int,
    cell_width: int,
    image_height: int,
    caption_height: int,
) -> Image.Image:
    if len(images) != len(captions):
        raise ValueError("images and captions must have the same length")
    rows = max((len(images) + cols - 1) // cols, 1)
    canvas = Image.new(
        "RGB",
        (cols * cell_width, rows * (image_height + caption_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = _safe_default_font()
    for index, (image, caption) in enumerate(zip(images, captions)):
        row = index // cols
        col = index % cols
        x = col * cell_width
        y = row * (image_height + caption_height)
        canvas.paste(image, (x, y))
        draw.rectangle(
            (x, y + image_height, x + cell_width, y + image_height + caption_height),
            fill=(247, 247, 247),
        )
        wrapped = _wrap_caption(caption, max((cell_width - 8) // 6, 12))
        draw.multiline_text((x + 4, y + image_height + 4), wrapped, fill=(0, 0, 0), font=font, spacing=2)
    return canvas


def _add_strip(image: Image.Image, labels: list[str]) -> Image.Image:
    strip_height = 16
    out = Image.new("RGB", (image.width, image.height + strip_height), color=(255, 255, 255))
    out.paste(image, (0, strip_height))
    draw = ImageDraw.Draw(out)
    font = _safe_default_font()
    if labels:
        segment_width = image.width // len(labels)
        for index, label in enumerate(labels):
            x = index * segment_width
            draw.text((x + 3, 2), label, fill=(0, 0, 0), font=font)
    return out


def _compose_generation_cell(target: Image.Image, generated_tokens: Image.Image, nearest: Image.Image) -> Image.Image:
    tile = 96
    gap = 6
    target = _resize_square(target, tile)
    generated_tokens = _resize_square(generated_tokens, tile)
    nearest = _resize_square(nearest, tile)
    width = tile * 3 + gap * 2
    canvas = Image.new("RGB", (width, tile), color=(255, 255, 255))
    canvas.paste(target, (0, 0))
    canvas.paste(generated_tokens, (tile + gap, 0))
    canvas.paste(nearest, ((tile + gap) * 2, 0))
    return _add_strip(canvas, ["target", "genTok", "nn"])


def _balanced_indices(labels: torch.Tensor, label_values: tuple[int, ...], total: int) -> list[int]:
    wanted = max(int(total), 0)
    if wanted == 0:
        return []
    per_label = max(1, math.ceil(wanted / max(len(label_values), 1)))
    allowed = {int(value) for value in label_values}
    counts: Counter[int] = Counter()
    selected: list[int] = []
    used: set[int] = set()
    for index, label_tensor in enumerate(labels):
        label = int(label_tensor)
        if label not in allowed or counts[label] >= per_label:
            continue
        selected.append(index)
        used.add(index)
        counts[label] += 1
        if len(selected) >= wanted:
            return selected
    for index in range(labels.shape[0]):
        if index in used:
            continue
        selected.append(index)
        if len(selected) >= wanted:
            break
    return selected


def _load_raw_dataset(config):
    try:
        return build_image_dataset(config.dataset.name, config.paths.data_dir, train=False)
    except Exception as exc:  # pragma: no cover - fallback is for remote data drift.
        print(f"warning: raw dataset unavailable, falling back to token heatmaps: {exc}")
        return None


def _raw_or_token_image(
    *,
    raw_dataset,
    index: int,
    image_tokens: torch.Tensor,
    grid_shape: tuple[int, int],
    codebook_size: int,
    image_size: int,
    display_size: int,
) -> Image.Image:
    if raw_dataset is not None:
        image, _ = raw_dataset[index]
        return _prepare_pil(image, image_size).resize((display_size, display_size), Image.BILINEAR)
    return _token_grid_image(image_tokens, grid_shape, codebook_size=codebook_size, size=display_size)


def _nearest_token_matches(
    generated: torch.Tensor,
    bank_tokens: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_dist = torch.full(
        (generated.shape[0],),
        generated.shape[1] + 1,
        dtype=torch.long,
        device=generated.device,
    )
    best_index = torch.full((generated.shape[0],), -1, dtype=torch.long, device=generated.device)
    for start in range(0, bank_tokens.shape[0], chunk_size):
        chunk = bank_tokens[start : start + chunk_size]
        distances = generated[:, None, :].ne(chunk[None, :, :]).sum(dim=-1)
        chunk_best_dist, chunk_best_index = distances.min(dim=1)
        update = chunk_best_dist < best_dist
        best_dist = torch.where(update, chunk_best_dist, best_dist)
        best_index = torch.where(update, chunk_best_index + start, best_index)
    return best_index, best_dist


def _resolve_output_dir(config, raw_out_dir: str | None) -> Path:
    if raw_out_dir:
        out_dir = Path(raw_out_dir)
        if not out_dir.is_absolute():
            out_dir = config.repo_root / out_dir
        return out_dir
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return config.paths.logs_dir / "visualizations" / f"flm_cases_{stamp}"


def _write_html(out_dir: Path, summary: dict) -> Path:
    metrics = json.dumps(summary["metrics"], indent=2, ensure_ascii=False)
    html_path = out_dir / "index.html"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "<meta charset=\"utf-8\">",
                "<title>FLM case visualization</title>",
                "<style>",
                "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;line-height:1.45}",
                "img{max-width:100%;border:1px solid #ddd}",
                "pre{background:#f6f8fa;padding:12px;overflow:auto}",
                "</style>",
                "</head>",
                "<body>",
                "<h1>FLM case visualization</h1>",
                f"<p>config: {html.escape(summary['config'])}</p>",
                f"<p>checkpoint: {html.escape(summary['checkpoint'])}</p>",
                f"<p>sampling steps: {summary['sampling_steps']}, temperature: {summary['temperature']}</p>",
                "<h2>Image to text understanding</h2>",
                "<img src=\"understanding_cases.png\" alt=\"understanding cases\">",
                "<h2>Text to image generation</h2>",
                "<p>Generation uses token-space visualization: target raw image, generated image-token grid, nearest real test image.</p>",
                "<img src=\"generation_cases_token_nn.png\" alt=\"generation token nearest-neighbor cases\">",
                "<h2>Case metrics</h2>",
                f"<pre>{html.escape(metrics)}</pre>",
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return html_path


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize FLM image-to-text and text-to-image cases without decoder/classifier dependencies."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--understanding-cases", type=int, default=20)
    parser.add_argument("--generation-cases", type=int, default=20)
    parser.add_argument("--bank-samples", type=int, default=5000)
    parser.add_argument("--bank-chunk-size", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_config = load_config(args.checkpoint_config or args.config)
    sampling_steps = args.sampling_steps or config.sampling.steps
    temperature = args.temperature if args.temperature is not None else config.sampling.temperature
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    model, tokenizer_state, layout = _load_stage2(
        config,
        device,
        checkpoint_config=checkpoint_config,
        checkpoint_path=checkpoint_path,
    )
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    grid_shape = tuple(int(value) for value in tokenizer_state["grid_shape"])

    dataset = TokenizedImageDataset(split_path(config, "test"))
    labels = dataset.labels.cpu()
    label_to_string = {int(value): text for value, text in zip(text_metadata.label_values, text_metadata.label_strings)}
    raw_dataset = _load_raw_dataset(config)

    understanding_indices = _balanced_indices(labels, text_metadata.label_values, args.understanding_cases)
    generation_indices = _balanced_indices(labels, text_metadata.label_values, args.generation_cases)

    understanding_image_tokens = dataset.image_tokens[understanding_indices].to(device)
    understanding_text_tokens = dataset.text_tokens[understanding_indices].to(device)
    understanding_labels = dataset.labels[understanding_indices].to(device)
    sampled_tokens, final_logits = _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        temperature=temperature,
        steps=sampling_steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        condition_image_tokens=understanding_image_tokens,
        condition_text_tokens=None,
    )
    sampled_text = sampled_tokens[:, layout.text_slice] - layout.text_offset
    sampled_strings = decode_text_tokens(sampled_text, text_metadata)
    target_strings = decode_text_tokens(understanding_text_tokens.cpu(), text_metadata)
    constrained_labels = constrained_text_label_values(
        final_logits[:, layout.text_slice],
        text_metadata,
        codebook_size=layout.codebook_size,
    )
    score_mask = text_scoring_mask(understanding_text_tokens.cpu(), text_metadata, include_bos=False, include_eos=True)
    text_token_correct = sampled_text.cpu().eq(understanding_text_tokens.cpu()).logical_and(score_mask)
    text_token_total = score_mask.sum(dim=1).clamp_min(1)
    text_token_acc = text_token_correct.sum(dim=1).float() / text_token_total.float()

    understanding_images: list[Image.Image] = []
    understanding_captions: list[str] = []
    understanding_cases: list[dict] = []
    for row, index in enumerate(understanding_indices):
        target = target_strings[row]
        pred = sampled_strings[row]
        label = int(understanding_labels[row].item())
        constrained = int(constrained_labels[row].item())
        exact = pred == target
        understanding_images.append(
            _raw_or_token_image(
                raw_dataset=raw_dataset,
                index=index,
                image_tokens=dataset.image_tokens[index],
                grid_shape=grid_shape,
                codebook_size=layout.codebook_size,
                image_size=config.tokenizer.image_size,
                display_size=112,
            )
        )
        understanding_captions.append(
            "\n".join(
                [
                    f"idx={index} label={label}",
                    f"gt={target}",
                    f"pred={pred or '<empty>'}",
                    f"forced={label_to_string.get(constrained, str(constrained))} {'OK' if constrained == label else 'BAD'}",
                ]
            )
        )
        understanding_cases.append(
            {
                "index": int(index),
                "label": label,
                "target": target,
                "prediction": pred,
                "free_exact": bool(exact),
                "constrained_label": constrained,
                "constrained_correct": bool(constrained == label),
                "text_token_accuracy": float(text_token_acc[row].item()),
            }
        )

    bank_count = min(max(args.bank_samples, 1), dataset.labels.shape[0])
    bank_tokens = dataset.image_tokens[:bank_count].to(device)
    bank_labels = dataset.labels[:bank_count].to(device)
    generation_image_tokens = dataset.image_tokens[generation_indices].to(device)
    generation_text_tokens = dataset.text_tokens[generation_indices].to(device)
    generation_labels = dataset.labels[generation_indices].to(device)
    generated = sample_unified(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        temperature=temperature,
        steps=sampling_steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        condition_image_tokens=None,
        condition_text_tokens=generation_text_tokens,
    )[:, layout.image_slice]
    nearest_indices, nearest_distances = _nearest_token_matches(
        generated,
        bank_tokens,
        chunk_size=args.bank_chunk_size,
    )
    nearest_labels = bank_labels.index_select(0, nearest_indices)
    target_token_acc = generated.eq(generation_image_tokens).float().mean(dim=1)

    generation_images: list[Image.Image] = []
    generation_captions: list[str] = []
    generation_cases: list[dict] = []
    for row, index in enumerate(generation_indices):
        label = int(generation_labels[row].item())
        nearest_index = int(nearest_indices[row].item())
        nearest_label = int(nearest_labels[row].item())
        target_image = _raw_or_token_image(
            raw_dataset=raw_dataset,
            index=index,
            image_tokens=dataset.image_tokens[index],
            grid_shape=grid_shape,
            codebook_size=layout.codebook_size,
            image_size=config.tokenizer.image_size,
            display_size=96,
        )
        generated_image = _token_grid_image(
            generated[row],
            grid_shape,
            codebook_size=layout.codebook_size,
            size=96,
        )
        nearest_image = _raw_or_token_image(
            raw_dataset=raw_dataset,
            index=nearest_index,
            image_tokens=dataset.image_tokens[nearest_index],
            grid_shape=grid_shape,
            codebook_size=layout.codebook_size,
            image_size=config.tokenizer.image_size,
            display_size=96,
        )
        generation_images.append(_compose_generation_cell(target_image, generated_image, nearest_image))
        generation_captions.append(
            "\n".join(
                [
                    f"cond={label_to_string.get(label, str(label))} idx={index}",
                    f"nn={label_to_string.get(nearest_label, str(nearest_label))} nn_idx={nearest_index}",
                    f"dist={int(nearest_distances[row].item())} pair_tok={target_token_acc[row].item():.3f}",
                ]
            )
        )
        generation_cases.append(
            {
                "index": int(index),
                "condition_label": label,
                "condition_text": label_to_string.get(label, str(label)),
                "nearest_index": nearest_index,
                "nearest_label": nearest_label,
                "nearest_text": label_to_string.get(nearest_label, str(nearest_label)),
                "nearest_distance": int(nearest_distances[row].item()),
                "nearest_label_correct": bool(nearest_label == label),
                "paired_target_token_accuracy": float(target_token_acc[row].item()),
            }
        )

    out_dir = _resolve_output_dir(config, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    understanding_grid = _make_grid(
        understanding_images,
        understanding_captions,
        cols=5,
        cell_width=112,
        image_height=112,
        caption_height=76,
    )
    generation_grid = _make_grid(
        generation_images,
        generation_captions,
        cols=2,
        cell_width=300,
        image_height=112,
        caption_height=62,
    )
    understanding_path = out_dir / "understanding_cases.png"
    generation_path = out_dir / "generation_cases_token_nn.png"
    understanding_grid.save(understanding_path)
    generation_grid.save(generation_path)

    metrics = {
        "understanding_case_free_exact": sum(case["free_exact"] for case in understanding_cases)
        / max(len(understanding_cases), 1),
        "understanding_case_constrained_accuracy": sum(case["constrained_correct"] for case in understanding_cases)
        / max(len(understanding_cases), 1),
        "understanding_case_mean_text_token_accuracy": sum(case["text_token_accuracy"] for case in understanding_cases)
        / max(len(understanding_cases), 1),
        "generation_case_token_nn_accuracy": sum(case["nearest_label_correct"] for case in generation_cases)
        / max(len(generation_cases), 1),
        "generation_case_mean_nearest_distance": sum(case["nearest_distance"] for case in generation_cases)
        / max(len(generation_cases), 1),
        "generation_case_mean_paired_target_token_accuracy": sum(
            case["paired_target_token_accuracy"] for case in generation_cases
        )
        / max(len(generation_cases), 1),
        "generation_nearest_label_counts": Counter(case["nearest_label"] for case in generation_cases).most_common(),
        "bank_samples": bank_count,
    }
    summary = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else "latest",
        "out_dir": str(out_dir),
        "sampling_steps": sampling_steps,
        "temperature": temperature,
        "understanding_cases_png": str(understanding_path),
        "generation_cases_token_nn_png": str(generation_path),
        "metrics": metrics,
        "understanding_cases": understanding_cases,
        "generation_cases": generation_cases,
        "notes": [
            "generation image is visualized as image-token heatmap because this script intentionally avoids decoder/classifier dependencies",
            "nearest-neighbor label is a token-space proxy, not a pixel decoder or classifier score",
        ],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_path = _write_html(out_dir, summary)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "understanding_cases_png": str(understanding_path),
                "generation_cases_token_nn_png": str(generation_path),
                "summary_json": str(summary_path),
                "index_html": str(html_path),
                "metrics": metrics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
