from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from uniindex.config import load_config
from uniindex.data import TokenizedImageDataset, split_path
from uniindex.datasets import build_image_dataset
from uniindex.eval import _decode_image_tokens, _load_stage2, sample_unified
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.text import metadata_from_state
from uniindex.tokenizer import build_tokenizer


def _nearest_matches(
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


def _safe_label(value: int, label_to_string: dict[int, str]) -> str:
    return label_to_string.get(int(value), str(int(value)))


def _condition_batch(text_metadata, repeats_per_label: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    condition_rows = []
    condition_labels = []
    for row_index, label_value in enumerate(text_metadata.label_values):
        condition_rows.append(text_metadata.label_text_tokens[row_index].repeat(repeats_per_label, 1))
        condition_labels.extend([int(label_value)] * repeats_per_label)
    return torch.cat(condition_rows, dim=0).to(device), torch.tensor(condition_labels, dtype=torch.long, device=device)


def _diversity_metrics(generated: torch.Tensor, condition_labels: torch.Tensor) -> dict[str, object]:
    unique_sequences = torch.unique(generated.detach().cpu(), dim=0).shape[0]
    per_condition = {}
    for label in sorted(set(int(value) for value in condition_labels.detach().cpu().tolist())):
        rows = generated[condition_labels == label]
        if rows.shape[0] < 2:
            mean_distance = 0.0
        else:
            distances = rows[:, None, :].ne(rows[None, :, :]).sum(dim=-1).float()
            tri = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
            mean_distance = float(distances[tri].mean().item())
        per_condition[str(label)] = {
            "unique_generated_token_sequences": int(torch.unique(rows.detach().cpu(), dim=0).shape[0]),
            "mean_pairwise_hamming": mean_distance,
        }
    return {
        "unique_generated_token_sequences": int(unique_sequences),
        "per_condition_diversity": per_condition,
    }


def _run_combo(
    *,
    model,
    tokenizer,
    tokenizer_state: dict,
    layout,
    schedule_tables: dict,
    config,
    text_metadata,
    raw_dataset,
    bank_tokens: torch.Tensor,
    bank_labels: torch.Tensor,
    steps: int,
    temperature: float,
    repeats_per_label: int,
    seed: int,
    out_dir: Path,
    grid_cols: int,
    cell_size: int,
    bank_chunk_size: int,
    sample_batch_size: int,
) -> dict[str, object]:
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    condition_text, condition_labels = _condition_batch(text_metadata, repeats_per_label, device)
    generated_chunks = []
    for start in range(0, condition_text.shape[0], sample_batch_size):
        text_chunk = condition_text[start : start + sample_batch_size]
        generated_chunks.append(
            sample_unified(
                model=model,
                layout=layout,
                schedule_tables=schedule_tables,
                temperature=temperature,
                steps=steps,
                image_time_power=config.sampling.image_time_power,
                text_time_power=config.sampling.text_time_power,
                image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
                condition_image_tokens=None,
                condition_text_tokens=text_chunk,
            )[:, layout.image_slice]
        )
    generated = torch.cat(generated_chunks, dim=0)
    nearest_indices, nearest_distances = _nearest_matches(
        generated,
        bank_tokens,
        chunk_size=bank_chunk_size,
    )
    nearest_labels = bank_labels.index_select(0, nearest_indices)
    decoded_chunks = []
    for start in range(0, generated.shape[0], sample_batch_size):
        decoded_chunks.append(
            _decode_image_tokens(
                tokenizer,
                generated[start : start + sample_batch_size],
                tokenizer_state,
                tuple(int(value) for value in tokenizer_state["grid_shape"]),
                device,
            ).cpu()
        )
    decoded = torch.cat(decoded_chunks, dim=0)

    label_to_string = {int(value): text for value, text in zip(text_metadata.label_values, text_metadata.label_strings)}
    cell_images = []
    records = []
    per_condition_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for row in range(generated.shape[0]):
        condition_label = int(condition_labels[row].item())
        nearest_index = int(nearest_indices[row].item())
        nearest_label = int(nearest_labels[row].item())
        per_condition_counts[str(condition_label)][nearest_label] += 1
        decoded_image = _tensor_to_pil(decoded[row], size=cell_size)
        nearest_image = _raw_image(raw_dataset, nearest_index, size=cell_size)
        pair = Image.new("RGB", (cell_size * 2, cell_size), color=(255, 255, 255))
        pair.paste(decoded_image, (0, 0))
        pair.paste(nearest_image, (cell_size, 0))
        caption = "\n".join(
            [
                f"cond={_safe_label(condition_label, label_to_string)}",
                f"nn={_safe_label(nearest_label, label_to_string)} d={int(nearest_distances[row].item())}",
            ]
        )
        cell_images.append(_add_caption(pair, caption, height=30))
        records.append(
            {
                "sample_index": row,
                "condition_label": condition_label,
                "condition_text": _safe_label(condition_label, label_to_string),
                "nearest_index": nearest_index,
                "nearest_label": nearest_label,
                "nearest_text": _safe_label(nearest_label, label_to_string),
                "nearest_distance": int(nearest_distances[row].item()),
                "nearest_label_correct": bool(nearest_label == condition_label),
            }
        )

    grid_path = out_dir / f"decoded_grid_steps{steps:03d}_temp{temperature:.2f}.png"
    _make_grid(cell_images, cols=grid_cols).save(grid_path)
    nearest_label_counts = Counter(int(value) for value in nearest_labels.detach().cpu().tolist())
    metrics = {
        "steps": steps,
        "temperature": temperature,
        "seed": seed,
        "total": int(generated.shape[0]),
        "repeats_per_label": repeats_per_label,
        "token_nn_accuracy": float(nearest_labels.eq(condition_labels).float().mean().item()),
        "mean_nearest_distance": float(nearest_distances.float().mean().item()),
        "unique_nearest_indices": int(torch.unique(nearest_indices.detach().cpu()).numel()),
        "unique_nearest_labels": int(torch.unique(nearest_labels.detach().cpu()).numel()),
        "nearest_label_counts": nearest_label_counts.most_common(),
        "per_condition_nearest_label_counts": {
            label: counter.most_common() for label, counter in sorted(per_condition_counts.items())
        },
        **_diversity_metrics(generated, condition_labels),
        "decoded_grid": str(grid_path),
        "records": records,
    }
    metrics_path = out_dir / f"summary_steps{steps:03d}_temp{temperature:.2f}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


def _write_index(out_dir: Path, payload: dict[str, object]) -> None:
    rows = []
    for combo in payload["combos"]:
        image_name = Path(combo["decoded_grid"]).name
        rows.append(
            "\n".join(
                [
                    f"<h2>steps={combo['steps']} temp={combo['temperature']}</h2>",
                    f"<p>token-NN acc={combo['token_nn_accuracy']:.4f}, unique labels={combo['unique_nearest_labels']}, unique NN={combo['unique_nearest_indices']}</p>",
                    f"<img src=\"{html.escape(image_name)}\" alt=\"{html.escape(image_name)}\">",
                ]
            )
        )
    (out_dir / "index.html").write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset=\"utf-8\"><title>Decoded text-to-image sweep</title>",
                "<style>body{font-family:system-ui;margin:24px}img{max-width:100%;border:1px solid #ddd}</style>",
                "</head><body>",
                "<h1>Decoded text-to-image sweep</h1>",
                *rows,
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Decode text-to-image samples and summarize token-space collapse.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[256])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.7])
    parser.add_argument("--repeats-per-label", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bank-samples", type=int, default=5000)
    parser.add_argument("--bank-chunk-size", type=int, default=512)
    parser.add_argument("--sample-batch-size", type=int, default=40)
    parser.add_argument("--grid-cols", type=int, default=8)
    parser.add_argument("--cell-size", type=int, default=80)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_config = load_config(args.checkpoint_config or args.config)
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
    tokenizer = build_tokenizer(config, device=device)
    dataset = TokenizedImageDataset(split_path(config, "test"))
    raw_dataset = build_image_dataset(config.dataset.name, config.paths.data_dir, train=False)
    bank_count = min(args.bank_samples, dataset.labels.shape[0])
    bank_tokens = dataset.image_tokens[:bank_count].to(device)
    bank_labels = dataset.labels[:bank_count].to(device)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = config.repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    combos = []
    base_seed = config.train.seed if args.seed is None else args.seed
    for steps in args.steps:
        for temp in args.temperatures:
            combo_seed = base_seed + steps * 1000 + int(round(temp * 100))
            combos.append(
                _run_combo(
                    model=model,
                    tokenizer=tokenizer,
                    tokenizer_state=tokenizer_state,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    config=config,
                    text_metadata=text_metadata,
                    raw_dataset=raw_dataset,
                    bank_tokens=bank_tokens,
                    bank_labels=bank_labels,
                    steps=steps,
                    temperature=temp,
                    repeats_per_label=args.repeats_per_label,
                    seed=combo_seed,
                    out_dir=out_dir,
                    grid_cols=args.grid_cols,
                    cell_size=args.cell_size,
                    bank_chunk_size=args.bank_chunk_size,
                    sample_batch_size=args.sample_batch_size,
                )
            )

    payload = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else "latest",
        "out_dir": str(out_dir),
        "elapsed_sec": round(time.time() - started, 3),
        "bank_samples": int(bank_tokens.shape[0]),
        "combos": combos,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_index(out_dir, payload)
    print(json.dumps({"out_dir": str(out_dir), "summary_json": str(out_dir / "summary.json"), "combos": len(combos)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
