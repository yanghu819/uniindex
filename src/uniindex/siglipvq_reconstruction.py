from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, prepare_assets, split_path
from .eval import _decode_image_tokens
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .text import metadata_from_state
from .tokenizer import build_tokenizer


def _tensor_to_pil(image: torch.Tensor, size: int = 112) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(array)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BICUBIC)
    return pil


def _make_grid(images: list[Image.Image], captions: list[str], cols: int = 4) -> Image.Image:
    cell_size = 112
    caption_height = 34
    rows = max((len(images) + cols - 1) // cols, 1)
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + caption_height)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (image, caption) in enumerate(zip(images, captions)):
        row = index // cols
        col = index % cols
        x = col * cell_size
        y = row * (cell_size + caption_height)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + caption_height), fill=(248, 248, 248))
        draw.multiline_text((x + 4, y + cell_size + 4), caption, fill=(0, 0, 0), font=font, spacing=2)
    return canvas


def probe_siglipvq_reconstruction(
    *,
    config: ProjectConfig,
    sample_count: int = 16,
    split: str = "test",
    run_context: RunContext | None = None,
) -> dict:
    if config.tokenizer.kind != "siglip_vq":
        raise ValueError("probe-siglipvq-reconstruction requires tokenizer.kind=siglip_vq")
    if sample_count < 1:
        raise ValueError(f"sample_count must be >= 1, got {sample_count}")
    if split not in {"train", "test"}:
        raise ValueError(f"split must be train or test, got {split}")

    ensure_project_dirs(config)
    set_seed(config.train.seed)
    prepare_assets(config)
    device = resolve_device(config.train.device, config.train.gpu_index)
    run_context = run_context or RunContext(config, "probe-siglipvq-reconstruction")
    run_context.set_device(device)

    tokenizer_state = load_tokenizer_state(config)
    text_metadata = metadata_from_state(tokenizer_state)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(classifier_path(config.paths.models_dir, config.dataset.name), config.dataset.name, device=device)
    loader = build_loader(
        split_path(config, split),
        batch_size=sample_count,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    batch = next(iter(loader))
    image_tokens = batch["image_tokens"][:sample_count].to(device)
    labels = batch["label"][:sample_count].to(device)
    with torch.inference_mode():
        decoded = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        predictions = classify_images(classifier, decoded, config.dataset.name)
    accuracy = float(predictions.eq(labels).float().mean().item())
    label_to_string = {value: text for value, text in zip(text_metadata.label_values, text_metadata.label_strings)}
    captions = [
        f"gt={label_to_string.get(int(label), str(int(label)))}\npred={label_to_string.get(int(pred), str(int(pred)))}"
        for label, pred in zip(labels.detach().cpu().tolist(), predictions.detach().cpu().tolist())
    ]
    grid = _make_grid([_tensor_to_pil(image) for image in decoded.detach().cpu()], captions)
    grid_path = run_context.log_path("reconstruction_grid.png")
    grid.save(grid_path)
    summary = {
        "source": "probe-siglipvq-reconstruction",
        "split": split,
        "sample_count": int(labels.numel()),
        "accuracy": accuracy,
        "grid_path": str(grid_path),
        "grid_shape": list(grid_shape),
        "image_size": int(tokenizer_state["image_size"]),
        "predictions": [
            {
                "target": label_to_string.get(int(label), str(int(label))),
                "prediction": label_to_string.get(int(pred), str(int(pred))),
            }
            for label, pred in zip(labels.detach().cpu().tolist(), predictions.detach().cpu().tolist())
        ],
    }
    summary_path = run_context.log_path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
