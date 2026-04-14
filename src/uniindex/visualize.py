from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _decode_image_tokens, _load_stage2, sample_unified
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .tokenizer import build_tokenizer


def _tensor_to_pil(image: torch.Tensor, size: int = 112) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    pil = Image.fromarray(array)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BICUBIC)
    return pil


def _make_grid(images: list[Image.Image], captions: list[str], cols: int, cell_size: int = 112, caption_height: int = 42) -> Image.Image:
    if len(images) != len(captions):
        raise ValueError("images and captions must have the same length")
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


@torch.inference_mode()
def export_visualizations(config: ProjectConfig, run_context: RunContext | None = None) -> dict[str, str]:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "visualize")
    run_context.set_device(device)

    model, tokenizer_state, image_seq_len = _load_stage2(config, device)
    codebook_size = int(tokenizer_state["codebook_size"])
    num_labels = len(config.labels.values)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(classifier_path(config.paths.models_dir), device=device)
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=16,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    batch = next(iter(test_loader))
    image_tokens = batch["image_tokens"].to(device)
    labels = batch["label"].to(device)
    decoded_real = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)

    sampled_labels = sample_unified(
        model=model,
        codebook_size=codebook_size,
        num_labels=num_labels,
        image_seq_len=image_seq_len,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        label_time_power=config.sampling.label_time_power,
        image_to_label_label_time_power=config.sampling.image_to_label_label_time_power,
        condition_image_tokens=image_tokens,
        condition_labels=None,
    )[:, image_seq_len] - codebook_size

    label_conditions = torch.arange(num_labels, device=device)
    sampled_images = sample_unified(
        model=model,
        codebook_size=codebook_size,
        num_labels=num_labels,
        image_seq_len=image_seq_len,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        label_time_power=config.sampling.label_time_power,
        image_to_label_label_time_power=config.sampling.image_to_label_label_time_power,
        condition_image_tokens=None,
        condition_labels=label_conditions,
    )[:, :image_seq_len]
    decoded_generated = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
    generated_preds = classify_images(classifier, decoded_generated)

    unconditional = sample_unified(
        model=model,
        codebook_size=codebook_size,
        num_labels=num_labels,
        image_seq_len=image_seq_len,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        label_time_power=config.sampling.label_time_power,
        image_to_label_label_time_power=config.sampling.image_to_label_label_time_power,
        batch_size=16,
        condition_image_tokens=None,
        condition_labels=None,
    )
    unconditional_images = _decode_image_tokens(tokenizer, unconditional[:, :image_seq_len], tokenizer_state, grid_shape, device)
    unconditional_clf = classify_images(classifier, unconditional_images)
    unconditional_tokens = unconditional[:, image_seq_len] - codebook_size

    image_to_label_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in decoded_real[:16]],
        captions=[f"gt={int(gt)}\npred={int(pred)}" for gt, pred in zip(labels[:16], sampled_labels[:16])],
        cols=4,
    )
    label_to_image_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in decoded_generated[:10]],
        captions=[f"cond={index}\nclf={int(pred)}" for index, pred in enumerate(generated_preds[:10])],
        cols=5,
    )
    unconditional_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in unconditional_images[:16]],
        captions=[f"tok={int(tok)}\nclf={int(pred)}" for tok, pred in zip(unconditional_tokens[:16], unconditional_clf[:16])],
        cols=4,
    )

    image_to_label_path = run_context.log_path("visuals/image_to_label_grid.png")
    label_to_image_path = run_context.log_path("visuals/label_to_image_grid.png")
    unconditional_path = run_context.log_path("visuals/unconditional_grid.png")
    image_to_label_grid.save(image_to_label_path)
    label_to_image_grid.save(label_to_image_path)
    unconditional_grid.save(unconditional_path)

    summary = {
        "image_to_label_grid": str(image_to_label_path),
        "label_to_image_grid": str(label_to_image_path),
        "unconditional_grid": str(unconditional_path),
        "image_to_label_pairs": [
            {"gt": int(gt), "pred": int(pred)} for gt, pred in zip(labels[:16].tolist(), sampled_labels[:16].tolist())
        ],
        "label_to_image_pairs": [
            {"condition": int(index), "classifier_pred": int(pred)} for index, pred in enumerate(generated_preds[:10].tolist())
        ],
        "unconditional_pairs": [
            {"token_label": int(tok), "classifier_pred": int(pred)}
            for tok, pred in zip(unconditional_tokens[:16].tolist(), unconditional_clf[:16].tolist())
        ],
    }
    summary_path = run_context.log_path("visuals/summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if own_context:
        run_context.update_status("ok")
    return {key: value for key, value in summary.items() if isinstance(value, str)}
