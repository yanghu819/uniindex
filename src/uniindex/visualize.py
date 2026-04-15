from __future__ import annotations

import json

import torch
from PIL import Image, ImageDraw, ImageFont

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _decode_image_tokens, _load_stage2, sample_unified
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .text import decode_text_tokens, label_values_from_text_tokens, metadata_from_state
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
    text_metadata = metadata_from_state(tokenizer_state)
    codebook_size = int(tokenizer_state["codebook_size"])
    text_seq_len = int(text_metadata.seq_len)
    text_vocab_size = int(text_metadata.vocab_size)
    schedule_tables = build_schedule_tables(config, image_vocab_size=codebook_size, text_vocab_size=text_vocab_size)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(classifier_path(config.paths.models_dir, config.dataset.name), config.dataset.name, device=device)
    label_to_string = {value: text for value, text in zip(text_metadata.label_values, text_metadata.label_strings)}
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=16,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    batch = next(iter(test_loader))
    image_tokens = batch["image_tokens"].to(device)
    text_tokens = batch["text_tokens"].to(device)
    labels = batch["label"].to(device)
    decoded_real = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)

    sampled_text = sample_unified(
        model=model,
        codebook_size=codebook_size,
        text_vocab_size=text_vocab_size,
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        schedule_tables=schedule_tables,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        condition_image_tokens=image_tokens,
        condition_text_tokens=None,
    )[:, image_seq_len:] - codebook_size
    sampled_text_strings = decode_text_tokens(sampled_text, text_metadata)
    gt_text_strings = decode_text_tokens(text_tokens, text_metadata)

    class_text_tokens = text_metadata.label_text_tokens.to(device)
    sampled_images = sample_unified(
        model=model,
        codebook_size=codebook_size,
        text_vocab_size=text_vocab_size,
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        schedule_tables=schedule_tables,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        condition_image_tokens=None,
        condition_text_tokens=class_text_tokens,
    )[:, :image_seq_len]
    decoded_generated = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
    generated_preds = classify_images(classifier, decoded_generated, config.dataset.name)

    unconditional = sample_unified(
        model=model,
        codebook_size=codebook_size,
        text_vocab_size=text_vocab_size,
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        schedule_tables=schedule_tables,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        text_time_power=config.sampling.text_time_power,
        image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
        batch_size=16,
        condition_image_tokens=None,
        condition_text_tokens=None,
    )
    unconditional_images = _decode_image_tokens(tokenizer, unconditional[:, :image_seq_len], tokenizer_state, grid_shape, device)
    unconditional_clf = classify_images(classifier, unconditional_images, config.dataset.name)
    unconditional_text = unconditional[:, image_seq_len:] - codebook_size
    unconditional_text_strings = decode_text_tokens(unconditional_text, text_metadata)
    unconditional_text_values = label_values_from_text_tokens(unconditional_text, text_metadata)

    image_to_text_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in decoded_real[:16]],
        captions=[f"gt={gt}\npred={pred}" for gt, pred in zip(gt_text_strings[:16], sampled_text_strings[:16])],
        cols=4,
    )
    text_to_image_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in decoded_generated[:10]],
        captions=[
            f"cond={condition}\nclf={label_to_string.get(int(pred), str(int(pred)))}"
            for condition, pred in zip(text_metadata.label_strings[:10], generated_preds[:10].tolist())
        ],
        cols=5,
    )
    unconditional_grid = _make_grid(
        images=[_tensor_to_pil(image) for image in unconditional_images[:16]],
        captions=[
            f"text={text}\nclf={label_to_string.get(int(pred), str(int(pred)))}"
            for text, pred in zip(unconditional_text_strings[:16], unconditional_clf[:16].tolist())
        ],
        cols=4,
    )

    image_to_text_path = run_context.log_path("visuals/image_to_text_grid.png")
    text_to_image_path = run_context.log_path("visuals/text_to_image_grid.png")
    unconditional_path = run_context.log_path("visuals/unconditional_grid.png")
    image_to_text_grid.save(image_to_text_path)
    text_to_image_grid.save(text_to_image_path)
    unconditional_grid.save(unconditional_path)

    summary = {
        "image_to_text_grid": str(image_to_text_path),
        "text_to_image_grid": str(text_to_image_path),
        "unconditional_grid": str(unconditional_path),
        "image_to_text_pairs": [
            {"gt": gt, "pred": pred} for gt, pred in zip(gt_text_strings[:16], sampled_text_strings[:16])
        ],
        "text_to_image_pairs": [
            {"condition": condition, "classifier_pred": label_to_string.get(int(pred), str(int(pred)))}
            for condition, pred in zip(text_metadata.label_strings[:10], generated_preds[:10].tolist())
        ],
        "unconditional_pairs": [
            {
                "text": text,
                "text_label_value": int(value),
                "classifier_pred": label_to_string.get(int(pred), str(int(pred))),
            }
            for text, value, pred in zip(
                unconditional_text_strings[:16],
                unconditional_text_values[:16].tolist(),
                unconditional_clf[:16].tolist(),
            )
        ],
    }
    summary_path = run_context.log_path("visuals/summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if own_context:
        run_context.update_status("ok")
    return {key: value for key, value in summary.items() if isinstance(value, str)}
