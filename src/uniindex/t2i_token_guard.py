from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _decode_image_tokens, _load_stage2, sample_unified
from .label_feature_probe import VQTokenLabelProbe, labels_to_class_indices
from .layout import TaskLayout
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .text import label_values_from_text_tokens, metadata_from_state
from .tokenizer import build_tokenizer


def confusion_matrix(predicted: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError(f"predicted and target must have the same shape, got {predicted.shape} and {target.shape}")
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    pred = predicted.detach().cpu().long().reshape(-1)
    tgt = target.detach().cpu().long().reshape(-1)
    valid = tgt.ge(0).logical_and(tgt.lt(num_classes)).logical_and(pred.ge(0)).logical_and(pred.lt(num_classes))
    flat = tgt[valid] * num_classes + pred[valid]
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _per_class_accuracy_from_confusion(matrix: torch.Tensor) -> list[float | None]:
    values: list[float | None] = []
    for index in range(matrix.shape[0]):
        total = int(matrix[index].sum().item())
        values.append(None if total == 0 else float(matrix[index, index].item() / total))
    return values


def _infinite(loader):
    while True:
        yield from loader


def _jsonable_counter(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _labels_for_class_indices(class_indices: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    return label_values.to(class_indices.device).index_select(0, class_indices.long())


def _label_value_to_class_indices(values: torch.Tensor, label_values: torch.Tensor) -> torch.Tensor:
    values_cpu = values.detach().cpu().long()
    mapping = {int(value): index for index, value in enumerate(label_values.detach().cpu().long().tolist())}
    indices = [mapping.get(int(value), -1) for value in values_cpu.tolist()]
    return torch.tensor(indices, dtype=torch.long, device=values.device)


@torch.no_grad()
def _evaluate_probe_on_split(
    *,
    config: ProjectConfig,
    probe: VQTokenLabelProbe,
    label_values: torch.Tensor,
    split: str,
) -> dict[str, float]:
    device = label_values.device
    loader = build_loader(
        split_path(config, split),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    correct = 0
    total = 0
    for batch in tqdm(loader, desc=f"t2i-token-guard/{split}-real"):
        image_tokens = batch["image_tokens"].to(device)
        targets = labels_to_class_indices(batch["label"].to(device), label_values)
        pred = probe(image_tokens).argmax(dim=1)
        correct += int(pred.eq(targets).sum().item())
        total += int(targets.numel())
    return {f"{split}_real_token_label_accuracy": correct / max(total, 1), f"{split}_real_total": float(total)}


def _train_probe(
    *,
    config: ProjectConfig,
    layout: TaskLayout,
    label_values: torch.Tensor,
    steps: int,
    eval_every: int,
    lr: float,
    run_context: RunContext,
) -> tuple[VQTokenLabelProbe, list[dict]]:
    if steps < 1:
        raise ValueError(f"probe_steps must be >= 1, got {steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")

    device = label_values.device
    probe = VQTokenLabelProbe(
        codebook_size=layout.codebook_size,
        seq_len=layout.image_seq_len,
        num_classes=int(label_values.numel()),
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=config.train.weight_decay)
    train_loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(train_loader)
    train_log = run_context.log_path("token_guard_probe_train.jsonl")
    eval_log = run_context.log_path("token_guard_probe_eval.jsonl")
    eval_steps = {0, steps}
    eval_steps.update(range(eval_every, steps + 1, eval_every))
    snapshots: list[dict] = []

    for step in range(steps + 1):
        if step in eval_steps:
            probe.eval()
            metrics = {
                "step": int(step),
                **_evaluate_probe_on_split(config=config, probe=probe, label_values=label_values, split="test"),
            }
            snapshots.append(metrics)
            append_jsonl(eval_log, metrics)
            probe.train()
        if step == steps:
            break
        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        targets = labels_to_class_indices(batch["label"].to(device), label_values)
        logits = probe(image_tokens)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(probe.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()
        if step == 0 or (step + 1) % config.train.log_every == 0:
            append_jsonl(
                train_log,
                {
                    "step": int(step + 1),
                    "loss": float(loss.detach().cpu().item()),
                    "batch_accuracy": float(logits.argmax(dim=1).eq(targets).float().mean().detach().cpu().item()),
                },
            )

    return probe, snapshots


def _sample_t2i_tokens(
    *,
    model,
    layout: TaskLayout,
    schedule_tables: dict,
    config: ProjectConfig,
    text_metadata,
    condition_text_tokens: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, condition_text_tokens.shape[0], batch_size), desc="t2i-token-guard/sample-conditioned"):
        stop = min(start + batch_size, condition_text_tokens.shape[0])
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
            condition_image_tokens=None,
            condition_text_tokens=condition_text_tokens[start:stop],
        )[:, layout.image_slice]
        chunks.append(sampled.detach().cpu())
    return torch.cat(chunks, dim=0)


def _sample_unconditional(
    *,
    model,
    layout: TaskLayout,
    schedule_tables: dict,
    config: ProjectConfig,
    text_metadata,
    count: int,
    batch_size: int,
) -> torch.Tensor:
    chunks = []
    for start in tqdm(range(0, count, batch_size), desc="t2i-token-guard/sample-unconditional"):
        current = min(batch_size, count - start)
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
            batch_size=current,
            condition_image_tokens=None,
            condition_text_tokens=None,
        )
        chunks.append(sampled.detach().cpu())
    return torch.cat(chunks, dim=0)


def _normalized_token_histogram(tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    counts = torch.bincount(tokens.detach().cpu().long().reshape(-1), minlength=codebook_size).float()
    return counts / counts.sum().clamp_min(1.0)


def _real_test_token_histogram(config: ProjectConfig, codebook_size: int) -> torch.Tensor:
    loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    counts = torch.zeros(codebook_size, dtype=torch.float)
    for batch in tqdm(loader, desc="t2i-token-guard/test-token-hist"):
        counts += torch.bincount(batch["image_tokens"].long().reshape(-1), minlength=codebook_size).float()
    return counts / counts.sum().clamp_min(1.0)


def _save_confusion_heatmap(matrix: torch.Tensor, labels: list[str], path: Path, title: str) -> None:
    cell = 34
    margin_left = 58
    margin_top = 42
    width = margin_left + cell * len(labels) + 8
    height = margin_top + cell * len(labels) + 34
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(0, 0, 0), font=font)
    max_value = float(matrix.max().item()) if matrix.numel() else 0.0
    for i, actual in enumerate(labels):
        draw.text((8, margin_top + i * cell + 10), actual, fill=(0, 0, 0), font=font)
    for j, predicted in enumerate(labels):
        draw.text((margin_left + j * cell + 9, 24), predicted, fill=(0, 0, 0), font=font)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(matrix[i, j].item())
            intensity = 0 if max_value <= 0 else int(round(220 * value / max_value))
            color = (255 - intensity, 255 - intensity, 255)
            x = margin_left + j * cell
            y = margin_top + i * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color, outline=(210, 210, 210))
            draw.text((x + 8, y + 10), str(value), fill=(0, 0, 0), font=font)
    draw.text((8, height - 22), "rows=condition / cols=token-classifier prediction", fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _tensor_to_pil(image: torch.Tensor, size: int = 112) -> Image.Image:
    image = image.detach().cpu().clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    pil = Image.fromarray(array)
    if pil.size != (size, size):
        pil = pil.resize((size, size), Image.BICUBIC)
    return pil


def _save_pixel_grid(
    *,
    images: torch.Tensor,
    captions: list[str],
    path: Path,
    cols: int = 5,
    cell_size: int = 112,
    caption_height: int = 44,
) -> None:
    rows = max((len(captions) + cols - 1) // cols, 1)
    canvas = Image.new("RGB", (cols * cell_size, rows * (cell_size + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, caption in enumerate(captions):
        row = index // cols
        col = index % cols
        x = col * cell_size
        y = row * (cell_size + caption_height)
        canvas.paste(_tensor_to_pil(images[index], size=cell_size), (x, y))
        draw.rectangle((x, y + cell_size, x + cell_size, y + cell_size + caption_height), fill=(248, 248, 248))
        draw.multiline_text((x + 4, y + cell_size + 4), caption, fill=(0, 0, 0), font=font, spacing=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _maybe_decode_pixel_grid(
    *,
    config: ProjectConfig,
    tokenizer_state: dict,
    generated_tokens: torch.Tensor,
    condition_class_indices: torch.Tensor,
    token_pred_indices: torch.Tensor,
    label_values: torch.Tensor,
    decode_samples: int,
    device: torch.device,
    run_context: RunContext,
) -> dict[str, str | None]:
    if decode_samples <= 0:
        return {"conditioned_pixel_grid": None}
    count = min(int(decode_samples), int(generated_tokens.shape[0]))
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(classifier_path(config.paths.models_dir, config.dataset.name), config.dataset.name, device=device)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    selected = generated_tokens[:count].to(device)
    decoded = _decode_image_tokens(tokenizer, selected, tokenizer_state, grid_shape, device)
    pixel_pred = classify_images(classifier, decoded, config.dataset.name).detach().cpu()
    label_values_cpu = label_values.detach().cpu()
    captions = []
    for index in range(count):
        cond_value = int(label_values_cpu[int(condition_class_indices[index].item())].item())
        token_value = int(label_values_cpu[int(token_pred_indices[index].item())].item())
        captions.append(f"cond={cond_value}\ntoken={token_value} pix={int(pixel_pred[index].item())}")
    path = run_context.log_path("visuals/conditioned_pixel_grid.png")
    _save_pixel_grid(images=decoded.detach().cpu(), captions=captions, path=path)
    return {"conditioned_pixel_grid": str(path)}


def run_t2i_token_guard(
    *,
    config: ProjectConfig,
    probe_steps: int = 200,
    eval_every: int = 50,
    samples_per_label: int = 8,
    unconditional_count: int = 32,
    decode_samples: int = 0,
    lr: float | None = None,
    run_context: RunContext | None = None,
) -> dict:
    if samples_per_label < 1:
        raise ValueError(f"samples_per_label must be >= 1, got {samples_per_label}")
    if unconditional_count < 0:
        raise ValueError(f"unconditional_count must be >= 0, got {unconditional_count}")
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    run_context = run_context or RunContext(config, "probe-t2i-token-guard")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=device)
    probe, snapshots = _train_probe(
        config=config,
        layout=layout,
        label_values=label_values,
        steps=probe_steps,
        eval_every=eval_every,
        lr=float(config.i2t_llm.lr if lr is None else lr),
        run_context=run_context,
    )
    probe.eval()

    sampling_seed = config.eval.sampling_seed if config.eval.sampling_seed is not None else config.train.seed
    torch.manual_seed(int(sampling_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(sampling_seed))

    num_classes = int(label_values.numel())
    condition_class_indices = torch.arange(num_classes, device=device).repeat_interleave(samples_per_label)
    condition_text_tokens = text_metadata.label_text_tokens.to(device).index_select(0, condition_class_indices)
    generated_tokens = _sample_t2i_tokens(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        config=config,
        text_metadata=text_metadata,
        condition_text_tokens=condition_text_tokens,
        batch_size=config.train.eval_batch_size,
    )
    with torch.no_grad():
        generated_logits = probe(generated_tokens.to(device))
        generated_pred_indices = generated_logits.argmax(dim=1).detach().cpu()
    condition_indices_cpu = condition_class_indices.detach().cpu()
    conditioned_confusion = confusion_matrix(
        generated_pred_indices,
        condition_indices_cpu,
        num_classes=num_classes,
    )
    conditioned_accuracy = float(generated_pred_indices.eq(condition_indices_cpu).float().mean().item())

    uncond_metrics: dict[str, float | int | dict[str, int]] = {}
    uncond_records: list[dict] = []
    if unconditional_count > 0:
        unconditional = _sample_unconditional(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            config=config,
            text_metadata=text_metadata,
            count=unconditional_count,
            batch_size=config.train.eval_batch_size,
        )
        uncond_image_tokens = unconditional[:, layout.image_slice]
        uncond_text_tokens = unconditional[:, layout.text_slice] - layout.text_offset
        with torch.no_grad():
            uncond_pred_indices = probe(uncond_image_tokens.to(device)).argmax(dim=1).detach().cpu()
        uncond_text_values = label_values_from_text_tokens(uncond_text_tokens, text_metadata).detach().cpu()
        uncond_pred_values = _labels_for_class_indices(uncond_pred_indices, label_values.detach().cpu())
        valid_text = uncond_text_values.ne(-1)
        consistency = uncond_pred_values.eq(uncond_text_values).float().mean().item()
        valid_consistency = (
            uncond_pred_values[valid_text].eq(uncond_text_values[valid_text]).float().mean().item()
            if bool(valid_text.any().item())
            else 0.0
        )
        uncond_metrics = {
            "unconditional_token_text_consistency": float(consistency),
            "unconditional_valid_text_consistency": float(valid_consistency),
            "unconditional_valid_text_fraction": float(valid_text.float().mean().item()),
            "unconditional_token_pred_histogram": _jsonable_counter(Counter(uncond_pred_values.tolist())),
            "unconditional_text_value_histogram": _jsonable_counter(Counter(uncond_text_values.tolist())),
        }
        for index in range(unconditional_count):
            uncond_records.append(
                {
                    "index": index,
                    "token_pred_label": int(uncond_pred_values[index].item()),
                    "text_label": int(uncond_text_values[index].item()),
                }
            )

    real_hist = _real_test_token_histogram(config, layout.codebook_size)
    generated_hist = _normalized_token_histogram(generated_tokens, layout.codebook_size)
    token_hist_l1 = float((real_hist - generated_hist).abs().sum().item())
    avg_unique_per_sample = float(torch.tensor([row.unique().numel() for row in generated_tokens]).float().mean().item())
    total_unique = int(generated_tokens.unique().numel())

    labels = [str(int(value)) for value in label_values.detach().cpu().tolist()]
    confusion_path = run_context.log_path("visuals/conditioned_token_confusion.png")
    _save_confusion_heatmap(conditioned_confusion, labels, confusion_path, "Text -> generated VQ token classifier")

    visual_paths = {
        "conditioned_token_confusion": str(confusion_path),
        **_maybe_decode_pixel_grid(
            config=config,
            tokenizer_state=tokenizer_state,
            generated_tokens=generated_tokens,
            condition_class_indices=condition_indices_cpu,
            token_pred_indices=generated_pred_indices,
            label_values=label_values.detach().cpu(),
            decode_samples=decode_samples,
            device=device,
            run_context=run_context,
        ),
    }

    label_values_cpu = label_values.detach().cpu()
    conditioned_records = [
        {
            "index": index,
            "condition_label": int(label_values_cpu[int(condition_indices_cpu[index].item())].item()),
            "token_pred_label": int(label_values_cpu[int(generated_pred_indices[index].item())].item()),
        }
        for index in range(generated_tokens.shape[0])
    ]
    records_path = run_context.log_path("token_guard_records.json")
    records_path.write_text(
        json.dumps(
            {
                "conditioned": conditioned_records,
                "unconditional": uncond_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {
        "source": "probe-t2i-token-guard",
        "probe_steps": int(probe_steps),
        "eval_every": int(eval_every),
        "samples_per_label": int(samples_per_label),
        "unconditional_count": int(unconditional_count),
        "decode_samples": int(decode_samples),
        "sampling_seed": int(sampling_seed),
        "probe_snapshots": snapshots,
        "best_real_probe": max(snapshots, key=lambda row: row["test_real_token_label_accuracy"]) if snapshots else {},
        "final_real_probe": snapshots[-1] if snapshots else {},
        "conditioned_token_label_accuracy": conditioned_accuracy,
        "conditioned_total": int(generated_tokens.shape[0]),
        "conditioned_confusion": conditioned_confusion.tolist(),
        "conditioned_per_label_accuracy": _per_class_accuracy_from_confusion(conditioned_confusion),
        "conditioned_token_pred_histogram": _jsonable_counter(
            Counter(_labels_for_class_indices(generated_pred_indices, label_values.detach().cpu()).tolist())
        ),
        "generated_unique_token_count": total_unique,
        "generated_avg_unique_tokens_per_sample": avg_unique_per_sample,
        "generated_vs_real_test_token_histogram_l1": token_hist_l1,
        **uncond_metrics,
        "visuals": visual_paths,
        "records": str(records_path),
    }
    summary_path = run_context.log_path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
