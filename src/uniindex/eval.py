from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import mask_logits, position_modalities, position_valid_token_mask, unified_vocab_size
from .model import UnifiedDenoiser
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, restore_image_tokens, sample_masked_noise
from .text import (
    decode_text_tokens,
    label_values_from_text_tokens,
    metadata_from_state,
    sequence_candidate_scores,
    shifted_label_text_tokens,
)
from .tokenizer import build_tokenizer
from .train import latest_checkpoint_path


def _load_stage2(config: ProjectConfig, device: torch.device) -> tuple[UnifiedDenoiser, dict, int]:
    stage2_path = latest_checkpoint_path(config, "stage2")
    if not stage2_path.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint at {stage2_path}")
    payload = torch.load(stage2_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    text_metadata = metadata_from_state(tokenizer_state)
    codebook_size = int(tokenizer_state["codebook_size"])
    image_seq_len = int(tokenizer_state["image_seq_len"])
    text_seq_len = int(text_metadata.seq_len)
    text_vocab_size = int(text_metadata.vocab_size)
    vocab_size = unified_vocab_size(codebook_size, text_vocab_size)
    model = UnifiedDenoiser(
        input_dim=vocab_size,
        seq_len=image_seq_len + text_seq_len,
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer_state, image_seq_len


def _decode_image_tokens(tokenizer, image_tokens: torch.Tensor, tokenizer_state: dict, grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    restored = restore_image_tokens(image_tokens.cpu(), tokenizer_state)
    return tokenizer.decode_token_batch(restored, grid_shape).to(device)


@torch.inference_mode()
def _sample_unified_with_logits(
    model: UnifiedDenoiser,
    codebook_size: int,
    text_vocab_size: int,
    image_seq_len: int,
    text_seq_len: int,
    schedule_tables: dict,
    temperature: float,
    steps: int,
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_text_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = batch_size or 1
    if condition_image_tokens is not None:
        batch = condition_image_tokens.shape[0]
    if condition_text_tokens is not None:
        batch = condition_text_tokens.shape[0]

    seq_len = image_seq_len + text_seq_len
    vocab_size = unified_vocab_size(codebook_size, text_vocab_size)
    modality_ids = position_modalities(image_seq_len, text_seq_len).to(device)
    valid_token_mask = position_valid_token_mask(image_seq_len, text_seq_len, codebook_size, text_vocab_size).to(device)
    z_t = sample_masked_noise(torch.zeros(batch, seq_len, vocab_size, device=device), valid_token_mask)

    if condition_image_tokens is not None:
        z_t[:, :image_seq_len] = build_flm_clean_state(condition_image_tokens.to(device), vocab_size)
    if condition_text_tokens is not None:
        text_targets = condition_text_tokens.to(device) + codebook_size
        z_t[:, image_seq_len:] = build_flm_clean_state(text_targets, vocab_size)

    effective_text_time_power = text_time_power
    if schedule_tables["kind"] == "power" and condition_image_tokens is not None and condition_text_tokens is None and image_to_text_text_time_power is not None:
        effective_text_time_power = image_to_text_text_time_power

    dt = 1.0 / max(steps, 1)
    for step in range(steps):
        progress = torch.full((batch,), step / max(steps, 1), device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        t_pos = condition_clean_timesteps(
            t_pos,
            image_seq_len,
            condition_image=condition_image_tokens is not None,
            condition_text=condition_text_tokens is not None,
        )
        logits = model(z_t, t_pos, modality_ids)
        logits = mask_logits(logits, image_seq_len, text_seq_len, codebook_size, text_vocab_size)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + dt * v_t
        if condition_image_tokens is not None:
            z_t[:, :image_seq_len] = build_flm_clean_state(condition_image_tokens.to(device), vocab_size)
        if condition_text_tokens is not None:
            z_t[:, image_seq_len:] = build_flm_clean_state(text_targets, vocab_size)

    final_t_pos = apply_schedule(
        progress=torch.ones(batch, device=device),
        modality_ids=modality_ids,
        schedule_tables=schedule_tables,
        image_time_power=image_time_power,
        text_time_power=effective_text_time_power,
    )
    final_logits = model(z_t, final_t_pos, modality_ids)
    final_logits = mask_logits(final_logits, image_seq_len, text_seq_len, codebook_size, text_vocab_size)
    return final_logits.argmax(dim=-1), final_logits


@torch.inference_mode()
def sample_unified(
    model: UnifiedDenoiser,
    codebook_size: int,
    text_vocab_size: int,
    image_seq_len: int,
    text_seq_len: int,
    schedule_tables: dict,
    temperature: float,
    steps: int,
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_text_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens, _ = _sample_unified_with_logits(
        model=model,
        codebook_size=codebook_size,
        text_vocab_size=text_vocab_size,
        image_seq_len=image_seq_len,
        text_seq_len=text_seq_len,
        schedule_tables=schedule_tables,
        temperature=temperature,
        steps=steps,
        image_time_power=image_time_power,
        text_time_power=text_time_power,
        image_to_text_text_time_power=image_to_text_text_time_power,
        batch_size=batch_size,
        condition_image_tokens=condition_image_tokens,
        condition_text_tokens=condition_text_tokens,
    )
    return tokens


def constrained_text_label_values(
    text_logits: torch.Tensor,
    text_metadata,
    *,
    codebook_size: int,
) -> torch.Tensor:
    candidate_tokens = shifted_label_text_tokens(text_metadata, token_offset=codebook_size).to(text_logits.device)
    scores = sequence_candidate_scores(text_logits, candidate_tokens)
    indices = scores.argmax(dim=1)
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=text_logits.device)
    return label_values.index_select(0, indices)


@torch.inference_mode()
def evaluate(
    config: ProjectConfig,
    run_context: RunContext | None = None,
    classifier_override_path: Path | None = None,
) -> dict[str, float]:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "eval")
    run_context.set_device(device)

    model, tokenizer_state, image_seq_len = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    codebook_size = int(tokenizer_state["codebook_size"])
    text_seq_len = int(text_metadata.seq_len)
    text_vocab_size = int(text_metadata.vocab_size)
    schedule_tables = build_schedule_tables(config, image_vocab_size=codebook_size, text_vocab_size=text_vocab_size)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier_source = classifier_override_path or classifier_path(config.paths.models_dir, config.dataset.name)
    classifier = load_classifier(classifier_source, config.dataset.name, device=device)
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    ceiling_correct = 0
    image_to_text_exact = 0
    image_to_text_token_correct = 0
    image_to_text_token_total = 0
    image_to_text_constrained_correct = 0
    text_to_image_correct = 0
    image_to_text_position_correct = torch.zeros(text_seq_len, dtype=torch.long)
    image_to_text_position_total = torch.zeros(text_seq_len, dtype=torch.long)
    generated_text_counter: Counter[str] = Counter()
    total = 0

    for batch in tqdm(test_loader, desc="eval"):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)

        decoded = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        ceiling_pred = classify_images(classifier, decoded, config.dataset.name)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        sampled_tokens, final_logits = _sample_unified_with_logits(
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
        )
        sampled_text = sampled_tokens[:, image_seq_len:] - codebook_size
        image_to_text_exact += sampled_text.eq(text_tokens).all(dim=1).sum().item()
        valid_text = text_tokens.ne(text_metadata.pad_id)
        image_to_text_token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
        image_to_text_token_total += valid_text.sum().item()
        image_to_text_position_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum(dim=0).cpu()
        image_to_text_position_total += valid_text.sum(dim=0).cpu()
        generated_text_counter.update(decode_text_tokens(sampled_text, text_metadata))
        constrained_values = constrained_text_label_values(
            final_logits[:, image_seq_len:],
            text_metadata,
            codebook_size=codebook_size,
        )
        image_to_text_constrained_correct += (constrained_values == labels).sum().item()

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
            condition_text_tokens=text_tokens,
        )[:, :image_seq_len]
        decoded_images = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
        image_pred = classify_images(classifier, decoded_images, config.dataset.name)
        text_to_image_correct += (image_pred == labels).sum().item()

        total += labels.numel()

    uncond_count = config.eval.num_unconditional_samples
    sampled = sample_unified(
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
        batch_size=uncond_count,
        condition_image_tokens=None,
        condition_text_tokens=None,
    )
    uncond_images = _decode_image_tokens(tokenizer, sampled[:, :image_seq_len], tokenizer_state, grid_shape, device)
    uncond_image_pred = classify_images(classifier, uncond_images, config.dataset.name)
    uncond_text_values = label_values_from_text_tokens(sampled[:, image_seq_len:] - codebook_size, text_metadata)
    consistency = (uncond_image_pred == uncond_text_values.to(device)).float().mean().item()

    metrics = {
        "tokenizer_ceiling": ceiling_correct / max(total, 1),
        "image_to_text_exact_match": image_to_text_exact / max(total, 1),
        "image_to_text_token_accuracy": image_to_text_token_correct / max(image_to_text_token_total, 1),
        "image_to_text_label_accuracy_constrained": image_to_text_constrained_correct / max(total, 1),
        "text_to_image_accuracy": text_to_image_correct / max(total, 1),
        "unconditional_consistency": consistency,
        "image_to_label_accuracy": image_to_text_exact / max(total, 1),
        "label_to_image_accuracy": text_to_image_correct / max(total, 1),
    }

    metrics_path = run_context.log_path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    diagnostics = {
        "image_to_text_position_accuracy": [
            correct / max(total_count, 1)
            for correct, total_count in zip(
                image_to_text_position_correct.tolist(),
                image_to_text_position_total.tolist(),
            )
        ],
        "image_to_text_generated_text_counts": dict(generated_text_counter.most_common(32)),
        "label_strings": list(text_metadata.label_strings),
    }
    diagnostics_path = run_context.log_path("diagnostics.json")
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    preview = {
        "generated_text_strings": decode_text_tokens(sampled[:, image_seq_len:] - codebook_size, text_metadata),
        "image_to_text_generated_text_counts": dict(generated_text_counter.most_common(32)),
        "label_strings": list(text_metadata.label_strings),
    }
    preview_path = run_context.log_path("text_preview.json")
    with preview_path.open("w", encoding="utf-8") as handle:
        json.dump(preview, handle, indent=2)

    if own_context:
        run_context.update_status("ok")
    return metrics
