from __future__ import annotations

import json

import torch
from tqdm import tqdm

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import mask_logits, position_modalities, unified_vocab_size
from .model import UnifiedDenoiser
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .state import apply_time_schedule, build_flm_clean_state, restore_image_tokens
from .tokenizer import build_tokenizer
from .train import latest_checkpoint_path


def _load_stage2(config: ProjectConfig, device: torch.device) -> tuple[UnifiedDenoiser, dict, int]:
    stage2_path = latest_checkpoint_path(config, "stage2")
    if not stage2_path.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint at {stage2_path}")
    payload = torch.load(stage2_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    codebook_size = int(tokenizer_state["codebook_size"])
    image_seq_len = int(tokenizer_state["image_seq_len"])
    num_labels = len(config.labels.values)
    vocab_size = unified_vocab_size(codebook_size, num_labels)
    model = UnifiedDenoiser(
        input_dim=vocab_size,
        seq_len=image_seq_len + 1,
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
def sample_unified(
    model: UnifiedDenoiser,
    codebook_size: int,
    num_labels: int,
    image_seq_len: int,
    temperature: float,
    steps: int,
    image_time_power: float,
    label_time_power: float,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    device = next(model.parameters()).device
    batch = batch_size or 1
    if condition_image_tokens is not None:
        batch = condition_image_tokens.shape[0]
    if condition_labels is not None:
        batch = condition_labels.shape[0]

    seq_len = image_seq_len + 1
    vocab_size = unified_vocab_size(codebook_size, num_labels)
    z_t = torch.randn(batch, seq_len, vocab_size, device=device)
    modality_ids = position_modalities(image_seq_len).to(device)

    if condition_image_tokens is not None:
        z_t[:, :image_seq_len] = build_flm_clean_state(condition_image_tokens.to(device), vocab_size)
    if condition_labels is not None:
        label_tokens = condition_labels.to(device).unsqueeze(1) + codebook_size
        z_t[:, image_seq_len:] = build_flm_clean_state(label_tokens, vocab_size)

    dt = 1.0 / max(steps, 1)
    for step in range(steps):
        progress = torch.full((batch,), step / max(steps, 1), device=device)
        t_pos = apply_time_schedule(progress, modality_ids, image_time_power, label_time_power)
        logits = model(z_t, t_pos, modality_ids)
        logits = mask_logits(logits, image_seq_len, codebook_size, num_labels)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + dt * v_t
        if condition_image_tokens is not None:
            z_t[:, :image_seq_len] = build_flm_clean_state(condition_image_tokens.to(device), vocab_size)
        if condition_labels is not None:
            z_t[:, image_seq_len:] = build_flm_clean_state(label_tokens, vocab_size)

    final_t_pos = apply_time_schedule(torch.ones(batch, device=device), modality_ids, image_time_power, label_time_power)
    final_logits = model(z_t, final_t_pos, modality_ids)
    final_logits = mask_logits(final_logits, image_seq_len, codebook_size, num_labels)
    return final_logits.argmax(dim=-1)


@torch.inference_mode()
def evaluate(config: ProjectConfig, run_context: RunContext | None = None) -> dict[str, float]:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "eval")
    run_context.set_device(device)

    model, tokenizer_state, image_seq_len = _load_stage2(config, device)
    codebook_size = int(tokenizer_state["codebook_size"])
    num_labels = len(config.labels.values)
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(classifier_path(config.paths.models_dir), device=device)
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    ceiling_correct = 0
    label_correct = 0
    image_correct = 0
    total = 0

    for batch in tqdm(test_loader, desc="eval"):
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        decoded = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        ceiling_pred = classify_images(classifier, decoded)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        sampled_labels = sample_unified(
            model=model,
            codebook_size=codebook_size,
            num_labels=num_labels,
            image_seq_len=image_seq_len,
            temperature=config.sampling.temperature,
            steps=config.sampling.steps,
            image_time_power=config.sampling.image_time_power,
            label_time_power=config.sampling.label_time_power,
            condition_image_tokens=image_tokens,
            condition_labels=None,
        )[:, image_seq_len]
        label_correct += ((sampled_labels - codebook_size) == labels).sum().item()

        sampled_images = sample_unified(
            model=model,
            codebook_size=codebook_size,
            num_labels=num_labels,
            image_seq_len=image_seq_len,
            temperature=config.sampling.temperature,
            steps=config.sampling.steps,
            image_time_power=config.sampling.image_time_power,
            label_time_power=config.sampling.label_time_power,
            condition_image_tokens=None,
            condition_labels=labels,
        )[:, :image_seq_len]
        decoded_images = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
        image_pred = classify_images(classifier, decoded_images)
        image_correct += (image_pred == labels).sum().item()

        total += labels.numel()

    uncond_count = config.eval.num_unconditional_samples
    sampled = sample_unified(
        model=model,
        codebook_size=codebook_size,
        num_labels=num_labels,
        image_seq_len=image_seq_len,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        image_time_power=config.sampling.image_time_power,
        label_time_power=config.sampling.label_time_power,
        batch_size=uncond_count,
        condition_image_tokens=None,
        condition_labels=None,
    )
    uncond_images = _decode_image_tokens(tokenizer, sampled[:, :image_seq_len], tokenizer_state, grid_shape, device)
    uncond_image_pred = classify_images(classifier, uncond_images)
    uncond_labels = sampled[:, image_seq_len] - codebook_size
    consistency = (uncond_image_pred == uncond_labels).float().mean().item()

    metrics = {
        "tokenizer_ceiling": ceiling_correct / max(total, 1),
        "image_to_label_accuracy": label_correct / max(total, 1),
        "label_to_image_accuracy": image_correct / max(total, 1),
        "unconditional_consistency": consistency,
    }

    metrics_path = run_context.log_path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if own_context:
        run_context.update_status("ok")
    return metrics
