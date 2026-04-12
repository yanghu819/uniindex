from __future__ import annotations

import json

import torch
import torch.nn as nn
from tqdm import tqdm

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .layout import mask_logits, position_modalities, unified_vocab_size
from .model import UnifiedDenoiser
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .tokenizer import build_tokenizer
from .train import latest_checkpoint_path


def _load_stage2(config: ProjectConfig, device: torch.device) -> tuple[UnifiedDenoiser, nn.Embedding, dict, int]:
    stage2_path = latest_checkpoint_path(config, "stage2")
    if not stage2_path.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint at {stage2_path}")
    payload = torch.load(stage2_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    codebook_size = int(tokenizer_state["codebook_size"])
    embed_dim = int(tokenizer_state["embed_dim"])
    image_seq_len = int(tokenizer_state["image_seq_len"])
    num_labels = len(config.labels.values)
    model = UnifiedDenoiser(
        input_dim=embed_dim,
        seq_len=image_seq_len + 1,
        vocab_size=unified_vocab_size(codebook_size, num_labels),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)
    label_embed = nn.Embedding(num_labels, embed_dim).to(device)
    model.load_state_dict(payload["model"])
    label_embed.load_state_dict(payload["label_embed"])
    model.eval()
    return model, label_embed, tokenizer_state, image_seq_len


def _all_embeddings(codebook: torch.Tensor, label_embed: nn.Embedding) -> torch.Tensor:
    return torch.cat([codebook, label_embed.weight], dim=0)


@torch.inference_mode()
def sample_unified(
    model: UnifiedDenoiser,
    codebook: torch.Tensor,
    label_embed: nn.Embedding,
    image_seq_len: int,
    temperature: float,
    steps: int,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_labels: torch.Tensor | None = None,
) -> torch.Tensor:
    device = codebook.device
    batch = batch_size or 1
    if condition_image_tokens is not None:
        batch = condition_image_tokens.shape[0]
    if condition_labels is not None:
        batch = condition_labels.shape[0]

    seq_len = image_seq_len + 1
    z_t = torch.randn(batch, seq_len, codebook.shape[1], device=device)
    modality_ids = position_modalities(image_seq_len).to(device)
    embeddings = _all_embeddings(codebook, label_embed)
    codebook_size = codebook.shape[0]
    num_labels = label_embed.num_embeddings

    if condition_image_tokens is not None:
        z_t[:, :image_seq_len] = codebook[condition_image_tokens.to(device)]
    if condition_labels is not None:
        z_t[:, image_seq_len:] = label_embed(condition_labels.to(device)).unsqueeze(1)

    dt = 1.0 / max(steps, 1)
    for step in range(steps):
        t_scalar = step / max(steps, 1)
        t = torch.full((batch,), t_scalar, device=device)
        logits = model(z_t, t, modality_ids)
        logits = mask_logits(logits, image_seq_len, codebook_size, num_labels)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        mu = probs @ embeddings
        v_t = (mu - z_t) / max(1.0 - t_scalar, 1e-4)
        z_t = z_t + dt * v_t
        if condition_image_tokens is not None:
            z_t[:, :image_seq_len] = codebook[condition_image_tokens.to(device)]
        if condition_labels is not None:
            z_t[:, image_seq_len:] = label_embed(condition_labels.to(device)).unsqueeze(1)

    final_logits = model(z_t, torch.ones(batch, device=device), modality_ids)
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

    model, label_embed, tokenizer_state, image_seq_len = _load_stage2(config, device)
    codebook = tokenizer_state["codebook"].float().to(device)
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
        decoded = tokenizer.decode_token_batch(image_tokens.cpu(), grid_shape).to(device)
        ceiling_pred = classify_images(classifier, decoded)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        sampled_labels = sample_unified(
            model=model,
            codebook=codebook,
            label_embed=label_embed,
            image_seq_len=image_seq_len,
            temperature=config.sampling.temperature,
            steps=config.sampling.steps,
            condition_image_tokens=image_tokens,
            condition_labels=None,
        )[:, image_seq_len]
        label_correct += ((sampled_labels - codebook.shape[0]) == labels).sum().item()

        sampled_images = sample_unified(
            model=model,
            codebook=codebook,
            label_embed=label_embed,
            image_seq_len=image_seq_len,
            temperature=config.sampling.temperature,
            steps=config.sampling.steps,
            condition_image_tokens=None,
            condition_labels=labels,
        )[:, :image_seq_len]
        decoded_images = tokenizer.decode_token_batch(sampled_images.cpu(), grid_shape).to(device)
        image_pred = classify_images(classifier, decoded_images)
        image_correct += (image_pred == labels).sum().item()

        total += labels.numel()

    uncond_count = config.eval.num_unconditional_samples
    sampled = sample_unified(
        model=model,
        codebook=codebook,
        label_embed=label_embed,
        image_seq_len=image_seq_len,
        temperature=config.sampling.temperature,
        steps=config.sampling.steps,
        batch_size=uncond_count,
        condition_image_tokens=None,
        condition_labels=None,
    )
    uncond_images = tokenizer.decode_token_batch(sampled[:, :image_seq_len].cpu(), grid_shape).to(device)
    uncond_image_pred = classify_images(classifier, uncond_images)
    uncond_labels = sampled[:, image_seq_len] - codebook.shape[0]
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
