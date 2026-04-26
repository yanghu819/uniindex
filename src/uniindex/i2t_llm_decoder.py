from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _apply_i2t_text_time_schedule
from .layout import TaskLayout
from .model import UnifiedDenoiser
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, sample_masked_noise
from .text import TextMetadata, metadata_from_state


@dataclass(frozen=True)
class LLMAssets:
    tokenizer: Any
    model: nn.Module
    embedding: nn.Module
    prompt_token_ids: torch.Tensor
    candidate_token_ids: torch.Tensor
    candidate_lengths: torch.Tensor
    label_values: torch.Tensor
    label_strings: tuple[str, ...]


class SoftPrefixAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, prefix_tokens: int, output_dim: int) -> None:
        super().__init__()
        self.prefix_tokens = int(prefix_tokens)
        self.output_dim = int(output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.prefix_tokens * self.output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        prefix = self.net(features)
        return prefix.reshape(features.shape[0], self.prefix_tokens, self.output_dim)


def freeze_module(module: nn.Module) -> dict[str, int]:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def _cache_dir(config: ProjectConfig) -> Path:
    raw = Path(config.i2t_llm.cache_dir)
    return raw if raw.is_absolute() else (config.repo_root / raw).resolve()


def _set_hf_cache_env(cache_dir: Path) -> None:
    hub_dir = cache_dir / "hub"
    transformers_dir = cache_dir / "transformers"
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(transformers_dir))


def _source_checkpoint_path(config: ProjectConfig) -> Path:
    raw_path = config.i2t_llm.source_checkpoint
    if raw_path is None:
        return config.paths.models_dir / "checkpoints" / "stage2_latest.pt"
    path = Path(raw_path)
    return path if path.is_absolute() else (config.repo_root / path).resolve()


def _build_denoiser(config: ProjectConfig, layout: TaskLayout, device: torch.device) -> UnifiedDenoiser:
    return UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
    ).to(device)


def _load_frozen_denoiser(
    config: ProjectConfig,
    device: torch.device,
) -> tuple[UnifiedDenoiser, dict, TaskLayout, TextMetadata]:
    checkpoint_path = _source_checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing i2t LLM source checkpoint at {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    text_metadata = metadata_from_state(tokenizer_state)
    layout = TaskLayout(
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(text_metadata.seq_len),
        codebook_size=int(tokenizer_state["codebook_size"]),
        text_vocab_size=int(text_metadata.vocab_size),
    )
    denoiser = _build_denoiser(config, layout, device)
    denoiser.load_state_dict(payload["model"])
    denoiser.eval()
    freeze_module(denoiser)
    return denoiser, tokenizer_state, layout, text_metadata


def _tokenize_candidate_labels(tokenizer: Any, text_metadata: TextMetadata, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("i2t LLM tokenizer must expose eos_token_id")
    encoded = []
    for label in text_metadata.label_strings:
        token_ids = tokenizer(f" {label}", add_special_tokens=False).input_ids
        token_ids = list(token_ids) + [int(eos_id)]
        encoded.append(token_ids)
    max_len = max(len(row) for row in encoded)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    padded = [row + [int(pad_id)] * (max_len - len(row)) for row in encoded]
    lengths = [len(row) for row in encoded]
    return (
        torch.tensor(padded, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def _load_llm_assets(config: ProjectConfig, text_metadata: TextMetadata, device: torch.device) -> LLMAssets:
    cache_dir = _cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_hf_cache_env(cache_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.i2t_llm.model_name, cache_dir=str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.i2t_llm.model_name, cache_dir=str(cache_dir)).to(device)
    model.eval()
    freeze_module(model)
    embedding = model.get_input_embeddings()
    prompt_ids = tokenizer(config.i2t_llm.prompt, add_special_tokens=False).input_ids
    if not prompt_ids:
        raise ValueError("i2t_llm.prompt must tokenize to at least one token")
    candidate_ids, candidate_lengths = _tokenize_candidate_labels(tokenizer, text_metadata, device)
    return LLMAssets(
        tokenizer=tokenizer,
        model=model,
        embedding=embedding,
        prompt_token_ids=torch.tensor(prompt_ids, dtype=torch.long, device=device),
        candidate_token_ids=candidate_ids,
        candidate_lengths=candidate_lengths,
        label_values=torch.tensor(text_metadata.label_values, dtype=torch.long, device=device),
        label_strings=tuple(str(label).lower() for label in text_metadata.label_strings),
    )


@torch.inference_mode()
def download_i2t_llm_assets(config: ProjectConfig) -> dict[str, str]:
    cache_dir = _cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_hf_cache_env(cache_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(config.i2t_llm.model_name, cache_dir=str(cache_dir))
    AutoModelForCausalLM.from_pretrained(config.i2t_llm.model_name, cache_dir=str(cache_dir))
    return {"model_name": config.i2t_llm.model_name, "cache_dir": str(cache_dir)}


@torch.no_grad()
def extract_i2t_image_features(
    *,
    denoiser: UnifiedDenoiser,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    image_tokens: torch.Tensor,
    base_state: torch.Tensor | None = None,
) -> torch.Tensor:
    device = image_tokens.device
    batch_size = image_tokens.shape[0]
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    if base_state is None:
        z_t = sample_masked_noise(
            torch.zeros(batch_size, layout.seq_len, layout.vocab_size, device=device),
            valid_token_mask,
        )
    else:
        z_t = base_state.to(device).clone()
    z_t[:, layout.image_slice] = build_flm_clean_state(image_tokens, layout.vocab_size)
    progress = torch.full((batch_size,), float(config.i2t_llm.feature_progress), device=device)
    effective_text_time_power = config.sampling.text_time_power
    if config.sampling.image_to_text_text_time_power is not None:
        effective_text_time_power = config.sampling.image_to_text_text_time_power
    t_pos = apply_schedule(
        progress=progress,
        modality_ids=modality_ids,
        schedule_tables=schedule_tables,
        image_time_power=config.sampling.image_time_power,
        text_time_power=effective_text_time_power,
    )
    t_pos = _apply_i2t_text_time_schedule(
        t_pos,
        progress,
        layout=layout,
        schedule=config.sampling.image_to_text_text_time_schedule,
        logit_normal_loc=config.sampling.image_to_text_logit_normal_loc,
        logit_normal_scale=config.sampling.image_to_text_logit_normal_scale,
    )
    t_pos = condition_clean_timesteps(
        t_pos,
        layout.image_seq_len,
        condition_image=True,
        condition_text=False,
    )
    features = denoiser.forward_features(z_t, t_pos, modality_ids)
    return features[:, layout.image_slice].mean(dim=1)


def _target_token_ids(labels: torch.Tensor, assets: LLMAssets) -> tuple[torch.Tensor, torch.Tensor]:
    value_to_row = {int(value): index for index, value in enumerate(assets.label_values.detach().cpu().tolist())}
    rows = [value_to_row[int(label)] for label in labels.detach().cpu().tolist()]
    row_ids = torch.tensor(rows, dtype=torch.long, device=labels.device)
    return (
        assets.candidate_token_ids.index_select(0, row_ids),
        assets.candidate_lengths.index_select(0, row_ids),
    )


def _sequence_inputs(
    *,
    prefix_embeds: torch.Tensor,
    embedding: nn.Module,
    prompt_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = prefix_embeds.shape[0]
    prompt_embeds = embedding(prompt_token_ids).unsqueeze(0).expand(batch, -1, -1)
    target_embeds = embedding(target_token_ids)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds, target_embeds], dim=1)

    labels = torch.full(
        inputs_embeds.shape[:2],
        -100,
        dtype=torch.long,
        device=prefix_embeds.device,
    )
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
    start = prefix_embeds.shape[1] + prompt_token_ids.shape[0]
    token_positions = torch.arange(target_token_ids.shape[1], device=prefix_embeds.device).unsqueeze(0)
    valid_targets = token_positions < target_lengths.unsqueeze(1)
    labels[:, start:] = torch.where(valid_targets, target_token_ids, torch.full_like(target_token_ids, -100))
    attention_mask[:, start:] = valid_targets.long()
    return inputs_embeds, attention_mask, labels


def llm_prefix_loss(prefix_embeds: torch.Tensor, labels: torch.Tensor, assets: LLMAssets) -> torch.Tensor:
    target_ids, target_lengths = _target_token_ids(labels, assets)
    inputs_embeds, attention_mask, target_labels = _sequence_inputs(
        prefix_embeds=prefix_embeds,
        embedding=assets.embedding,
        prompt_token_ids=assets.prompt_token_ids,
        target_token_ids=target_ids,
        target_lengths=target_lengths,
    )
    output = assets.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=target_labels)
    return output.loss


def score_candidate_token_ids(
    *,
    model: nn.Module,
    embedding: nn.Module,
    prefix_embeds: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_lengths: torch.Tensor,
) -> torch.Tensor:
    batch, prefix_len, hidden_dim = prefix_embeds.shape
    num_candidates, candidate_len = candidate_token_ids.shape
    flat_prefix = (
        prefix_embeds.unsqueeze(1)
        .expand(batch, num_candidates, prefix_len, hidden_dim)
        .reshape(batch * num_candidates, prefix_len, hidden_dim)
    )
    flat_candidates = (
        candidate_token_ids.unsqueeze(0)
        .expand(batch, num_candidates, candidate_len)
        .reshape(batch * num_candidates, candidate_len)
    )
    flat_lengths = candidate_lengths.unsqueeze(0).expand(batch, num_candidates).reshape(batch * num_candidates)
    inputs_embeds, attention_mask, _ = _sequence_inputs(
        prefix_embeds=flat_prefix,
        embedding=embedding,
        prompt_token_ids=prompt_token_ids,
        target_token_ids=flat_candidates,
        target_lengths=flat_lengths,
    )
    output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    start = prefix_len + prompt_token_ids.shape[0]
    shifted_logits = output.logits[:, start - 1 : start + candidate_len - 1]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=flat_candidates.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(candidate_len, device=prefix_embeds.device).unsqueeze(0)
    mask = positions < flat_lengths.unsqueeze(1)
    scores = gathered.masked_fill(~mask, 0.0).sum(dim=1)
    return scores.reshape(batch, num_candidates)


@torch.inference_mode()
def _greedy_generate(prefix_embeds: torch.Tensor, assets: LLMAssets, max_new_tokens: int) -> list[str]:
    batch = prefix_embeds.shape[0]
    prompt_embeds = assets.embedding(assets.prompt_token_ids).unsqueeze(0).expand(batch, -1, -1)
    inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
    generated = []
    for _ in range(max_new_tokens):
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=prefix_embeds.device)
        output = assets.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        next_tokens = output.logits[:, -1].argmax(dim=-1)
        generated.append(next_tokens)
        inputs_embeds = torch.cat([inputs_embeds, assets.embedding(next_tokens).unsqueeze(1)], dim=1)
    token_matrix = torch.stack(generated, dim=1)
    decoded = assets.tokenizer.batch_decode(token_matrix.detach().cpu().tolist(), skip_special_tokens=True)
    return [text.strip().lower() for text in decoded]


@torch.inference_mode()
def evaluate_i2t_llm_decoder(
    *,
    denoiser: UnifiedDenoiser,
    adapter: SoftPrefixAdapter,
    config: ProjectConfig,
    layout: TaskLayout,
    schedule_tables: dict,
    assets: LLMAssets,
) -> dict[str, float]:
    loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    totals = {
        "candidate_correct": 0,
        "shuffled_candidate_correct": 0,
        "free_generation_correct": 0,
        "total": 0,
    }
    label_to_text = dict(zip((int(value) for value in assets.label_values.detach().cpu().tolist()), assets.label_strings))
    metadata_values = assets.label_values
    for batch in tqdm(loader, desc="eval-i2t-llm"):
        image_tokens = batch["image_tokens"].to(metadata_values.device)
        labels = batch["label"].to(metadata_values.device)
        valid_token_mask = layout.position_valid_token_mask().to(image_tokens.device)
        base_state = sample_masked_noise(
            torch.zeros(image_tokens.shape[0], layout.seq_len, layout.vocab_size, device=image_tokens.device),
            valid_token_mask,
        )
        features = extract_i2t_image_features(
            denoiser=denoiser,
            config=config,
            layout=layout,
            schedule_tables=schedule_tables,
            image_tokens=image_tokens,
            base_state=base_state,
        )
        prefix = adapter(features)
        scores = score_candidate_token_ids(
            model=assets.model,
            embedding=assets.embedding,
            prefix_embeds=prefix,
            prompt_token_ids=assets.prompt_token_ids,
            candidate_token_ids=assets.candidate_token_ids,
            candidate_lengths=assets.candidate_lengths,
        )
        predicted = metadata_values.index_select(0, scores.argmax(dim=1))
        totals["candidate_correct"] += (predicted == labels).sum().item()

        shuffled_indices = torch.arange(image_tokens.shape[0], device=image_tokens.device).roll(1)
        shuffled_images = image_tokens.index_select(0, shuffled_indices)
        shuffled_features = extract_i2t_image_features(
            denoiser=denoiser,
            config=config,
            layout=layout,
            schedule_tables=schedule_tables,
            image_tokens=shuffled_images,
            base_state=base_state,
        )
        shuffled_prefix = adapter(shuffled_features)
        shuffled_scores = score_candidate_token_ids(
            model=assets.model,
            embedding=assets.embedding,
            prefix_embeds=shuffled_prefix,
            prompt_token_ids=assets.prompt_token_ids,
            candidate_token_ids=assets.candidate_token_ids,
            candidate_lengths=assets.candidate_lengths,
        )
        shuffled_predicted = metadata_values.index_select(0, shuffled_scores.argmax(dim=1))
        totals["shuffled_candidate_correct"] += (shuffled_predicted == labels).sum().item()

        generated = _greedy_generate(prefix, assets, config.i2t_llm.max_new_tokens)
        label_strings = [label_to_text[int(value)] for value in labels.detach().cpu().tolist()]
        totals["free_generation_correct"] += sum(pred == target for pred, target in zip(generated, label_strings))
        totals["total"] += labels.shape[0]

    total = max(totals["total"], 1)
    return {
        "candidate_accuracy": totals["candidate_correct"] / total,
        "shuffled_candidate_accuracy": totals["shuffled_candidate_correct"] / total,
        "image_dependence_margin": (totals["candidate_correct"] - totals["shuffled_candidate_correct"]) / total,
        "free_generation_exact": totals["free_generation_correct"] / total,
        "total": float(totals["total"]),
    }


def _infinite(loader):
    while True:
        yield from loader


def run_i2t_llm_decoder_probe(
    *,
    config: ProjectConfig,
    steps: int | None = None,
    eval_every: int = 50,
    run_context: RunContext | None = None,
) -> dict:
    if not config.i2t_llm.enabled:
        raise ValueError("probe-i2t-llm-decoder requires i2t_llm.enabled=true")
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    run_context = run_context or RunContext(config, "probe-i2t-llm-decoder")
    run_context.set_device(device)
    denoiser, _, layout, text_metadata = _load_frozen_denoiser(config, device)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    assets = _load_llm_assets(config, text_metadata, device)
    llm_hidden_dim = int(assets.embedding.weight.shape[1])
    adapter = SoftPrefixAdapter(
        input_dim=config.model.d_model,
        hidden_dim=config.i2t_llm.adapter_hidden_dim,
        prefix_tokens=config.i2t_llm.prefix_tokens,
        output_dim=llm_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.i2t_llm.lr)
    train_loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(train_loader)
    total_steps = int(config.i2t_llm.train_steps if steps is None else steps)
    if total_steps < 1:
        raise ValueError(f"steps must be >= 1, got {total_steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")

    train_log = run_context.log_path("i2t_llm_train.jsonl")
    eval_log = run_context.log_path("i2t_llm_eval.jsonl")
    eval_steps = {0, total_steps}
    eval_steps.update(range(eval_every, total_steps + 1, eval_every))
    snapshots = []
    summary = {
        "source": "probe-i2t-llm-decoder",
        "model_name": config.i2t_llm.model_name,
        "prefix_tokens": config.i2t_llm.prefix_tokens,
        "adapter_hidden_dim": config.i2t_llm.adapter_hidden_dim,
        "source_checkpoint": str(_source_checkpoint_path(config)),
        "steps": total_steps,
        "eval_every": eval_every,
        "snapshots": snapshots,
    }

    for step in range(total_steps + 1):
        if step in eval_steps:
            adapter.eval()
            metrics = evaluate_i2t_llm_decoder(
                denoiser=denoiser,
                adapter=adapter,
                config=config,
                layout=layout,
                schedule_tables=schedule_tables,
                assets=assets,
            )
            snapshot = {"step": step, **metrics}
            snapshots.append(snapshot)
            append_jsonl(eval_log, snapshot)
            adapter.train()
        if step == total_steps:
            break

        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        features = extract_i2t_image_features(
            denoiser=denoiser,
            config=config,
            layout=layout,
            schedule_tables=schedule_tables,
            image_tokens=image_tokens,
        )
        prefix = adapter(features)
        loss = llm_prefix_loss(prefix, labels, assets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()
        if (step + 1) % config.train.log_every == 0 or step == 0:
            append_jsonl(train_log, {"step": step + 1, "loss": float(loss.detach().cpu().item())})

    summary["final"] = snapshots[-1] if snapshots else {}
    summary_path = run_context.log_path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "summary": summary,
            "i2t_llm": dict(config.i2t_llm.__dict__),
        },
        run_context.log_path("adapter.pt"),
    )
    return summary
