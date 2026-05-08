from __future__ import annotations

from contextlib import contextmanager
from collections import Counter
import json
from pathlib import Path
from typing import Iterator

import torch
from tqdm import tqdm

from .classifier import classify_images, classifier_path, load_classifier
from .config import ProjectConfig
from .data import build_loader, split_path
from .layout import TaskLayout, mask_logits
from .model import UnifiedDenoiser
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import apply_schedule, build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, restore_image_tokens, sample_masked_noise
from .text import decode_text_tokens, label_values_from_text_tokens, metadata_from_state, text_scoring_mask
from .tokenizer import build_tokenizer
from .train import latest_checkpoint_path


_EVAL_RNG_BRANCH_OFFSETS = {"text_to_image": 20_000, "unconditional": 30_000}
_SQRT_2 = 2.0**0.5


def _load_stage2(config: ProjectConfig, device: torch.device) -> tuple[UnifiedDenoiser, dict, TaskLayout]:
    stage2_path = latest_checkpoint_path(config, "stage2")
    if not stage2_path.exists():
        raise FileNotFoundError(f"missing stage2 checkpoint at {stage2_path}")
    payload = torch.load(stage2_path, map_location=device)
    tokenizer_state = payload["tokenizer_state"]
    text_metadata = metadata_from_state(tokenizer_state)
    layout = TaskLayout(
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(text_metadata.seq_len),
        codebook_size=int(tokenizer_state["codebook_size"]),
        text_vocab_size=int(text_metadata.vocab_size),
    )
    model = UnifiedDenoiser(
        input_dim=layout.vocab_size,
        seq_len=layout.seq_len,
        vocab_size=layout.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        position_encoding=config.model.position_encoding,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer_state, layout


def _decode_image_tokens(tokenizer, image_tokens: torch.Tensor, tokenizer_state: dict, grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    restored = restore_image_tokens(image_tokens.cpu(), tokenizer_state)
    return tokenizer.decode_token_batch(restored, grid_shape).to(device)


class _EvalSamplingRngStreams:
    def __init__(self, *, enabled: bool, base_seed: int, device: torch.device) -> None:
        self.enabled = enabled
        self.base_seed = int(base_seed)
        self.cuda_devices = []
        if device.type == "cuda":
            self.cuda_devices.append(device.index if device.index is not None else torch.cuda.current_device())
        self._states: dict[str, tuple[torch.Tensor, list[torch.Tensor]]] = {}
        if enabled:
            self._states["image_to_text"] = self._current_state()
            for branch, offset in _EVAL_RNG_BRANCH_OFFSETS.items():
                self._states[branch] = self._seeded_state(self.base_seed + offset)

    def _current_state(self) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return (
            torch.random.get_rng_state(),
            [torch.cuda.get_rng_state(device) for device in self.cuda_devices],
        )

    def _set_state(self, state: tuple[torch.Tensor, list[torch.Tensor]]) -> None:
        cpu_state, cuda_states = state
        torch.random.set_rng_state(cpu_state)
        for device, cuda_state in zip(self.cuda_devices, cuda_states):
            torch.cuda.set_rng_state(cuda_state, device)

    def _seeded_state(self, seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.manual_seed(seed)
            if self.cuda_devices:
                torch.cuda.manual_seed_all(seed)
            return self._current_state()

    @contextmanager
    def branch(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if name not in self._states:
            raise ValueError(f"unsupported eval RNG branch: {name}")
        outer_state = self._current_state()
        self._set_state(self._states[name])
        try:
            yield
        finally:
            self._states[name] = self._current_state()
            self._set_state(outer_state)


def _logit_normal_gamma_from_progress(progress: torch.Tensor, *, loc: float, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError(f"logit-normal scale must be > 0, got {scale}")
    clipped = progress.clamp(1e-6, 1.0 - 1e-6)
    normal_quantile = _SQRT_2 * torch.erfinv(2.0 * clipped - 1.0)
    gamma = torch.sigmoid(float(loc) + float(scale) * normal_quantile)
    gamma = torch.where(progress <= 0.0, torch.zeros_like(gamma), gamma)
    return torch.where(progress >= 1.0, torch.ones_like(gamma), gamma)


def _apply_i2t_text_time_schedule(
    t_pos: torch.Tensor,
    progress: torch.Tensor,
    *,
    layout: TaskLayout,
    schedule: str,
    logit_normal_loc: float,
    logit_normal_scale: float,
) -> torch.Tensor:
    if schedule == "power":
        return t_pos
    if schedule == "logit_normal":
        updated = t_pos.clone()
        gamma = _logit_normal_gamma_from_progress(progress, loc=logit_normal_loc, scale=logit_normal_scale)
        updated[:, layout.text_slice] = gamma[:, None]
        return updated
    raise ValueError(f"unsupported image_to_text_text_time_schedule: {schedule}")


@torch.inference_mode()
def _sample_unified_with_logits(
    model: UnifiedDenoiser,
    layout: TaskLayout,
    schedule_tables: dict,
    temperature: float,
    steps: int,
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    image_to_text_text_time_schedule: str = "power",
    image_to_text_logit_normal_loc: float = 0.0,
    image_to_text_logit_normal_scale: float = 1.0,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_text_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 1:
        raise ValueError(f"sampling steps must be >= 1, got {steps}")
    if image_to_text_text_time_schedule not in {"power", "logit_normal"}:
        raise ValueError(f"unsupported image_to_text_text_time_schedule: {image_to_text_text_time_schedule}")
    if image_to_text_logit_normal_scale <= 0.0:
        raise ValueError(f"image_to_text_logit_normal_scale must be > 0, got {image_to_text_logit_normal_scale}")

    device = next(model.parameters()).device
    batch = batch_size or 1
    if condition_image_tokens is not None:
        batch = condition_image_tokens.shape[0]
    if condition_text_tokens is not None:
        batch = condition_text_tokens.shape[0]

    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    z_t = sample_masked_noise(torch.zeros(batch, layout.seq_len, layout.vocab_size, device=device), valid_token_mask)

    text_targets = None
    if condition_image_tokens is not None:
        z_t[:, layout.image_slice] = build_flm_clean_state(condition_image_tokens.to(device), layout.vocab_size)
    if condition_text_tokens is not None:
        text_targets = condition_text_tokens.to(device) + layout.text_offset
        z_t[:, layout.text_slice] = build_flm_clean_state(text_targets, layout.vocab_size)

    effective_text_time_power = text_time_power
    is_i2t = condition_image_tokens is not None and condition_text_tokens is None
    if is_i2t and image_to_text_text_time_power is not None:
        effective_text_time_power = image_to_text_text_time_power

    last_logits = None
    for step in range(steps):
        progress = torch.full((batch,), step / steps, device=device)
        next_progress = torch.full((batch,), (step + 1) / steps, device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        next_t_pos = apply_schedule(
            progress=next_progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        if is_i2t:
            t_pos = _apply_i2t_text_time_schedule(
                t_pos,
                progress,
                layout=layout,
                schedule=image_to_text_text_time_schedule,
                logit_normal_loc=image_to_text_logit_normal_loc,
                logit_normal_scale=image_to_text_logit_normal_scale,
            )
            next_t_pos = _apply_i2t_text_time_schedule(
                next_t_pos,
                next_progress,
                layout=layout,
                schedule=image_to_text_text_time_schedule,
                logit_normal_loc=image_to_text_logit_normal_loc,
                logit_normal_scale=image_to_text_logit_normal_scale,
            )
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=condition_image_tokens is not None,
            condition_text=condition_text_tokens is not None,
        )
        next_t_pos = condition_clean_timesteps(
            next_t_pos,
            layout.image_seq_len,
            condition_image=condition_image_tokens is not None,
            condition_text=condition_text_tokens is not None,
        )
        logits = mask_logits(model(z_t, t_pos, modality_ids), layout=layout)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + (next_t_pos - t_pos).unsqueeze(-1) * v_t
        last_logits = logits
        if condition_image_tokens is not None:
            z_t[:, layout.image_slice] = build_flm_clean_state(condition_image_tokens.to(device), layout.vocab_size)
        if text_targets is not None:
            z_t[:, layout.text_slice] = build_flm_clean_state(text_targets, layout.vocab_size)

    if last_logits is None:
        raise RuntimeError("sampling requires at least one step")
    final_progress = torch.full((batch,), 1.0, device=device)
    final_t_pos = apply_schedule(
        progress=final_progress,
        modality_ids=modality_ids,
        schedule_tables=schedule_tables,
        image_time_power=image_time_power,
        text_time_power=effective_text_time_power,
    )
    if is_i2t:
        final_t_pos = _apply_i2t_text_time_schedule(
            final_t_pos,
            final_progress,
            layout=layout,
            schedule=image_to_text_text_time_schedule,
            logit_normal_loc=image_to_text_logit_normal_loc,
            logit_normal_scale=image_to_text_logit_normal_scale,
        )
    final_t_pos = condition_clean_timesteps(
        final_t_pos,
        layout.image_seq_len,
        condition_image=condition_image_tokens is not None,
        condition_text=condition_text_tokens is not None,
    )
    final_logits = mask_logits(model(z_t, final_t_pos, modality_ids), layout=layout)
    return final_logits.argmax(dim=-1), final_logits


@torch.inference_mode()
def sample_unified(
    model: UnifiedDenoiser,
    layout: TaskLayout,
    schedule_tables: dict,
    temperature: float,
    steps: int,
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    image_to_text_text_time_schedule: str = "power",
    image_to_text_logit_normal_loc: float = 0.0,
    image_to_text_logit_normal_scale: float = 1.0,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_text_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens, _ = _sample_unified_with_logits(
        model=model,
        layout=layout,
        schedule_tables=schedule_tables,
        temperature=temperature,
        steps=steps,
        image_time_power=image_time_power,
        text_time_power=text_time_power,
        image_to_text_text_time_power=image_to_text_text_time_power,
        image_to_text_text_time_schedule=image_to_text_text_time_schedule,
        image_to_text_logit_normal_loc=image_to_text_logit_normal_loc,
        image_to_text_logit_normal_scale=image_to_text_logit_normal_scale,
        batch_size=batch_size,
        condition_image_tokens=condition_image_tokens,
        condition_text_tokens=condition_text_tokens,
    )
    return tokens


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

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
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
    text_to_image_correct = 0
    generated_text_counter: Counter[str] = Counter()
    total = 0
    isolate_sampling_rng = config.eval.isolate_sampling_rng
    eval_sampling_seed = config.eval.sampling_seed if config.eval.sampling_seed is not None else config.train.seed
    rng_streams = _EvalSamplingRngStreams(enabled=isolate_sampling_rng, base_seed=eval_sampling_seed, device=device)

    for batch in tqdm(test_loader, desc="eval"):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)

        decoded = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        ceiling_pred = classify_images(classifier, decoded, config.dataset.name)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        with rng_streams.branch("image_to_text"):
            sampled_tokens, _ = _sample_unified_with_logits(
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
                condition_image_tokens=image_tokens,
                condition_text_tokens=None,
            )
        sampled_text = sampled_tokens[:, layout.text_slice] - layout.text_offset
        sampled_text_strings = decode_text_tokens(sampled_text, text_metadata)
        target_text_strings = decode_text_tokens(text_tokens, text_metadata)
        image_to_text_exact += sum(pred == target for pred, target in zip(sampled_text_strings, target_text_strings))
        valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
        image_to_text_token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
        image_to_text_token_total += valid_text.sum().item()
        generated_text_counter.update(sampled_text_strings)

        with rng_streams.branch("text_to_image"):
            sampled_images = sample_unified(
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
                condition_image_tokens=None,
                condition_text_tokens=text_tokens,
            )[:, layout.image_slice]
        decoded_images = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
        image_pred = classify_images(classifier, decoded_images, config.dataset.name)
        text_to_image_correct += (image_pred == labels).sum().item()
        total += labels.numel()

    with rng_streams.branch("unconditional"):
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
            batch_size=config.eval.num_unconditional_samples,
            condition_image_tokens=None,
            condition_text_tokens=None,
        )
    uncond_images = _decode_image_tokens(tokenizer, sampled[:, layout.image_slice], tokenizer_state, grid_shape, device)
    uncond_image_pred = classify_images(classifier, uncond_images, config.dataset.name)
    uncond_text_values = label_values_from_text_tokens(sampled[:, layout.text_slice] - layout.text_offset, text_metadata)
    consistency = (uncond_image_pred == uncond_text_values.to(device)).float().mean().item()

    metrics = {
        "tokenizer_ceiling": ceiling_correct / max(total, 1),
        "image_to_text_exact_match": image_to_text_exact / max(total, 1),
        "image_to_text_token_accuracy": image_to_text_token_correct / max(image_to_text_token_total, 1),
        "text_to_image_accuracy": text_to_image_correct / max(total, 1),
        "unconditional_consistency": consistency,
    }
    with run_context.log_path("metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    diagnostics = {
        "isolate_sampling_rng": isolate_sampling_rng,
        "sampling_seed": eval_sampling_seed,
        "image_to_text_generated_text_counts": dict(generated_text_counter.most_common(32)),
        "label_strings": list(text_metadata.label_strings),
    }
    with run_context.log_path("diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return metrics
