from __future__ import annotations

from contextlib import contextmanager
import json
from collections import Counter
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
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise, restore_image_tokens, sample_masked_noise
from .text import (
    decode_text_tokens,
    label_values_from_text_tokens,
    metadata_from_state,
    sequence_candidate_scores,
    shifted_label_text_tokens,
    text_scoring_mask,
)
from .tokenizer import build_tokenizer
from .train import latest_checkpoint_path


_CANDIDATE_SCORE_CHUNK_SIZE = 32
_EVAL_RNG_BRANCH_OFFSETS = {
    "image_to_text": 10_000,
    "text_to_image": 20_000,
    "unconditional": 30_000,
}


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
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer_state, layout


def _decode_image_tokens(tokenizer, image_tokens: torch.Tensor, tokenizer_state: dict, grid_shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    restored = restore_image_tokens(image_tokens.cpu(), tokenizer_state)
    return tokenizer.decode_token_batch(restored, grid_shape).to(device)


@contextmanager
def _eval_sampling_rng(
    *,
    enabled: bool,
    base_seed: int,
    device: torch.device,
    branch: str,
    batch_index: int,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    if branch not in _EVAL_RNG_BRANCH_OFFSETS:
        raise ValueError(f"unsupported eval RNG branch: {branch}")
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices.append(device.index if device.index is not None else torch.cuda.current_device())
    seed = int(base_seed) + _EVAL_RNG_BRANCH_OFFSETS[branch] + int(batch_index) * 1_009
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(seed)
        yield


def _projection_step_index(steps: int, progress: float) -> int:
    if steps < 1:
        raise ValueError(f"sampling steps must be >= 1, got {steps}")
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError(f"projection progress must be in [0, 1], got {progress}")
    return min(range(steps), key=lambda index: abs((index / steps) - float(progress)))


def _projection_step_indices(steps: int, progresses: list[float]) -> set[int]:
    if not progresses:
        raise ValueError("projection progresses must contain at least one value")
    return {_projection_step_index(steps, float(progress)) for progress in progresses}


def _project_text_state(
    text_logits: torch.Tensor,
    *,
    layout: TaskLayout,
    projection: str,
    text_metadata,
) -> torch.Tensor:
    if projection == "argmax_renoise":
        return text_logits.argmax(dim=-1)
    if projection == "candidate_renoise":
        if text_metadata is None:
            raise ValueError("candidate_renoise projection requires text_metadata")
        candidate_targets = shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size).to(
            text_logits.device
        )
        scores = sequence_candidate_scores(text_logits, candidate_targets)
        selected = scores.argmax(dim=1)
        return candidate_targets.index_select(0, selected)
    raise ValueError(f"unsupported image_to_text_projection: {projection}")


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
    integrator: str = "legacy_progress_euler",
    final_decode: str = "final_model_call",
    final_model_progress: float = 1.0,
    image_to_text_projection: str = "none",
    image_to_text_projection_progress: float = 0.5,
    image_to_text_projection_progresses: list[float] | None = None,
    text_metadata=None,
    batch_size: int | None = None,
    condition_image_tokens: torch.Tensor | None = None,
    condition_text_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if steps < 1:
        raise ValueError(f"sampling steps must be >= 1, got {steps}")
    if integrator not in {"scheduled_euler", "legacy_progress_euler"}:
        raise ValueError(f"unsupported sampling integrator: {integrator}")
    if final_decode not in {"last_endpoint", "final_model_call"}:
        raise ValueError(f"unsupported sampling final_decode: {final_decode}")
    if not 0.0 <= final_model_progress <= 1.0:
        raise ValueError(f"final_model_progress must be in [0, 1], got {final_model_progress}")
    if image_to_text_projection not in {"none", "argmax_renoise", "candidate_renoise"}:
        raise ValueError(f"unsupported image_to_text_projection: {image_to_text_projection}")
    if not 0.0 <= image_to_text_projection_progress <= 1.0:
        raise ValueError(
            f"image_to_text_projection_progress must be in [0, 1], got {image_to_text_projection_progress}"
        )
    projection_progresses = (
        [float(progress) for progress in image_to_text_projection_progresses]
        if image_to_text_projection_progresses is not None
        else [float(image_to_text_projection_progress)]
    )
    if not projection_progresses:
        raise ValueError("image_to_text_projection_progresses must contain at least one value")
    for progress in projection_progresses:
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"image_to_text_projection_progresses values must be in [0, 1], got {progress}")

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
    if condition_image_tokens is not None and condition_text_tokens is None and image_to_text_text_time_power is not None:
        effective_text_time_power = image_to_text_text_time_power
    should_project_i2t = (
        image_to_text_projection != "none"
        and condition_image_tokens is not None
        and condition_text_tokens is None
    )
    projection_steps = _projection_step_indices(steps, projection_progresses) if should_project_i2t else set()

    last_logits = None
    for step in range(steps):
        progress = torch.full((batch,), step / steps, device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=condition_image_tokens is not None,
            condition_text=condition_text_tokens is not None,
        )
        if integrator == "scheduled_euler":
            next_progress = torch.full((batch,), (step + 1) / steps, device=device)
            next_t_pos = apply_schedule(
                progress=next_progress,
                modality_ids=modality_ids,
                schedule_tables=schedule_tables,
                image_time_power=image_time_power,
                text_time_power=effective_text_time_power,
            )
            next_t_pos = condition_clean_timesteps(
                next_t_pos,
                layout.image_seq_len,
                condition_image=condition_image_tokens is not None,
                condition_text=condition_text_tokens is not None,
            )
            dt_pos = next_t_pos - t_pos
        else:
            dt_pos = torch.full_like(t_pos, 1.0 / steps)
            next_progress = torch.full((batch,), (step + 1) / steps, device=device)
            next_t_pos = apply_schedule(
                progress=next_progress,
                modality_ids=modality_ids,
                schedule_tables=schedule_tables,
                image_time_power=image_time_power,
                text_time_power=effective_text_time_power,
            )
            next_t_pos = condition_clean_timesteps(
                next_t_pos,
                layout.image_seq_len,
                condition_image=condition_image_tokens is not None,
                condition_text=condition_text_tokens is not None,
            )

        logits = model(z_t, t_pos, modality_ids)
        logits = mask_logits(logits, layout=layout)
        probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1)
        v_t = (probs - z_t) / (1.0 - t_pos).unsqueeze(-1).clamp_min(1e-4)
        z_t = z_t + dt_pos.unsqueeze(-1) * v_t
        last_logits = logits
        if condition_image_tokens is not None:
            z_t[:, layout.image_slice] = build_flm_clean_state(condition_image_tokens.to(device), layout.vocab_size)
        if text_targets is not None:
            z_t[:, layout.text_slice] = build_flm_clean_state(text_targets, layout.vocab_size)
        elif should_project_i2t and step in projection_steps:
            projected_targets = _project_text_state(
                logits[:, layout.text_slice],
                layout=layout,
                projection=image_to_text_projection,
                text_metadata=text_metadata,
            )
            projected_clean = build_flm_clean_state(projected_targets, layout.vocab_size)
            z_t[:, layout.text_slice] = mix_flm_noise(
                projected_clean,
                next_t_pos[:, layout.text_slice],
                valid_token_mask[layout.text_slice],
            )

    if final_decode == "final_model_call":
        final_t_pos = apply_schedule(
            progress=torch.full((batch,), float(final_model_progress), device=device),
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        final_t_pos = condition_clean_timesteps(
            final_t_pos,
            layout.image_seq_len,
            condition_image=condition_image_tokens is not None,
            condition_text=condition_text_tokens is not None,
        )
        final_logits = model(z_t, final_t_pos, modality_ids)
        final_logits = mask_logits(final_logits, layout=layout)
    else:
        if last_logits is None:
            raise RuntimeError("last endpoint decode requires at least one sampling step")
        final_logits = last_logits
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
    integrator: str = "legacy_progress_euler",
    final_decode: str = "final_model_call",
    final_model_progress: float = 1.0,
    image_to_text_projection: str = "none",
    image_to_text_projection_progress: float = 0.5,
    image_to_text_projection_progresses: list[float] | None = None,
    text_metadata=None,
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
        integrator=integrator,
        final_decode=final_decode,
        final_model_progress=final_model_progress,
        image_to_text_projection=image_to_text_projection,
        image_to_text_projection_progress=image_to_text_projection_progress,
        image_to_text_projection_progresses=image_to_text_projection_progresses,
        text_metadata=text_metadata,
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


def _candidate_score_progress_values(progress_values: list[float] | None) -> list[float]:
    values = [0.5, 0.75, 0.9, 0.95] if progress_values is None else list(progress_values)
    if not values:
        raise ValueError("candidate_score_progress must contain at least one value")
    for value in values:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"candidate_score_progress values must be in [0, 1], got {value}")
    return [float(value) for value in values]


@torch.inference_mode()
def _candidate_denoiser_score_text(
    model: UnifiedDenoiser,
    layout: TaskLayout,
    schedule_tables: dict,
    image_tokens: torch.Tensor,
    text_metadata,
    *,
    image_time_power: float,
    text_time_power: float,
    image_to_text_text_time_power: float | None,
    progress_values: list[float] | None,
    num_noise: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_noise < 1:
        raise ValueError(f"candidate_score_num_noise must be >= 1, got {num_noise}")

    device = image_tokens.device
    batch = image_tokens.shape[0]
    candidate_text = text_metadata.label_text_tokens.to(device)
    candidate_targets = shifted_label_text_tokens(text_metadata, token_offset=layout.codebook_size).to(device)
    if candidate_targets.shape[1] != layout.text_seq_len:
        raise ValueError(
            f"candidate text length {candidate_targets.shape[1]} does not match layout text length {layout.text_seq_len}"
        )

    num_candidates = candidate_targets.shape[0]
    flat_batch = batch * num_candidates
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)
    text_valid_token_mask = valid_token_mask[layout.text_slice]
    effective_text_time_power = text_time_power
    if image_to_text_text_time_power is not None:
        effective_text_time_power = image_to_text_text_time_power

    flat_image_tokens = image_tokens[:, None, :].expand(batch, num_candidates, layout.image_seq_len)
    flat_image_tokens = flat_image_tokens.reshape(flat_batch, layout.image_seq_len)
    flat_candidate_targets = candidate_targets[None, :, :].expand(batch, num_candidates, layout.text_seq_len)
    flat_candidate_targets = flat_candidate_targets.reshape(flat_batch, layout.text_seq_len)
    flat_candidate_indices = torch.arange(num_candidates, device=device).repeat(batch)
    progress_schedule = _candidate_score_progress_values(progress_values)

    scores = torch.zeros(batch, num_candidates, device=device)
    flat_scores = scores.reshape(-1)
    for progress_value in progress_schedule:
        progress = torch.full((flat_batch,), progress_value, device=device)
        t_pos = apply_schedule(
            progress=progress,
            modality_ids=modality_ids,
            schedule_tables=schedule_tables,
            image_time_power=image_time_power,
            text_time_power=effective_text_time_power,
        )
        t_pos = condition_clean_timesteps(
            t_pos,
            layout.image_seq_len,
            condition_image=True,
            condition_text=False,
        )
        for _ in range(num_noise):
            for start in range(0, flat_batch, _CANDIDATE_SCORE_CHUNK_SIZE):
                end = min(start + _CANDIDATE_SCORE_CHUNK_SIZE, flat_batch)
                chunk_t_pos = t_pos[start:end]
                z_t = torch.zeros(end - start, layout.seq_len, layout.vocab_size, device=device)
                z_t[:, layout.image_slice] = build_flm_clean_state(flat_image_tokens[start:end], layout.vocab_size)
                clean_text = build_flm_clean_state(flat_candidate_targets[start:end], layout.vocab_size)
                z_t[:, layout.text_slice] = mix_flm_noise(
                    clean_text,
                    chunk_t_pos[:, layout.text_slice],
                    text_valid_token_mask,
                )

                logits = model(z_t, chunk_t_pos, modality_ids)
                logits = mask_logits(logits, layout=layout)
                candidate_scores = sequence_candidate_scores(logits[:, layout.text_slice], candidate_targets)
                own_scores = candidate_scores.gather(
                    dim=1,
                    index=flat_candidate_indices[start:end, None],
                ).squeeze(1)
                flat_scores[start:end] += own_scores

    scores /= float(len(progress_schedule) * num_noise)
    selected_indices = scores.argmax(dim=1)
    selected_text = candidate_text.index_select(0, selected_indices)
    return selected_text, selected_indices, scores


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
    image_to_text_constrained_correct = 0
    text_to_image_correct = 0
    image_to_text_position_correct = torch.zeros(layout.text_seq_len, dtype=torch.long)
    image_to_text_position_total = torch.zeros(layout.text_seq_len, dtype=torch.long)
    generated_text_counter: Counter[str] = Counter()
    total = 0
    image_to_text_decoder = config.sampling.image_to_text_decoder
    if image_to_text_decoder not in {"sample", "candidate_denoiser_score"}:
        raise ValueError(f"unsupported image_to_text_decoder: {image_to_text_decoder}")
    label_values = torch.tensor(text_metadata.label_values, dtype=torch.long, device=device)
    isolate_sampling_rng = config.eval.isolate_sampling_rng
    eval_sampling_seed = config.eval.sampling_seed if config.eval.sampling_seed is not None else config.train.seed

    for batch_index, batch in enumerate(tqdm(test_loader, desc="eval")):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)

        decoded = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        ceiling_pred = classify_images(classifier, decoded, config.dataset.name)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        with _eval_sampling_rng(
            enabled=isolate_sampling_rng,
            base_seed=eval_sampling_seed,
            device=device,
            branch="image_to_text",
            batch_index=batch_index,
        ):
            if image_to_text_decoder == "sample":
                sampled_tokens, final_logits = _sample_unified_with_logits(
                    model=model,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    temperature=config.sampling.temperature,
                    steps=config.sampling.steps,
                    image_time_power=config.sampling.image_time_power,
                    text_time_power=config.sampling.text_time_power,
                    image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
                    integrator=config.sampling.integrator,
                    final_decode=config.sampling.final_decode,
                    final_model_progress=config.sampling.final_model_progress,
                    image_to_text_projection=config.sampling.image_to_text_projection,
                    image_to_text_projection_progress=config.sampling.image_to_text_projection_progress,
                    image_to_text_projection_progresses=config.sampling.image_to_text_projection_progresses,
                    text_metadata=text_metadata,
                    condition_image_tokens=image_tokens,
                    condition_text_tokens=None,
                )
                sampled_text = sampled_tokens[:, layout.text_slice] - layout.text_offset
                constrained_values = constrained_text_label_values(
                    final_logits[:, layout.text_slice],
                    text_metadata,
                    codebook_size=layout.codebook_size,
                )
            else:
                sampled_text, selected_candidate_indices, _ = _candidate_denoiser_score_text(
                    model=model,
                    layout=layout,
                    schedule_tables=schedule_tables,
                    image_tokens=image_tokens,
                    text_metadata=text_metadata,
                    image_time_power=config.sampling.image_time_power,
                    text_time_power=config.sampling.text_time_power,
                    image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
                    progress_values=config.sampling.candidate_score_progress,
                    num_noise=config.sampling.candidate_score_num_noise,
                )
                constrained_values = label_values.index_select(0, selected_candidate_indices)

        sampled_text_strings = decode_text_tokens(sampled_text, text_metadata)
        target_text_strings = decode_text_tokens(text_tokens, text_metadata)
        image_to_text_exact += sum(pred == target for pred, target in zip(sampled_text_strings, target_text_strings))
        valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
        image_to_text_token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
        image_to_text_token_total += valid_text.sum().item()
        image_to_text_position_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum(dim=0).cpu()
        image_to_text_position_total += valid_text.sum(dim=0).cpu()
        generated_text_counter.update(sampled_text_strings)
        image_to_text_constrained_correct += (constrained_values == labels).sum().item()

        with _eval_sampling_rng(
            enabled=isolate_sampling_rng,
            base_seed=eval_sampling_seed,
            device=device,
            branch="text_to_image",
            batch_index=batch_index,
        ):
            sampled_images = sample_unified(
                model=model,
                layout=layout,
                schedule_tables=schedule_tables,
                temperature=config.sampling.temperature,
                steps=config.sampling.steps,
                image_time_power=config.sampling.image_time_power,
                text_time_power=config.sampling.text_time_power,
                image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
                integrator=config.sampling.integrator,
                final_decode=config.sampling.final_decode,
                final_model_progress=config.sampling.final_model_progress,
                image_to_text_projection=config.sampling.image_to_text_projection,
                image_to_text_projection_progress=config.sampling.image_to_text_projection_progress,
                image_to_text_projection_progresses=config.sampling.image_to_text_projection_progresses,
                text_metadata=text_metadata,
                condition_image_tokens=None,
                condition_text_tokens=text_tokens,
            )[:, layout.image_slice]
        decoded_images = _decode_image_tokens(tokenizer, sampled_images, tokenizer_state, grid_shape, device)
        image_pred = classify_images(classifier, decoded_images, config.dataset.name)
        text_to_image_correct += (image_pred == labels).sum().item()

        total += labels.numel()

    uncond_count = config.eval.num_unconditional_samples
    with _eval_sampling_rng(
        enabled=isolate_sampling_rng,
        base_seed=eval_sampling_seed,
        device=device,
        branch="unconditional",
        batch_index=0,
    ):
        sampled = sample_unified(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            temperature=config.sampling.temperature,
            steps=config.sampling.steps,
            image_time_power=config.sampling.image_time_power,
            text_time_power=config.sampling.text_time_power,
            image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
            integrator=config.sampling.integrator,
            final_decode=config.sampling.final_decode,
            final_model_progress=config.sampling.final_model_progress,
            image_to_text_projection=config.sampling.image_to_text_projection,
            image_to_text_projection_progress=config.sampling.image_to_text_projection_progress,
            image_to_text_projection_progresses=config.sampling.image_to_text_projection_progresses,
            text_metadata=text_metadata,
            batch_size=uncond_count,
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
        "image_to_text_label_accuracy_constrained": image_to_text_constrained_correct / max(total, 1),
        "text_to_image_accuracy": text_to_image_correct / max(total, 1),
        "unconditional_consistency": consistency,
    }

    metrics_path = run_context.log_path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    diagnostics = {
        "image_to_text_decoder": image_to_text_decoder,
        "final_model_progress": config.sampling.final_model_progress,
        "image_to_text_projection": config.sampling.image_to_text_projection,
        "image_to_text_projection_progress": config.sampling.image_to_text_projection_progress,
        "image_to_text_projection_progresses": config.sampling.image_to_text_projection_progresses,
        "isolate_sampling_rng": isolate_sampling_rng,
        "sampling_seed": eval_sampling_seed,
        "candidate_score_progress": _candidate_score_progress_values(config.sampling.candidate_score_progress)
        if image_to_text_decoder == "candidate_denoiser_score"
        else None,
        "candidate_score_num_noise": config.sampling.candidate_score_num_noise
        if image_to_text_decoder == "candidate_denoiser_score"
        else None,
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
        "generated_text_strings": decode_text_tokens(sampled[:, layout.text_slice] - layout.text_offset, text_metadata),
        "image_to_text_generated_text_counts": dict(generated_text_counter.most_common(32)),
        "label_strings": list(text_metadata.label_strings),
    }
    preview_path = run_context.log_path("text_preview.json")
    with preview_path.open("w", encoding="utf-8") as handle:
        json.dump(preview, handle, indent=2)

    if own_context:
        run_context.update_status("ok")
    return metrics
