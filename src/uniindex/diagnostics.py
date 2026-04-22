from __future__ import annotations

import json
from collections import Counter

import torch
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, split_path
from .eval import _load_stage2, constrained_text_label_values
from .layout import mask_logits, unified_targets
from .runtime import RunContext, ensure_project_dirs, resolve_device, set_seed
from .schedule import build_schedule_tables
from .state import build_flm_clean_state, condition_clean_timesteps, mix_flm_noise
from .text import decode_text_tokens, metadata_from_state, text_scoring_mask
from .train import _task_time_schedule


DEFAULT_I2T_PROGRESS = (0.25, 0.5, 0.75, 0.9, 0.95)
IMAGE_DEPENDENCE_MODES = ("true_image", "shuffled_image", "random_image_tokens")


def _image_condition_tokens(
    image_tokens: torch.Tensor,
    *,
    mode: str,
    codebook_size: int,
) -> torch.Tensor:
    if mode == "true_image":
        return image_tokens
    if mode == "shuffled_image":
        order = torch.arange(image_tokens.shape[0], device=image_tokens.device).roll(1)
        return image_tokens.index_select(0, order)
    if mode == "random_image_tokens":
        return torch.randint(
            low=0,
            high=codebook_size,
            size=image_tokens.shape,
            device=image_tokens.device,
        )
    raise ValueError(f"unsupported image-dependence mode: {mode}")


def _image_dependence_margins(results: list[dict]) -> list[dict]:
    by_progress = {}
    for result in results:
        by_progress.setdefault(float(result["progress"]), {})[result["mode"]] = result

    margins = []
    for progress, modes in sorted(by_progress.items()):
        true_result = modes.get("true_image")
        shuffled = modes.get("shuffled_image")
        random_tokens = modes.get("random_image_tokens")
        if true_result is None:
            continue
        record = {"progress": progress}
        if shuffled is not None:
            record["true_minus_shuffled_label"] = (
                true_result["image_to_text_label_accuracy_constrained"]
                - shuffled["image_to_text_label_accuracy_constrained"]
            )
            record["true_minus_shuffled_token"] = (
                true_result["image_to_text_token_accuracy"] - shuffled["image_to_text_token_accuracy"]
            )
            record["true_minus_shuffled_exact"] = (
                true_result["image_to_text_exact_match"] - shuffled["image_to_text_exact_match"]
            )
        if random_tokens is not None:
            record["true_minus_random_label"] = (
                true_result["image_to_text_label_accuracy_constrained"]
                - random_tokens["image_to_text_label_accuracy_constrained"]
            )
            record["true_minus_random_token"] = (
                true_result["image_to_text_token_accuracy"] - random_tokens["image_to_text_token_accuracy"]
            )
            record["true_minus_random_exact"] = (
                true_result["image_to_text_exact_match"] - random_tokens["image_to_text_exact_match"]
            )
        margins.append(record)
    return margins


@torch.inference_mode()
def diagnose_i2t_denoiser(
    config: ProjectConfig,
    progress_values: tuple[float, ...] = DEFAULT_I2T_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)

    results = []
    for progress_value in progress_values:
        exact = 0
        token_correct = 0
        token_total = 0
        constrained_correct = 0
        position_correct = torch.zeros(layout.text_seq_len, dtype=torch.long)
        position_total = torch.zeros(layout.text_seq_len, dtype=torch.long)
        generated_counter: Counter[str] = Counter()
        total = 0
        image_t_sum = 0.0
        text_t_sum = 0.0

        for batch in tqdm(test_loader, desc=f"diagnose-i2t@{progress_value:g}"):
            image_tokens = batch["image_tokens"].to(device)
            text_tokens = batch["text_tokens"].to(device)
            labels = batch["label"].to(device)
            batch_size = labels.shape[0]

            targets = unified_targets(image_tokens, text_tokens, layout.codebook_size)
            x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
            progress = torch.full((batch_size,), float(progress_value), device=device)
            t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
            t_pos = condition_clean_timesteps(
                t_pos,
                layout.image_seq_len,
                condition_image=True,
                condition_text=False,
            )

            z_t = x1.clone()
            z_t[:, layout.text_slice] = mix_flm_noise(
                x1[:, layout.text_slice],
                t_pos[:, layout.text_slice],
                valid_token_mask[layout.text_slice],
            )
            logits = model(z_t, t_pos, modality_ids)
            logits = mask_logits(logits, layout=layout)
            text_logits = logits[:, layout.text_slice]
            sampled_text = text_logits.argmax(dim=-1) - layout.text_offset

            sampled_strings = decode_text_tokens(sampled_text, text_metadata)
            target_strings = decode_text_tokens(text_tokens, text_metadata)
            exact += sum(pred == target for pred, target in zip(sampled_strings, target_strings))
            valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
            token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
            token_total += valid_text.sum().item()
            position_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum(dim=0).cpu()
            position_total += valid_text.sum(dim=0).cpu()
            generated_counter.update(sampled_strings)
            constrained_values = constrained_text_label_values(
                text_logits,
                text_metadata,
                codebook_size=layout.codebook_size,
            )
            constrained_correct += (constrained_values == labels).sum().item()
            total += batch_size
            image_t_sum += float(t_pos[:, layout.image_slice].mean().item()) * batch_size
            text_t_sum += float(t_pos[:, layout.text_slice].mean().item()) * batch_size

        results.append(
            {
                "progress": float(progress_value),
                "image_t_mean": image_t_sum / max(total, 1),
                "text_t_mean": text_t_sum / max(total, 1),
                "image_to_text_exact_match": exact / max(total, 1),
                "image_to_text_token_accuracy": token_correct / max(token_total, 1),
                "image_to_text_label_accuracy_constrained": constrained_correct / max(total, 1),
                "image_to_text_position_accuracy": [
                    correct / max(total_count, 1)
                    for correct, total_count in zip(position_correct.tolist(), position_total.tolist())
                ],
                "image_to_text_generated_text_counts": dict(generated_counter.most_common(32)),
            }
        )

    summary = {
        "progress_values": [float(value) for value in progress_values],
        "results": results,
        "label_strings": list(text_metadata.label_strings),
    }
    path = run_context.log_path("i2t_diagnostics.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary


@torch.inference_mode()
def diagnose_i2t_image_dependence(
    config: ProjectConfig,
    progress_values: tuple[float, ...] = DEFAULT_I2T_PROGRESS,
    run_context: RunContext | None = None,
) -> dict:
    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    own_context = run_context is None
    run_context = run_context or RunContext(config, "diagnose-i2t-image-dependence")
    run_context.set_device(device)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    test_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    modality_ids = layout.position_modalities().to(device)
    valid_token_mask = layout.position_valid_token_mask().to(device)

    results = []
    for progress_value in progress_values:
        for mode in IMAGE_DEPENDENCE_MODES:
            exact = 0
            token_correct = 0
            token_total = 0
            constrained_correct = 0
            total = 0
            image_t_sum = 0.0
            text_t_sum = 0.0

            for batch in tqdm(test_loader, desc=f"diagnose-i2t-image-dependence/{mode}@{progress_value:g}"):
                image_tokens = batch["image_tokens"].to(device)
                text_tokens = batch["text_tokens"].to(device)
                labels = batch["label"].to(device)
                batch_size = labels.shape[0]

                condition_image_tokens = _image_condition_tokens(
                    image_tokens,
                    mode=mode,
                    codebook_size=layout.codebook_size,
                )
                targets = unified_targets(condition_image_tokens, text_tokens, layout.codebook_size)
                x1 = build_flm_clean_state(targets, layout.vocab_size).to(device)
                progress = torch.full((batch_size,), float(progress_value), device=device)
                t_pos = _task_time_schedule(config, progress, modality_ids, schedule_tables, "image_to_text")
                t_pos = condition_clean_timesteps(
                    t_pos,
                    layout.image_seq_len,
                    condition_image=True,
                    condition_text=False,
                )

                z_t = x1.clone()
                z_t[:, layout.text_slice] = mix_flm_noise(
                    x1[:, layout.text_slice],
                    t_pos[:, layout.text_slice],
                    valid_token_mask[layout.text_slice],
                )
                logits = model(z_t, t_pos, modality_ids)
                logits = mask_logits(logits, layout=layout)
                text_logits = logits[:, layout.text_slice]
                sampled_text = text_logits.argmax(dim=-1) - layout.text_offset

                sampled_strings = decode_text_tokens(sampled_text, text_metadata)
                target_strings = decode_text_tokens(text_tokens, text_metadata)
                exact += sum(pred == target for pred, target in zip(sampled_strings, target_strings))
                valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True)
                token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
                token_total += valid_text.sum().item()
                constrained_values = constrained_text_label_values(
                    text_logits,
                    text_metadata,
                    codebook_size=layout.codebook_size,
                )
                constrained_correct += (constrained_values == labels).sum().item()
                total += batch_size
                image_t_sum += float(t_pos[:, layout.image_slice].mean().item()) * batch_size
                text_t_sum += float(t_pos[:, layout.text_slice].mean().item()) * batch_size

            results.append(
                {
                    "progress": float(progress_value),
                    "mode": mode,
                    "image_t_mean": image_t_sum / max(total, 1),
                    "text_t_mean": text_t_sum / max(total, 1),
                    "image_to_text_exact_match": exact / max(total, 1),
                    "image_to_text_token_accuracy": token_correct / max(token_total, 1),
                    "image_to_text_label_accuracy_constrained": constrained_correct / max(total, 1),
                }
            )

    summary = {
        "progress_values": [float(value) for value in progress_values],
        "modes": list(IMAGE_DEPENDENCE_MODES),
        "results": results,
        "margins": _image_dependence_margins(results),
        "label_strings": list(text_metadata.label_strings),
    }
    path = run_context.log_path("i2t_image_dependence.json")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if own_context:
        run_context.update_status("ok")
    return summary
