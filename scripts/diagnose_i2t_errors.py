from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
from tqdm import tqdm

from uniindex.config import load_config
from uniindex.data import build_loader, split_path
from uniindex.eval import _load_stage2, _sample_unified_with_logits
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.text import decode_text_tokens, label_values_from_text_tokens, metadata_from_state, text_scoring_mask


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose image-to-text exact-match failures.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--image-to-text-text-time-power", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.image_to_text_text_time_power is not None:
        config = replace(
            config,
            train=replace(config.train, image_to_text_text_time_power=args.image_to_text_text_time_power),
            sampling=replace(config.sampling, image_to_text_text_time_power=args.image_to_text_text_time_power),
        )
    sampling_steps = args.sampling_steps or config.sampling.steps
    temperature = args.temperature if args.temperature is not None else config.sampling.temperature
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
    label_values = list(text_metadata.label_values)
    label_to_index = {int(value): index for index, value in enumerate(label_values)}
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    exact = 0
    canonical_correct = 0
    canonical_total = 0
    invalid = 0
    token_correct = 0
    token_total = 0
    total = 0
    generated_text_counter: Counter[str] = Counter()
    invalid_text_counter: Counter[str] = Counter()
    confusion = torch.zeros(len(label_values), len(label_values), dtype=torch.long)
    per_class_total = torch.zeros(len(label_values), dtype=torch.long)
    per_class_exact = torch.zeros(len(label_values), dtype=torch.long)
    position_correct = torch.zeros(layout.text_seq_len, dtype=torch.long)
    position_total = torch.zeros(layout.text_seq_len, dtype=torch.long)
    started = time.time()

    for batch in tqdm(loader, desc="i2t-error-diagnosis"):
        if args.max_samples is not None and total >= args.max_samples:
            break
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].long()
        if args.max_samples is not None:
            remaining = args.max_samples - total
            if remaining <= 0:
                break
            image_tokens = image_tokens[:remaining]
            text_tokens = text_tokens[:remaining]
            labels = labels[:remaining]

        sampled_tokens, _ = _sample_unified_with_logits(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            temperature=temperature,
            steps=sampling_steps,
            image_time_power=config.sampling.image_time_power,
            text_time_power=config.sampling.text_time_power,
            image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
            condition_image_tokens=image_tokens,
            condition_text_tokens=None,
        )
        sampled_text = sampled_tokens[:, layout.text_slice] - layout.text_offset
        pred_strings = decode_text_tokens(sampled_text, text_metadata)
        target_strings = decode_text_tokens(text_tokens, text_metadata)
        pred_values = label_values_from_text_tokens(sampled_text, text_metadata, unknown_value=-1).cpu()

        valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True).to(device)
        matches = sampled_text.eq(text_tokens).cpu()
        position_correct += matches.logical_and(valid_text.cpu()).sum(dim=0)
        position_total += valid_text.cpu().sum(dim=0)
        token_correct += matches.to(device).logical_and(valid_text).sum().item()
        token_total += valid_text.sum().item()

        for label, pred_value, pred_string, target_string in zip(
            labels.tolist(),
            pred_values.tolist(),
            pred_strings,
            target_strings,
        ):
            label = int(label)
            label_index = label_to_index[label]
            per_class_total[label_index] += 1
            generated_text_counter[pred_string] += 1
            is_exact = pred_string == target_string
            exact += int(is_exact)
            per_class_exact[label_index] += int(is_exact)
            if pred_value == -1:
                invalid += 1
                invalid_text_counter[pred_string] += 1
                continue
            canonical_total += 1
            pred_index = label_to_index[int(pred_value)]
            confusion[label_index, pred_index] += 1
            is_canonical_correct = int(pred_value) == label
            canonical_correct += int(is_canonical_correct)

        total += text_tokens.shape[0]

    per_class_accuracy = {
        str(label): (int(per_class_exact[index].item()) / max(int(per_class_total[index].item()), 1))
        for index, label in enumerate(label_values)
    }
    metrics = {
        "image_to_text_exact_match": exact / max(total, 1),
        "image_to_text_token_accuracy": token_correct / max(token_total, 1),
        "canonical_label_accuracy": canonical_correct / max(total, 1),
        "canonical_label_accuracy_given_valid": canonical_correct / max(canonical_total, 1),
        "canonical_valid_rate": canonical_total / max(total, 1),
        "invalid_rate": invalid / max(total, 1),
        "total": total,
        "elapsed_sec": round(time.time() - started, 3),
        "sampling_steps": sampling_steps,
        "temperature": temperature,
        "max_samples": args.max_samples,
        "label_values": label_values,
        "confusion_matrix_rows_target_cols_pred": confusion.tolist(),
        "per_class_exact_match": per_class_accuracy,
        "position_accuracy": [
            int(correct) / max(int(count), 1)
            for correct, count in zip(position_correct.tolist(), position_total.tolist())
        ],
        "generated_text_top20": generated_text_counter.most_common(20),
        "invalid_text_top20": invalid_text_counter.most_common(20),
    }
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
