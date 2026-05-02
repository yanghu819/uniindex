from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from uniindex.config import load_config
from uniindex.data import build_loader, split_path
from uniindex.eval import _load_stage2, _sample_unified_with_logits, constrained_text_label_values
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.text import decode_text_tokens, metadata_from_state, text_scoring_mask


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate image-to-text on tokenized data only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)

    model, tokenizer_state, layout = _load_stage2(config, device)
    text_metadata = metadata_from_state(tokenizer_state)
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
    token_correct = 0
    token_total = 0
    constrained_correct = 0
    total = 0
    generated_text_counter: Counter[str] = Counter()
    started = time.time()

    for batch in tqdm(loader, desc="i2t-token-eval"):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)

        sampled_tokens, final_logits = _sample_unified_with_logits(
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

        exact += sum(pred == target for pred, target in zip(pred_strings, target_strings))
        valid_text = text_scoring_mask(text_tokens, text_metadata, include_bos=False, include_eos=True).to(device)
        token_correct += sampled_text.eq(text_tokens).logical_and(valid_text).sum().item()
        token_total += valid_text.sum().item()
        constrained_values = constrained_text_label_values(
            final_logits[:, layout.text_slice],
            text_metadata,
            codebook_size=layout.codebook_size,
        )
        constrained_correct += (constrained_values == labels).sum().item()
        total += labels.numel()
        generated_text_counter.update(pred_strings)

    metrics = {
        "image_to_text_exact_match": exact / max(total, 1),
        "image_to_text_token_accuracy": token_correct / max(token_total, 1),
        "image_to_text_label_accuracy_constrained": constrained_correct / max(total, 1),
        "total": total,
        "elapsed_sec": round(time.time() - started, 3),
        "sampling_steps": sampling_steps,
        "temperature": temperature,
        "generated_text_top20": generated_text_counter.most_common(20),
    }
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
    sampling_steps = args.sampling_steps or config.sampling.steps
    temperature = args.temperature if args.temperature is not None else config.sampling.temperature
