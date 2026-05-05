from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from uniindex.classifier import classify_images, classifier_path, load_classifier
from uniindex.config import load_config
from uniindex.data import build_loader, split_path
from uniindex.eval import _decode_image_tokens, _load_stage2, sample_unified
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables
from uniindex.tokenizer import build_tokenizer


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate text-to-image on a bounded tokenized subset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    checkpoint_config = load_config(args.checkpoint_config or args.config)
    sampling_steps = args.sampling_steps or config.sampling.steps
    temperature = args.temperature if args.temperature is not None else config.sampling.temperature
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    model, tokenizer_state, layout = _load_stage2(
        config,
        device,
        checkpoint_config=checkpoint_config,
        checkpoint_path=checkpoint_path,
    )
    schedule_tables = build_schedule_tables(
        config,
        image_vocab_size=layout.codebook_size,
        text_vocab_size=layout.text_vocab_size,
    )
    grid_shape = tuple(tokenizer_state["grid_shape"])
    tokenizer = build_tokenizer(config, device=device)
    classifier = load_classifier(
        classifier_path(config.paths.models_dir, config.dataset.name),
        config.dataset.name,
        device=device,
    )
    loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    ceiling_correct = 0
    generated_correct = 0
    total = 0
    generated_label_counter: Counter[int] = Counter()
    started = time.time()

    for batch in tqdm(loader, desc="t2i-token-eval"):
        if args.max_samples is not None and total >= args.max_samples:
            break
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)
        if args.max_samples is not None:
            remaining = args.max_samples - total
            if remaining <= 0:
                break
            image_tokens = image_tokens[:remaining]
            text_tokens = text_tokens[:remaining]
            labels = labels[:remaining]

        decoded_real = _decode_image_tokens(tokenizer, image_tokens, tokenizer_state, grid_shape, device)
        ceiling_pred = classify_images(classifier, decoded_real, config.dataset.name)
        ceiling_correct += (ceiling_pred == labels).sum().item()

        sampled = sample_unified(
            model=model,
            layout=layout,
            schedule_tables=schedule_tables,
            temperature=temperature,
            steps=sampling_steps,
            image_time_power=config.sampling.image_time_power,
            text_time_power=config.sampling.text_time_power,
            image_to_text_text_time_power=config.sampling.image_to_text_text_time_power,
            condition_image_tokens=None,
            condition_text_tokens=text_tokens,
        )[:, layout.image_slice]
        decoded_generated = _decode_image_tokens(tokenizer, sampled, tokenizer_state, grid_shape, device)
        generated_pred = classify_images(classifier, decoded_generated, config.dataset.name)
        generated_correct += (generated_pred == labels).sum().item()
        generated_label_counter.update(int(v) for v in generated_pred.detach().cpu().tolist())
        total += labels.numel()

    metrics = {
        "text_to_image_accuracy": generated_correct / max(total, 1),
        "tokenizer_ceiling": ceiling_correct / max(total, 1),
        "total": total,
        "elapsed_sec": round(time.time() - started, 3),
        "sampling_steps": sampling_steps,
        "temperature": temperature,
        "max_samples": args.max_samples,
        "generated_label_top20": generated_label_counter.most_common(20),
    }
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
