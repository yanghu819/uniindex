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
from uniindex.eval import _load_stage2, sample_unified
from uniindex.runtime import resolve_device, set_seed
from uniindex.schedule import build_schedule_tables


def _nearest_labels(
    generated: torch.Tensor,
    bank_tokens: torch.Tensor,
    bank_labels: torch.Tensor,
    *,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    best_dist = torch.full((generated.shape[0],), generated.shape[1] + 1, dtype=torch.long, device=generated.device)
    best_label = torch.full((generated.shape[0],), -1, dtype=torch.long, device=generated.device)
    for start in range(0, bank_tokens.shape[0], chunk_size):
        chunk = bank_tokens[start : start + chunk_size]
        distances = generated[:, None, :].ne(chunk[None, :, :]).sum(dim=-1)
        chunk_best_dist, chunk_best_idx = distances.min(dim=1)
        update = chunk_best_dist < best_dist
        best_dist = torch.where(update, chunk_best_dist, best_dist)
        best_label = torch.where(update, bank_labels[start + chunk_best_idx], best_label)
    return best_label, best_dist


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate text-to-image with a local image-token nearest-neighbor proxy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bank-samples", type=int, default=5000)
    parser.add_argument("--bank-chunk-size", type=int, default=512)
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
    loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    bank_images: list[torch.Tensor] = []
    bank_labels: list[torch.Tensor] = []
    bank_total = 0
    for batch in loader:
        remaining = args.bank_samples - bank_total
        if remaining <= 0:
            break
        bank_images.append(batch["image_tokens"][:remaining])
        bank_labels.append(batch["label"][:remaining])
        bank_total += bank_labels[-1].shape[0]
    if not bank_images:
        raise ValueError("empty token bank")
    bank_tokens = torch.cat(bank_images, dim=0).to(device)
    bank_label_tensor = torch.cat(bank_labels, dim=0).to(device)

    eval_loader = build_loader(
        split_path(config, "test"),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )

    nn_correct = 0
    target_token_correct = 0
    target_token_total = 0
    min_distance_sum = 0
    total = 0
    generated_label_counter: Counter[int] = Counter()
    started = time.time()

    for batch in tqdm(eval_loader, desc="t2i-token-nn-eval"):
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
        nn_labels, min_distances = _nearest_labels(
            sampled,
            bank_tokens,
            bank_label_tensor,
            chunk_size=args.bank_chunk_size,
        )
        nn_correct += (nn_labels == labels).sum().item()
        min_distance_sum += min_distances.sum().item()
        target_token_correct += sampled.eq(image_tokens).sum().item()
        target_token_total += image_tokens.numel()
        generated_label_counter.update(int(v) for v in nn_labels.detach().cpu().tolist())
        total += labels.numel()

    metrics = {
        "text_to_image_token_nn_accuracy": nn_correct / max(total, 1),
        "target_image_token_accuracy": target_token_correct / max(target_token_total, 1),
        "mean_nearest_token_distance": min_distance_sum / max(total, 1),
        "total": total,
        "bank_samples": int(bank_tokens.shape[0]),
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
