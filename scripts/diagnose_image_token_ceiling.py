from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import trange

from uniindex.config import load_config
from uniindex.data import split_path
from uniindex.runtime import resolve_device, set_seed


def _batches(tokens: torch.Tensor, labels: torch.Tensor, batch_size: int, *, shuffle: bool, device: torch.device):
    indices = torch.randperm(labels.shape[0]) if shuffle else torch.arange(labels.shape[0])
    for start in range(0, labels.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield tokens[batch_indices].to(device), labels[batch_indices].to(device)


@torch.inference_mode()
def _accuracy(model: torch.nn.Module, tokens: torch.Tensor, labels: torch.Tensor, batch_size: int, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    for batch_tokens, batch_labels in _batches(tokens, labels, batch_size, shuffle=False, device=device):
        logits = model(batch_tokens)
        correct += logits.argmax(dim=-1).eq(batch_labels).sum().item()
        total += batch_labels.numel()
    return correct / max(total, 1)


class MeanTokenClassifier(torch.nn.Module):
    def __init__(self, codebook_size: int, num_classes: int) -> None:
        super().__init__()
        self.token_logits = torch.nn.Embedding(codebook_size, num_classes)
        self.bias = torch.nn.Parameter(torch.zeros(num_classes))

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        return self.token_logits(image_tokens).mean(dim=1) + self.bias


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate label ceiling from frozen image tokens.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.train.seed + 2027)
    device = resolve_device(config.train.device, config.train.gpu_index)
    train_payload = torch.load(split_path(config, "train"), map_location="cpu")
    test_payload = torch.load(split_path(config, "test"), map_location="cpu")
    train_tokens = train_payload["image_tokens"].long()
    train_labels = train_payload["labels"].long()
    test_tokens = test_payload["image_tokens"].long()
    test_labels = test_payload["labels"].long()
    codebook_size = int(train_payload["codebook_size"])
    num_classes = int(max(train_labels.max().item(), test_labels.max().item()) + 1)

    model = MeanTokenClassifier(codebook_size=codebook_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    started = time.time()
    for epoch in trange(1, args.epochs + 1, desc="image-token-ceiling"):
        model.train()
        total_loss = 0.0
        total = 0
        for batch_tokens, batch_labels in _batches(train_tokens, train_labels, args.batch_size, shuffle=True, device=device):
            logits = model(batch_tokens)
            loss = F.cross_entropy(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_labels.numel()
            total += batch_labels.numel()
        train_acc = _accuracy(model, train_tokens, train_labels, args.batch_size, device)
        test_acc = _accuracy(model, test_tokens, test_labels, args.batch_size, device)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / max(total, 1),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
            }
        )

    payload = {
        "train_size": int(train_labels.numel()),
        "test_size": int(test_labels.numel()),
        "codebook_size": codebook_size,
        "image_seq_len": int(train_tokens.shape[1]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "elapsed_sec": round(time.time() - started, 3),
        "final": history[-1],
        "history": history,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
