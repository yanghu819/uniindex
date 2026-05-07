from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from uniindex.config import load_config
from uniindex.data import TokenizedImageDataset, load_tokenizer_state, split_path
from uniindex.runtime import ensure_project_dirs, resolve_device, set_seed
from uniindex.text import metadata_from_state


class QwenFLMImageToText(nn.Module):
    """Qwen backbone for sparse categorical FLM over image-code + text-token sequences."""

    def __init__(
        self,
        model_path: str,
        codebook_size: int,
        *,
        dtype: torch.dtype,
        freeze_qwen: bool,
    ) -> None:
        super().__init__()
        self.qwen_lm = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.qwen = self.qwen_lm.model
        self.qwen_vocab_size = int(self.qwen_lm.config.vocab_size)
        self.codebook_size = int(codebook_size)
        self.ext_vocab_size = self.qwen_vocab_size + self.codebook_size
        hidden_size = int(self.qwen_lm.config.hidden_size)
        self.image_embed = nn.Embedding(self.codebook_size, hidden_size)
        self.modality_embed = nn.Embedding(2, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.head = nn.Linear(hidden_size, self.ext_vocab_size)
        self.head.weight.data[: self.qwen_vocab_size].copy_(self.qwen_lm.lm_head.weight.data.float())
        nn.init.zeros_(self.head.weight.data[self.qwen_vocab_size :])
        nn.init.zeros_(self.head.bias)
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad_(False)

    def _embed_ext_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        text_mask = token_ids.lt(self.qwen_vocab_size)
        safe_text = token_ids.clamp(min=0, max=self.qwen_vocab_size - 1)
        text_embeds = self.qwen.embed_tokens(safe_text)
        image_ids = (token_ids - self.qwen_vocab_size).clamp(min=0, max=self.codebook_size - 1)
        image_embeds = self.image_embed(image_ids).to(text_embeds.dtype)
        return torch.where(text_mask.unsqueeze(-1), text_embeds, image_embeds)

    def forward(self, noisy_tokens: torch.Tensor, t_pos: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self._embed_ext_tokens(noisy_tokens)
        modality_embeds = self.modality_embed(modality_ids).to(token_embeds.dtype).unsqueeze(0)
        time_embeds = self.time_mlp(t_pos.unsqueeze(-1).float()).to(token_embeds.dtype)
        hidden = self.qwen(
            inputs_embeds=token_embeds + modality_embeds + time_embeds,
            use_cache=False,
        ).last_hidden_state
        return self.head(hidden.float())


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _label_text_rows(tokenizer, label_strings: list[str], device: torch.device) -> list[torch.Tensor]:
    eos = tokenizer.eos_token_id
    rows = []
    for text in label_strings:
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
        if eos is not None:
            ids.append(eos)
        rows.append(torch.tensor(ids, dtype=torch.long, device=device))
    return rows


def _batch_text_targets(labels: torch.Tensor, label_values: list[int], rows: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    value_to_index = {value: index for index, value in enumerate(label_values)}
    selected = [rows[value_to_index[int(label.item())]] for label in labels]
    max_len = max(row.numel() for row in selected)
    ids = torch.full((len(selected), max_len), pad_id, dtype=torch.long, device=labels.device)
    mask = torch.zeros((len(selected), max_len), dtype=torch.bool, device=labels.device)
    for index, row in enumerate(selected):
        ids[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return ids, mask


def _build_targets(
    image_tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    qwen_vocab_size: int,
    label_values: list[int],
    text_rows: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image_targets = image_tokens + qwen_vocab_size
    text_targets, text_mask = _batch_text_targets(labels, label_values, text_rows, pad_id)
    targets = torch.cat([image_targets, text_targets], dim=1)
    image_mask = torch.ones_like(image_targets, dtype=torch.bool)
    valid_mask = torch.cat([image_mask, text_mask], dim=1)
    modality_ids = torch.cat(
        [
            torch.zeros(image_targets.shape[1], dtype=torch.long, device=image_targets.device),
            torch.ones(text_targets.shape[1], dtype=torch.long, device=image_targets.device),
        ]
    )
    return targets, valid_mask, modality_ids


def _sample_noisy_tokens(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    modality_ids: torch.Tensor,
    *,
    qwen_vocab_size: int,
    codebook_size: int,
    text_pad_id: int,
    task: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len = targets.shape
    device = targets.device
    progress = torch.rand(batch, device=device)
    t_pos = progress[:, None].expand(batch, seq_len).clone()
    if task == "image_to_text":
        t_pos[:, modality_ids.eq(0)] = 1.0
    elif task == "text_to_image":
        t_pos[:, modality_ids.eq(1)] = 1.0
    elif task != "joint":
        raise ValueError(f"unknown task: {task}")

    keep = torch.rand(batch, seq_len, device=device).lt(t_pos).logical_or(~valid_mask)
    noisy = targets.clone()
    random_text = torch.randint(0, qwen_vocab_size, (batch, seq_len), device=device)
    random_image = torch.randint(0, codebook_size, (batch, seq_len), device=device) + qwen_vocab_size
    random_tokens = torch.where(modality_ids.eq(0).unsqueeze(0), random_image, random_text)
    random_tokens = torch.where(valid_mask, random_tokens, torch.full_like(random_tokens, text_pad_id))
    noisy = torch.where(keep, noisy, random_tokens)
    return noisy, t_pos


def _position_masked_logits(logits: torch.Tensor, modality_ids: torch.Tensor, qwen_vocab_size: int) -> torch.Tensor:
    text_positions = modality_ids.eq(1)
    image_positions = modality_ids.eq(0)
    masked = logits.clone()
    masked[:, image_positions, :qwen_vocab_size] = float("-inf")
    masked[:, text_positions, qwen_vocab_size:] = float("-inf")
    return masked


def _loss_for_task(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    modality_ids: torch.Tensor,
    *,
    qwen_vocab_size: int,
    task: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    masked_logits = _position_masked_logits(logits, modality_ids, qwen_vocab_size)
    image_pos = modality_ids.eq(0).unsqueeze(0).expand_as(valid_mask)
    text_pos = modality_ids.eq(1).unsqueeze(0).expand_as(valid_mask)
    if task == "image_to_text":
        loss_mask = valid_mask.logical_and(text_pos)
    elif task == "text_to_image":
        loss_mask = valid_mask.logical_and(image_pos)
    else:
        loss_mask = valid_mask
    loss = F.cross_entropy(masked_logits[loss_mask], targets[loss_mask])
    parts = {
        "loss_positions": float(loss_mask.float().mean().item()),
    }
    if torch.any(valid_mask.logical_and(image_pos)):
        parts["image_loss"] = float(
            F.cross_entropy(
                masked_logits[valid_mask.logical_and(image_pos)],
                targets[valid_mask.logical_and(image_pos)],
            ).item()
        )
    if torch.any(valid_mask.logical_and(text_pos)):
        parts["text_loss"] = float(
            F.cross_entropy(
                masked_logits[valid_mask.logical_and(text_pos)],
                targets[valid_mask.logical_and(text_pos)],
            ).item()
        )
    return loss, parts


@torch.no_grad()
def evaluate_i2t(
    model: QwenFLMImageToText,
    loader: DataLoader,
    label_values: list[int],
    text_rows: list[torch.Tensor],
    pad_id: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    losses = []
    for batch_index, batch in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        targets, valid_mask, modality_ids = _build_targets(
            image_tokens,
            labels,
            qwen_vocab_size=model.qwen_vocab_size,
            label_values=label_values,
            text_rows=text_rows,
            pad_id=pad_id,
        )
        noisy, t_pos = _sample_noisy_tokens(
            targets,
            valid_mask,
            modality_ids,
            qwen_vocab_size=model.qwen_vocab_size,
            codebook_size=model.codebook_size,
            text_pad_id=pad_id,
            task="image_to_text",
        )
        logits = model(noisy, t_pos, modality_ids)
        loss, _ = _loss_for_task(
            logits,
            targets,
            valid_mask,
            modality_ids,
            qwen_vocab_size=model.qwen_vocab_size,
            task="image_to_text",
        )
        losses.append(float(loss.item()))
        text_logits = _position_masked_logits(logits, modality_ids, model.qwen_vocab_size)[:, modality_ids.eq(1), :]
        text_mask = valid_mask[:, modality_ids.eq(1)]
        scores = []
        for row in text_rows:
            length = row.numel()
            score = F.log_softmax(text_logits[:, :length, :], dim=-1).gather(
                -1,
                row.unsqueeze(0).unsqueeze(-1).expand(text_logits.shape[0], -1, 1),
            ).squeeze(-1).sum(dim=-1)
            scores.append(score)
        pred_indices = torch.stack(scores, dim=1).argmax(dim=1)
        pred_values = torch.tensor([label_values[index] for index in pred_indices.tolist()], device=device)
        correct += int(pred_values.eq(labels).sum().item())
        total += int(labels.numel())
    return {
        "loss": sum(losses) / max(len(losses), 1),
        "candidate_exact": correct / max(total, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Qwen-backed sparse categorical FLM for image-to-text.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="runs/qwen_flm_i2t")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-max-batches", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--adapter-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--freeze-qwen", action="store_true")
    parser.add_argument("--task", choices=["image_to_text", "joint"], default="image_to_text")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    set_seed(config.train.seed + 3031)
    device = resolve_device(config.train.device, config.train.gpu_index)

    tokenizer_state = load_tokenizer_state(config)
    text_metadata = metadata_from_state(tokenizer_state)
    label_values = list(text_metadata.label_values)
    label_strings = list(text_metadata.label_strings)
    codebook_size = int(tokenizer_state["codebook_size"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    text_rows = _label_text_rows(tokenizer, label_strings, device)
    model = QwenFLMImageToText(
        args.model_path,
        codebook_size,
        dtype=_dtype(args.dtype),
        freeze_qwen=args.freeze_qwen,
    ).to(device)
    if hasattr(model.qwen, "gradient_checkpointing_enable") and not args.freeze_qwen:
        model.qwen.gradient_checkpointing_enable()

    train_loader = DataLoader(
        TokenizedImageDataset(split_path(config, "train")),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        TokenizedImageDataset(split_path(config, "test")),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    adapter_params = (
        list(model.image_embed.parameters())
        + list(model.modality_embed.parameters())
        + list(model.time_mlp.parameters())
        + list(model.head.parameters())
    )
    adapter_ids = {id(param) for param in adapter_params}
    qwen_params = [param for param in model.parameters() if id(param) not in adapter_ids and param.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": args.adapter_lr},
            {"params": qwen_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "model_path": args.model_path,
                "task": args.task,
                "qwen_vocab_size": model.qwen_vocab_size,
                "codebook_size": codebook_size,
                "image_token_offset": model.qwen_vocab_size,
                "label_strings": label_strings,
                "objective": "sparse categorical FLM: corrupt tokens at random t, recover clean unified tokens",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    iterator = iter(train_loader)
    for step in tqdm(range(1, args.steps + 1), desc="qwen-flm-i2t"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        targets, valid_mask, modality_ids = _build_targets(
            image_tokens,
            labels,
            qwen_vocab_size=model.qwen_vocab_size,
            label_values=label_values,
            text_rows=text_rows,
            pad_id=pad_id,
        )
        noisy, t_pos = _sample_noisy_tokens(
            targets,
            valid_mask,
            modality_ids,
            qwen_vocab_size=model.qwen_vocab_size,
            codebook_size=model.codebook_size,
            text_pad_id=pad_id,
            task=args.task,
        )
        logits = model(noisy, t_pos, modality_ids)
        loss, parts = _loss_for_task(
            logits,
            targets,
            valid_mask,
            modality_ids,
            qwen_vocab_size=model.qwen_vocab_size,
            task=args.task,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            with (out_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "loss": float(loss.item()), **parts}) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_i2t(
                model,
                test_loader,
                label_values,
                text_rows,
                pad_id,
                device,
                args.eval_max_batches,
            )
            with (out_dir / "eval.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **metrics}) + "\n")
            torch.save(
                {
                    "step": step,
                    "image_embed": model.image_embed.state_dict(),
                    "modality_embed": model.modality_embed.state_dict(),
                    "time_mlp": model.time_mlp.state_dict(),
                    "head": model.head.state_dict(),
                    "qwen": model.qwen.state_dict() if not args.freeze_qwen else None,
                    "metrics": metrics,
                },
                out_dir / "latest.pt",
            )
            model.train()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
