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


class QwenImageToText(nn.Module):
    def __init__(
        self,
        model_path: str,
        codebook_size: int,
        *,
        dtype: torch.dtype,
        freeze_qwen: bool,
    ) -> None:
        super().__init__()
        self.qwen = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=dtype,
        )
        hidden_size = int(self.qwen.config.hidden_size)
        self.image_embed = nn.Embedding(codebook_size, hidden_size)
        self.image_norm = nn.LayerNorm(hidden_size)
        if freeze_qwen:
            for param in self.qwen.parameters():
                param.requires_grad_(False)

    def token_embedding(self) -> nn.Module:
        return self.qwen.get_input_embeddings()

    def forward(
        self,
        image_tokens: torch.Tensor,
        prompt_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embed = self.token_embedding()
        prompt_embeds = token_embed(prompt_ids)
        image_embeds = self.image_norm(self.image_embed(image_tokens)).to(prompt_embeds.dtype)
        answer_inputs = answer_ids.masked_fill(~answer_mask, 0)
        answer_embeds = token_embed(answer_inputs)
        inputs_embeds = torch.cat([image_embeds, prompt_embeds, answer_embeds], dim=1)

        ignore = torch.full(
            (image_tokens.shape[0], image_tokens.shape[1] + prompt_ids.shape[1]),
            -100,
            dtype=torch.long,
            device=image_tokens.device,
        )
        labels = torch.cat([ignore, answer_ids.masked_fill(~answer_mask, -100)], dim=1)
        return self.qwen(inputs_embeds=inputs_embeds, labels=labels).loss


def _dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _answer_rows(tokenizer, label_strings: list[str], device: torch.device) -> list[torch.Tensor]:
    eos = tokenizer.eos_token_id
    rows = []
    for text in label_strings:
        ids = tokenizer.encode(" " + text, add_special_tokens=False)
        if eos is not None:
            ids.append(eos)
        rows.append(torch.tensor(ids, dtype=torch.long, device=device))
    return rows


def _batch_answer_ids(labels: torch.Tensor, label_values: list[int], answer_rows: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    value_to_index = {value: index for index, value in enumerate(label_values)}
    rows = [answer_rows[value_to_index[int(label.item())]] for label in labels]
    max_len = max(row.numel() for row in rows)
    ids = torch.zeros((len(rows), max_len), dtype=torch.long, device=labels.device)
    mask = torch.zeros((len(rows), max_len), dtype=torch.bool, device=labels.device)
    for index, row in enumerate(rows):
        ids[index, : row.numel()] = row
        mask[index, : row.numel()] = True
    return ids, mask


@torch.no_grad()
def evaluate(
    model: QwenImageToText,
    loader: DataLoader,
    tokenizer,
    label_values: list[int],
    label_strings: list[str],
    prompt_ids: torch.Tensor,
    answer_rows: list[torch.Tensor],
    device: torch.device,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    losses = []
    token_embed = model.token_embedding()

    for batch_index, batch in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
        if max_batches > 0 and batch_index > max_batches:
            break
        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        prompt = prompt_ids.expand(image_tokens.shape[0], -1)
        answer_ids, answer_mask = _batch_answer_ids(labels, label_values, answer_rows)
        losses.append(float(model(image_tokens, prompt, answer_ids, answer_mask).item()))

        scores = []
        prompt_embeds = token_embed(prompt)
        image_embeds = model.image_norm(model.image_embed(image_tokens)).to(prompt_embeds.dtype)
        for candidate in answer_rows:
            candidate = candidate.unsqueeze(0).expand(image_tokens.shape[0], -1)
            candidate_embeds = token_embed(candidate)
            inputs_embeds = torch.cat([image_embeds, prompt_embeds, candidate_embeds], dim=1)
            logits = model.qwen(inputs_embeds=inputs_embeds).logits
            prefix_len = image_embeds.shape[1] + prompt_embeds.shape[1]
            pred_logits = logits[:, prefix_len - 1 : prefix_len + candidate.shape[1] - 1]
            log_probs = F.log_softmax(pred_logits.float(), dim=-1)
            score = log_probs.gather(-1, candidate.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            scores.append(score)
        pred_indices = torch.stack(scores, dim=1).argmax(dim=1)
        pred_values = torch.tensor([label_values[i] for i in pred_indices.tolist()], device=device)
        correct += int(pred_values.eq(labels).sum().item())
        total += int(labels.numel())

    return {
        "loss": sum(losses) / max(len(losses), 1),
        "candidate_exact": correct / max(total, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 for image-token to label-text prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default="models/Qwen3-0.6B")
    parser.add_argument("--out-dir", default="runs/qwen_i2t")
    parser.add_argument("--prompt", default="The image is a handwritten digit. Answer with its English word:")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--image-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-max-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--freeze-qwen", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_project_dirs(config)
    set_seed(config.train.seed + 2026)
    device = resolve_device(config.train.device, config.train.gpu_index)
    dtype = _dtype(args.dtype)

    tokenizer_state = load_tokenizer_state(config)
    text_metadata = metadata_from_state(tokenizer_state)
    label_values = list(text_metadata.label_values)
    label_strings = list(text_metadata.label_strings)
    codebook_size = int(tokenizer_state["codebook_size"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids = torch.tensor(
        tokenizer.encode(args.prompt, add_special_tokens=False),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    answer_rows = _answer_rows(tokenizer, label_strings, device)

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

    model = QwenImageToText(
        args.model_path,
        codebook_size,
        dtype=dtype,
        freeze_qwen=args.freeze_qwen,
    ).to(device)
    model.train()
    if hasattr(model.qwen, "gradient_checkpointing_enable") and not args.freeze_qwen:
        model.qwen.gradient_checkpointing_enable()

    image_params = list(model.image_embed.parameters()) + list(model.image_norm.parameters())
    image_param_ids = {id(param) for param in image_params}
    qwen_params = [param for param in model.parameters() if id(param) not in image_param_ids and param.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": image_params, "lr": args.image_lr},
            {"params": qwen_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.jsonl"
    metadata = {
        "config": args.config,
        "model_path": args.model_path,
        "prompt": args.prompt,
        "codebook_size": codebook_size,
        "image_seq_len": int(tokenizer_state["image_seq_len"]),
        "label_strings": label_strings,
        "freeze_qwen": args.freeze_qwen,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    iterator = iter(train_loader)
    for step in tqdm(range(1, args.steps + 1), desc="qwen-i2t"):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        image_tokens = batch["image_tokens"].to(device)
        labels = batch["label"].to(device)
        prompt = prompt_ids.expand(image_tokens.shape[0], -1)
        answer_ids, answer_mask = _batch_answer_ids(labels, label_values, answer_rows)
        loss = model(image_tokens, prompt, answer_ids, answer_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            row = {"step": step, "loss": float(loss.item())}
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                test_loader,
                tokenizer,
                label_values,
                label_strings,
                prompt_ids,
                answer_rows,
                device,
                args.eval_max_batches,
            )
            row = {"step": step, **metrics}
            with (out_dir / "eval.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            model.train()
            torch.save(
                {
                    "step": step,
                    "image_embed": model.image_embed.state_dict(),
                    "image_norm": model.image_norm.state_dict(),
                    "qwen": model.qwen.state_dict() if not args.freeze_qwen else None,
                    "metadata": metadata,
                    "metrics": metrics,
                },
                out_dir / "latest.pt",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
