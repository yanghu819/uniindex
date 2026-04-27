from __future__ import annotations

import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import ProjectConfig
from .data import build_loader, load_tokenizer_state, split_path
from .label_feature_probe import labels_to_class_indices
from .runtime import RunContext, append_jsonl, ensure_project_dirs, resolve_device, set_seed
from .text import TextMetadata, decode_text_tokens, metadata_from_state, text_scoring_mask


class VQTextDecoder(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int,
        image_seq_len: int,
        text_seq_len: int,
        text_vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(codebook_size, d_model)
        self.image_pos_embed = nn.Embedding(image_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.context_norm = nn.LayerNorm(d_model)
        self.text_pos_embed = nn.Embedding(text_seq_len, d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, text_vocab_size),
        )

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        image_positions = torch.arange(image_tokens.shape[1], device=image_tokens.device)
        hidden = self.token_embed(image_tokens) + self.image_pos_embed(image_positions).unsqueeze(0)
        hidden = self.encoder(hidden)
        context = self.context_norm(hidden.mean(dim=1))
        text_positions = torch.arange(self.text_pos_embed.num_embeddings, device=image_tokens.device)
        text_hidden = context.unsqueeze(1) + self.text_pos_embed(text_positions).unsqueeze(0)
        return self.head(text_hidden)


def _compatible_heads(d_model: int, requested_heads: int) -> int:
    for heads in range(min(d_model, requested_heads), 0, -1):
        if d_model % heads == 0:
            return heads
    return 1


def text_sequence_loss(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> torch.Tensor:
    if logits.shape[:2] != targets.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not match targets shape {tuple(targets.shape)}")
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=int(pad_id))


def masked_candidate_scores(logits: torch.Tensor, metadata: TextMetadata) -> torch.Tensor:
    candidates = metadata.label_text_tokens.to(logits.device, dtype=torch.long)
    if logits.shape[1] != candidates.shape[1]:
        raise ValueError(f"logits text length {logits.shape[1]} does not match candidate length {candidates.shape[1]}")
    log_probs = F.log_softmax(logits, dim=-1)
    expanded = log_probs.unsqueeze(1).expand(-1, candidates.shape[0], -1, -1)
    gathered = expanded.gather(
        dim=-1,
        index=candidates.unsqueeze(0).unsqueeze(-1).expand(logits.shape[0], -1, -1, 1),
    ).squeeze(-1)
    valid = candidates.ne(metadata.pad_id)
    lengths = valid.sum(dim=1).clamp_min(1).to(logits.dtype)
    return gathered.masked_fill(~valid.unsqueeze(0), 0.0).sum(dim=-1) / lengths.unsqueeze(0)


def _decode_exact(logits: torch.Tensor, text_tokens: torch.Tensor, metadata: TextMetadata) -> int:
    predicted = logits.argmax(dim=-1)
    pred_strings = decode_text_tokens(predicted, metadata)
    target_strings = decode_text_tokens(text_tokens, metadata)
    return sum(pred == target for pred, target in zip(pred_strings, target_strings))


@torch.no_grad()
def _evaluate(
    *,
    config: ProjectConfig,
    decoder: VQTextDecoder,
    metadata: TextMetadata,
    label_values: torch.Tensor,
    split: str,
) -> dict[str, float]:
    device = label_values.device
    loader = build_loader(
        split_path(config, split),
        batch_size=config.train.eval_batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
    )
    totals = {
        "free_exact": 0,
        "candidate_correct": 0,
        "shuffled_candidate_correct": 0,
        "token_correct": 0,
        "token_total": 0,
        "total": 0,
    }
    for batch in tqdm(loader, desc=f"probe-vq-text-decoder/{split}"):
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)
        logits = decoder(image_tokens)
        scores = masked_candidate_scores(logits, metadata)
        predicted_labels = label_values.index_select(0, scores.argmax(dim=1))
        totals["candidate_correct"] += int(predicted_labels.eq(labels).sum().item())
        totals["free_exact"] += int(_decode_exact(logits, text_tokens, metadata))
        predicted_tokens = logits.argmax(dim=-1)
        token_mask = text_scoring_mask(text_tokens, metadata, include_bos=False, include_eos=True).to(device)
        totals["token_correct"] += int(predicted_tokens.eq(text_tokens).logical_and(token_mask).sum().item())
        totals["token_total"] += int(token_mask.sum().item())

        shuffled_indices = torch.arange(image_tokens.shape[0], device=device).roll(1)
        shuffled_logits = decoder(image_tokens.index_select(0, shuffled_indices))
        shuffled_scores = masked_candidate_scores(shuffled_logits, metadata)
        shuffled_predicted = label_values.index_select(0, shuffled_scores.argmax(dim=1))
        totals["shuffled_candidate_correct"] += int(shuffled_predicted.eq(labels).sum().item())
        totals["total"] += int(labels.shape[0])

    total = max(totals["total"], 1)
    return {
        f"{split}_free_exact": totals["free_exact"] / total,
        f"{split}_candidate_accuracy": totals["candidate_correct"] / total,
        f"{split}_shuffled_candidate_accuracy": totals["shuffled_candidate_correct"] / total,
        f"{split}_image_dependence_margin": (totals["candidate_correct"] - totals["shuffled_candidate_correct"]) / total,
        f"{split}_token_accuracy": totals["token_correct"] / max(totals["token_total"], 1),
        f"{split}_total": float(totals["total"]),
    }


def _contrast_loss(
    *,
    true_logits: torch.Tensor,
    shuffled_logits: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    metadata: TextMetadata,
    margin: float,
) -> torch.Tensor:
    target_rows = labels_to_class_indices(labels, label_values)
    true_scores = masked_candidate_scores(true_logits, metadata).gather(1, target_rows[:, None]).squeeze(1)
    shuffled_scores = masked_candidate_scores(shuffled_logits, metadata).gather(1, target_rows[:, None]).squeeze(1)
    return F.relu(float(margin) - true_scores + shuffled_scores).mean()


def _infinite(loader):
    while True:
        yield from loader


def run_vq_text_decoder_probe(
    *,
    config: ProjectConfig,
    steps: int = 200,
    eval_every: int = 50,
    d_model: int | None = None,
    n_layers: int = 2,
    lr: float | None = None,
    contrast_weight: float = 0.0,
    contrast_margin: float = 1.0,
    run_context: RunContext | None = None,
) -> dict:
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if eval_every < 1:
        raise ValueError(f"eval_every must be >= 1, got {eval_every}")
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if contrast_weight < 0.0:
        raise ValueError(f"contrast_weight must be >= 0, got {contrast_weight}")
    if contrast_margin <= 0.0:
        raise ValueError(f"contrast_margin must be > 0, got {contrast_margin}")

    ensure_project_dirs(config)
    set_seed(config.train.seed)
    device = resolve_device(config.train.device, config.train.gpu_index)
    run_context = run_context or RunContext(config, "probe-vq-text-decoder")
    run_context.set_device(device)
    tokenizer_state = load_tokenizer_state(config)
    metadata = metadata_from_state(tokenizer_state)
    label_values = torch.tensor(metadata.label_values, dtype=torch.long, device=device)
    model_dim = int(config.model.d_model if d_model is None else d_model)
    decoder = VQTextDecoder(
        codebook_size=int(tokenizer_state["codebook_size"]),
        image_seq_len=int(tokenizer_state["image_seq_len"]),
        text_seq_len=int(metadata.seq_len),
        text_vocab_size=int(metadata.vocab_size),
        d_model=model_dim,
        n_heads=_compatible_heads(model_dim, config.model.n_heads),
        n_layers=int(n_layers),
        dropout=config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(config.i2t_llm.lr if lr is None else lr),
        weight_decay=config.train.weight_decay,
    )
    train_loader = build_loader(
        split_path(config, "train"),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
    )
    train_iter = _infinite(train_loader)
    train_log = run_context.log_path("vq_text_decoder_train.jsonl")
    eval_log = run_context.log_path("vq_text_decoder_eval.jsonl")
    eval_steps = {0, steps}
    eval_steps.update(range(eval_every, steps + 1, eval_every))
    snapshots: list[dict] = []
    summary = {
        "source": "probe-vq-text-decoder",
        "tokenizer_kind": config.tokenizer.kind,
        "tokenizer_model_name": config.tokenizer.model_name,
        "image_size": config.tokenizer.image_size,
        "train_limit": config.dataset.train_limit,
        "test_limit": config.dataset.test_limit,
        "steps": int(steps),
        "eval_every": int(eval_every),
        "d_model": model_dim,
        "n_heads": _compatible_heads(model_dim, config.model.n_heads),
        "n_layers": int(n_layers),
        "lr": float(config.i2t_llm.lr if lr is None else lr),
        "contrast_weight": float(contrast_weight),
        "contrast_margin": float(contrast_margin),
        "snapshots": snapshots,
    }

    for step in range(steps + 1):
        if step in eval_steps:
            decoder.eval()
            metrics = {"step": step, **_evaluate(config=config, decoder=decoder, metadata=metadata, label_values=label_values, split="test")}
            snapshots.append(metrics)
            append_jsonl(eval_log, metrics)
            decoder.train()
        if step == steps:
            break

        batch = next(train_iter)
        image_tokens = batch["image_tokens"].to(device)
        text_tokens = batch["text_tokens"].to(device)
        labels = batch["label"].to(device)
        logits = decoder(image_tokens)
        seq_loss = text_sequence_loss(logits, text_tokens, metadata.pad_id)
        loss = seq_loss
        record = {"step": step + 1, "sequence_loss": float(seq_loss.detach().cpu().item())}
        if contrast_weight > 0.0:
            shuffled_indices = torch.arange(image_tokens.shape[0], device=device).roll(1)
            shuffled_logits = decoder(image_tokens.index_select(0, shuffled_indices))
            contrast = _contrast_loss(
                true_logits=logits,
                shuffled_logits=shuffled_logits,
                labels=labels,
                label_values=label_values,
                metadata=metadata,
                margin=contrast_margin,
            )
            loss = loss + float(contrast_weight) * contrast
            record["contrast_loss"] = float(contrast.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=config.train.grad_clip_norm)
        optimizer.step()
        if (step + 1) % config.train.log_every == 0 or step == 0:
            record["loss"] = float(loss.detach().cpu().item())
            append_jsonl(train_log, record)

    summary["final"] = snapshots[-1] if snapshots else {}
    summary_path = run_context.log_path("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "summary": summary,
            "tokenizer_state": tokenizer_state,
        },
        run_context.log_path("decoder.pt"),
    )
    return summary
