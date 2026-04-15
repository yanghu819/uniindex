from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TextMetadata:
    kind: str
    pad_token: str
    pad_id: int
    vocab_tokens: tuple[str, ...]
    seq_len: int
    label_values: tuple[int, ...]
    label_strings: tuple[str, ...]
    label_text_tokens: torch.Tensor

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_tokens)


def build_text_metadata(kind: str, label_values: list[int], strings: list[str], pad_token: str) -> TextMetadata:
    if kind != "char":
        raise ValueError(f"unsupported text kind: {kind}")
    if len(label_values) != len(strings):
        raise ValueError("labels.values and text.strings must have the same length")
    if len(set(label_values)) != len(label_values):
        raise ValueError("labels.values must be unique")
    if len(set(strings)) != len(strings):
        raise ValueError("text.strings must be unique")

    charset = sorted({char for text in strings for char in text})
    vocab_tokens = (pad_token, *charset)
    token_to_id = {token: index for index, token in enumerate(vocab_tokens)}
    seq_len = max(len(text) for text in strings)

    rows = []
    for text in strings:
        token_ids = [token_to_id[char] for char in text]
        token_ids += [token_to_id[pad_token]] * (seq_len - len(token_ids))
        rows.append(token_ids)

    return TextMetadata(
        kind=kind,
        pad_token=pad_token,
        pad_id=token_to_id[pad_token],
        vocab_tokens=vocab_tokens,
        seq_len=seq_len,
        label_values=tuple(int(value) for value in label_values),
        label_strings=tuple(strings),
        label_text_tokens=torch.tensor(rows, dtype=torch.long),
    )


def metadata_from_state(state: dict) -> TextMetadata:
    return TextMetadata(
        kind=str(state["text_kind"]),
        pad_token=str(state["text_pad_token"]),
        pad_id=int(state["text_pad_id"]),
        vocab_tokens=tuple(state["text_vocab_tokens"]),
        seq_len=int(state["text_seq_len"]),
        label_values=tuple(int(value) for value in torch.as_tensor(state["label_values"]).tolist()),
        label_strings=tuple(state["text_strings"]),
        label_text_tokens=torch.as_tensor(state["label_text_tokens"], dtype=torch.long),
    )


def text_state_dict(metadata: TextMetadata) -> dict:
    return {
        "text_kind": metadata.kind,
        "text_pad_token": metadata.pad_token,
        "text_pad_id": metadata.pad_id,
        "text_vocab_tokens": list(metadata.vocab_tokens),
        "text_seq_len": metadata.seq_len,
        "text_strings": list(metadata.label_strings),
        "label_text_tokens": metadata.label_text_tokens.clone(),
    }


def encode_labels(labels: torch.Tensor, metadata: TextMetadata) -> torch.Tensor:
    labels = labels.long()
    value_to_row = {value: index for index, value in enumerate(metadata.label_values)}
    indices = [value_to_row[int(value)] for value in labels.detach().cpu().tolist()]
    index_tensor = torch.tensor(indices, dtype=torch.long, device=labels.device)
    return metadata.label_text_tokens.to(labels.device).index_select(0, index_tensor)


def decode_text_tokens(tokens: torch.Tensor, metadata: TextMetadata) -> list[str]:
    rows = torch.as_tensor(tokens, dtype=torch.long).cpu()
    outputs: list[str] = []
    for row in rows:
        chars = []
        for token_id in row.tolist():
            if token_id == metadata.pad_id:
                break
            chars.append(metadata.vocab_tokens[token_id])
        outputs.append("".join(chars))
    return outputs


def label_values_from_text_tokens(tokens: torch.Tensor, metadata: TextMetadata, unknown_value: int = -1) -> torch.Tensor:
    rows = torch.as_tensor(tokens, dtype=torch.long)
    reference = metadata.label_text_tokens.to(rows.device)
    matches = rows.unsqueeze(1).eq(reference.unsqueeze(0)).all(dim=-1)
    any_match = matches.any(dim=1)
    indices = matches.float().argmax(dim=1)
    label_values = torch.tensor(metadata.label_values, dtype=torch.long, device=rows.device)
    resolved = label_values.index_select(0, indices.clamp_max(len(metadata.label_values) - 1))
    return torch.where(any_match, resolved, torch.full_like(resolved, unknown_value))


def shifted_label_text_tokens(metadata: TextMetadata, token_offset: int = 0) -> torch.Tensor:
    return metadata.label_text_tokens + int(token_offset)


def sequence_candidate_scores(logits: torch.Tensor, candidate_tokens: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"expected logits with shape (batch, seq, vocab), got {tuple(logits.shape)}")
    if candidate_tokens.dim() != 2:
        raise ValueError(
            f"expected candidate_tokens with shape (num_candidates, seq), got {tuple(candidate_tokens.shape)}"
        )

    batch, seq_len, vocab_size = logits.shape
    num_candidates, candidate_seq_len = candidate_tokens.shape
    if seq_len != candidate_seq_len:
        raise ValueError(f"logit seq_len {seq_len} does not match candidate seq_len {candidate_seq_len}")

    targets = candidate_tokens.to(logits.device, dtype=torch.long)
    if torch.any(targets.lt(0)) or torch.any(targets.ge(vocab_size)):
        raise ValueError("candidate token ids must stay within the logit vocabulary range")

    log_probs = F.log_softmax(logits, dim=-1)
    expanded_log_probs = log_probs.unsqueeze(1).expand(batch, num_candidates, seq_len, vocab_size)
    gathered = expanded_log_probs.gather(
        dim=-1,
        index=targets.unsqueeze(0).unsqueeze(-1).expand(batch, num_candidates, seq_len, 1),
    ).squeeze(-1)
    return gathered.sum(dim=-1)
