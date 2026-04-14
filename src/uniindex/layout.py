from __future__ import annotations

import torch


def label_offset(codebook_size: int) -> int:
    return codebook_size


def unified_vocab_size(codebook_size: int, num_labels: int) -> int:
    return codebook_size + num_labels


def position_modalities(image_seq_len: int, label_seq_len: int = 1) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(image_seq_len, dtype=torch.long),
            torch.ones(label_seq_len, dtype=torch.long),
        ],
        dim=0,
    )


def modality_vocab_mask(codebook_size: int, num_labels: int) -> torch.Tensor:
    vocab_size = unified_vocab_size(codebook_size, num_labels)
    masks = torch.zeros(2, vocab_size, dtype=torch.bool)
    masks[0, :codebook_size] = True
    masks[1, codebook_size:] = True
    return masks


def position_valid_token_mask(image_seq_len: int, codebook_size: int, num_labels: int) -> torch.Tensor:
    position_types = position_modalities(image_seq_len)
    return modality_vocab_mask(codebook_size, num_labels)[position_types]


def unified_targets(image_tokens: torch.Tensor, labels: torch.Tensor, codebook_size: int) -> torch.Tensor:
    label_tokens = labels.unsqueeze(1) + codebook_size
    return torch.cat([image_tokens, label_tokens], dim=1)


def mask_logits(logits: torch.Tensor, image_seq_len: int, codebook_size: int, num_labels: int) -> torch.Tensor:
    position_types = position_modalities(image_seq_len).to(logits.device)
    valid = modality_vocab_mask(codebook_size, num_labels).to(logits.device)[position_types]
    return logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
