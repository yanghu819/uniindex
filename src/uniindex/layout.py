from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskLayout:
    image_seq_len: int
    text_seq_len: int
    codebook_size: int
    text_vocab_size: int

    @property
    def seq_len(self) -> int:
        return self.image_seq_len + self.text_seq_len

    @property
    def vocab_size(self) -> int:
        return unified_vocab_size(self.codebook_size, self.text_vocab_size)

    @property
    def image_slice(self) -> slice:
        return slice(0, self.image_seq_len)

    @property
    def text_slice(self) -> slice:
        return slice(self.image_seq_len, self.seq_len)

    @property
    def text_offset(self) -> int:
        return text_offset(self.codebook_size)

    @property
    def label_offset(self) -> int:
        return label_offset(self.codebook_size)

    def position_modalities(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.zeros(self.image_seq_len, dtype=torch.long),
                torch.ones(self.text_seq_len, dtype=torch.long),
            ],
            dim=0,
        )

    def modality_vocab_mask(self) -> torch.Tensor:
        masks = torch.zeros(2, self.vocab_size, dtype=torch.bool)
        masks[0, : self.codebook_size] = True
        masks[1, self.codebook_size :] = True
        return masks

    def position_valid_token_mask(self) -> torch.Tensor:
        return self.modality_vocab_mask()[self.position_modalities()]


def text_offset(codebook_size: int) -> int:
    return codebook_size


def label_offset(codebook_size: int) -> int:
    return text_offset(codebook_size)


def unified_vocab_size(codebook_size: int, text_vocab_size: int) -> int:
    return codebook_size + text_vocab_size


def unified_targets(image_tokens: torch.Tensor, text_tokens: torch.Tensor, codebook_size: int) -> torch.Tensor:
    shifted_text = text_tokens + codebook_size
    return torch.cat([image_tokens, shifted_text], dim=1)


def mask_logits(logits: torch.Tensor, *, layout: TaskLayout) -> torch.Tensor:
    position_types = layout.position_modalities().to(logits.device)
    valid = layout.modality_vocab_mask().to(logits.device)[position_types]
    return logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
