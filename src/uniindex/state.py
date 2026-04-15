from __future__ import annotations

import torch
import torch.nn.functional as F


def build_flm_clean_state(targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(targets.long(), num_classes=vocab_size).float()


def sample_masked_noise(x1: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    noise = torch.randn_like(x1)
    if valid_token_mask is None:
        return noise
    mask = valid_token_mask.to(device=x1.device, dtype=x1.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    noise = noise * mask
    return noise


def mix_flm_noise(x1: torch.Tensor, t: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    if t.dim() == 1:
        weight = t[:, None, None]
    elif t.dim() == 2:
        weight = t.unsqueeze(-1)
    else:
        raise ValueError(f"expected t to have rank 1 or 2, got {t.dim()}")
    noise = sample_masked_noise(x1, valid_token_mask)
    return (1.0 - weight) * noise + weight * x1


def condition_clean_timesteps(
    t_pos: torch.Tensor,
    image_seq_len: int,
    *,
    condition_image: bool,
    condition_text: bool,
) -> torch.Tensor:
    adjusted = t_pos.clone()
    if condition_image:
        adjusted[:, :image_seq_len] = 1.0
    if condition_text:
        adjusted[:, image_seq_len:] = 1.0
    return adjusted


def restore_image_tokens(tokens: torch.Tensor, tokenizer_state: dict) -> torch.Tensor:
    original_token_ids = tokenizer_state.get("original_token_ids")
    if original_token_ids is None:
        return tokens.long()
    original_token_ids = torch.as_tensor(original_token_ids, dtype=torch.long)
    return original_token_ids[tokens.long()]
