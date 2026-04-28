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


def _mix_weight(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        return t[:, None, None]
    if t.dim() == 2:
        return t.unsqueeze(-1)
    raise ValueError(f"expected t to have rank 1 or 2, got {t.dim()}")


def mix_flm_noise(x1: torch.Tensor, t: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    weight = _mix_weight(t)
    noise = sample_masked_noise(x1, valid_token_mask)
    return (1.0 - weight) * noise + weight * x1


def simplex_uniform_base(x1: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    if valid_token_mask is None:
        return torch.full_like(x1, 1.0 / x1.shape[-1])
    mask = valid_token_mask.to(device=x1.device, dtype=x1.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return mask / denom


def mix_flm_simplex(x1: torch.Tensor, t: torch.Tensor, valid_token_mask: torch.Tensor | None = None) -> torch.Tensor:
    weight = _mix_weight(t)
    base = simplex_uniform_base(x1, valid_token_mask)
    return (1.0 - weight) * base + weight * x1


def mix_flm_state(
    x1: torch.Tensor,
    t: torch.Tensor,
    *,
    path: str = "gaussian",
    valid_token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if path == "gaussian":
        return mix_flm_noise(x1, t, valid_token_mask)
    if path == "simplex":
        return mix_flm_simplex(x1, t, valid_token_mask)
    raise ValueError(f"unsupported FLM state path: {path}")


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
