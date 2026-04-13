from __future__ import annotations

import torch
import torch.nn.functional as F


def build_flm_clean_state(targets: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return F.one_hot(targets.long(), num_classes=vocab_size).float()


def mix_flm_noise(x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 1:
        weight = t[:, None, None]
    elif t.dim() == 2:
        weight = t.unsqueeze(-1)
    else:
        raise ValueError(f"expected t to have rank 1 or 2, got {t.dim()}")
    noise = torch.randn_like(x1)
    return (1.0 - weight) * noise + weight * x1


def apply_time_schedule(
    progress: torch.Tensor,
    modality_ids: torch.Tensor,
    image_time_power: float,
    label_time_power: float,
) -> torch.Tensor:
    if progress.dim() != 1:
        raise ValueError(f"expected progress to have shape (batch,), got {tuple(progress.shape)}")
    powers = torch.where(
        modality_ids.long() == 0,
        torch.full_like(modality_ids, float(image_time_power), dtype=torch.float32),
        torch.full_like(modality_ids, float(label_time_power), dtype=torch.float32),
    ).to(progress.device)
    return progress[:, None].pow(powers.unsqueeze(0))


def restore_image_tokens(tokens: torch.Tensor, tokenizer_state: dict) -> torch.Tensor:
    original_token_ids = tokenizer_state.get("original_token_ids")
    if original_token_ids is None:
        return tokens.long()
    original_token_ids = torch.as_tensor(original_token_ids, dtype=torch.long)
    return original_token_ids[tokens.long()]
