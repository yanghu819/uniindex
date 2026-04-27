from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    original_shape = t.shape
    t = t.reshape(-1)
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
    angles = t[:, None] * frequencies[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb.reshape(*original_shape, dim)


class UnifiedDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_ratio: int,
        dropout: float,
        image_summary_to_text: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.image_summary_to_text = bool(image_summary_to_text)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.modality_embed = nn.Embedding(2, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        try:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        except TypeError:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        if self.image_summary_to_text:
            self.image_summary_proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.image_summary_proj = None
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _image_summary_gate(self, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            clean = t
            noisy = t
        elif t.dim() == 2:
            modality_ids = modality_ids.to(t.device)
            image_mask = modality_ids.eq(0)
            text_mask = modality_ids.eq(1)
            if not bool(image_mask.any().item()) or not bool(text_mask.any().item()):
                return torch.zeros(t.shape[0], device=t.device, dtype=t.dtype)
            clean = t[:, image_mask].mean(dim=1)
            noisy = t[:, text_mask].mean(dim=1)
        else:
            raise ValueError(f"expected t to have rank 1 or 2, got {t.dim()}")
        return (clean.clamp(0.0, 1.0) * (1.0 - noisy.clamp(0.0, 1.0))).clamp(0.0, 1.0)

    def _inject_image_summary_to_text(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        modality_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.image_summary_proj is None:
            return h
        modality_ids = modality_ids.to(h.device)
        image_mask = modality_ids.eq(0)
        text_mask = modality_ids.eq(1)
        if not bool(image_mask.any().item()) or not bool(text_mask.any().item()):
            return h
        image_mask_float = image_mask.to(dtype=h.dtype, device=h.device).view(1, h.shape[1], 1)
        image_count = image_mask_float.sum(dim=1).clamp_min(1.0)
        image_summary = (h * image_mask_float).sum(dim=1) / image_count
        summary_delta = self.image_summary_proj(image_summary)
        gate = self._image_summary_gate(t, modality_ids).to(device=h.device, dtype=h.dtype)
        text_mask_float = text_mask.to(dtype=h.dtype, device=h.device).view(1, h.shape[1], 1)
        return h + text_mask_float * gate.view(-1, 1, 1) * summary_delta.unsqueeze(1)

    def forward_features(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = z_t.shape
        if seq_len != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {seq_len}")
        h = self.input_proj(z_t)
        positions = torch.arange(seq_len, device=z_t.device)
        h = h + self.pos_embed(positions).unsqueeze(0)
        h = h + self.modality_embed(modality_ids.to(z_t.device)).unsqueeze(0)
        time_emb = self.time_proj(sinusoidal_time_embedding(t, h.shape[-1]).to(h.dtype))
        if t.dim() == 1:
            h = h + time_emb.unsqueeze(1)
        elif t.dim() == 2:
            h = h + time_emb
        else:
            raise ValueError(f"expected t to have rank 1 or 2, got {t.dim()}")
        h = self._inject_image_summary_to_text(h, t, modality_ids)
        h = self.transformer(h)
        return self.norm(h)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(z_t, t, modality_ids))
