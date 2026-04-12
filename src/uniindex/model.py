from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    scale = math.log(10000) / max(half - 1, 1)
    frequencies = torch.exp(torch.arange(half, device=t.device) * -scale)
    angles = t[:, None] * frequencies[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


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
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
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
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = z_t.shape
        if seq_len != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {seq_len}")
        h = self.input_proj(z_t)
        positions = torch.arange(seq_len, device=z_t.device)
        h = h + self.pos_embed(positions).unsqueeze(0)
        h = h + self.modality_embed(modality_ids.to(z_t.device)).unsqueeze(0)
        h = h + self.time_proj(sinusoidal_time_embedding(t, h.shape[-1])).unsqueeze(1)
        h = self.transformer(h)
        h = self.norm(h)
        return self.head(h)
