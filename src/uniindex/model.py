from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def _rope_frequencies(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = positions[:, None] * inv_freq[None, :]
    return angles.cos().to(dtype=dtype), angles.sin().to(dtype=dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = _rope_frequencies(seq_len, self.head_dim, x.device, q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attended)


class RoPETransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPESelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attn(self.norm1(x)))
        return x + self.dropout2(self.mlp(self.norm2(x)))


class UnifiedDenoiser(nn.Module):
    """Clean shared FLM denoiser: one sequence, one backbone, one head."""

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
        position_encoding: str = "learned",
    ) -> None:
        super().__init__()
        if position_encoding not in {"learned", "rope"}:
            raise ValueError(f"unsupported position_encoding: {position_encoding}")
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.position_encoding = position_encoding
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model) if position_encoding == "learned" else None
        self.modality_embed = nn.Embedding(2, d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        if position_encoding == "rope":
            self.transformer = nn.Sequential(
                *[
                    RoPETransformerBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
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

    def forward_features(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = z_t.shape
        del batch
        if seq_len != self.seq_len:
            raise ValueError(f"expected sequence length {self.seq_len}, got {seq_len}")
        h = self.input_proj(z_t)
        if self.pos_embed is not None:
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
        h = self.transformer(h)
        return self.norm(h)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, modality_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(z_t, t, modality_ids))
