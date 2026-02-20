"""Multi-head self-attention with PyTorch 2.x SDPA (Flash Attention)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal encoding, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, d_model) for broadcasting
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding. x: (batch, seq_len, d_model)."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention using PyTorch 2.x scaled_dot_product_attention.

    Automatically uses Flash Attention, Memory-Efficient Attention, or
    math-based attention depending on hardware and input.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        # Single projection for Q, K, V (3x faster than separate projections)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            attn_mask: optional attention mask
            is_causal: if True, apply causal mask

        Returns:
            (batch, seq_len, d_model)
        """
        B, T, C = x.shape

        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, nhead, head_dim)

        # Transpose to (B, nhead, T, head_dim) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # PyTorch 2.x SDPA: automatically picks Flash/Memory-Efficient/Math backend
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        # Reshape back: (B, nhead, T, head_dim) -> (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_out)
