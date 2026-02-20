"""Transformer encoder with pre-norm architecture for time-series."""

from __future__ import annotations

import torch
import torch.nn as nn

from quant_lab.models.transformer.attention import MultiHeadSelfAttention, PositionalEncoding


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer.

    Pre-norm (norm before attention/FFN) is more stable for training
    than post-norm, especially with deeper models.

    Architecture: x -> LN -> Attn -> +residual -> LN -> FFN -> +residual
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (batch, seq_len, d_model)."""
        # Pre-norm attention block
        x = x + self.dropout1(self.attn(self.norm1(x)))
        # Pre-norm FFN block
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Full transformer encoder: input projection + positional encoding + N layers.

    Takes raw feature sequences and produces contextualized representations.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Project raw features to model dimension
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # Learnable [CLS] token for sequence-level representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])

        # Final normalization (needed for pre-norm architecture)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a feature sequence.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            (batch, seq_len + 1, d_model) - includes CLS token at position 0
        """
        B = x.size(0)

        # Project features to d_model
        x = self.input_proj(x)
        x = self.pos_encoding(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1 + seq_len, d_model)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)
