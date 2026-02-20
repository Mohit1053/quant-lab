"""Patch-based tokenization for time-series data."""

from __future__ import annotations

import torch
import torch.nn as nn

from quant_lab.models.transformer.attention import PositionalEncoding


class PatchTokenizer(nn.Module):
    """Divide time-series into non-overlapping patches and project to d_model.

    Converts (batch, seq_len, num_features) into (batch, num_patches, d_model)
    where num_patches = seq_len // patch_size.

    Each patch spans `patch_size` time steps across all features, creating
    a token that captures short-term local patterns.
    """

    def __init__(
        self,
        num_features: int,
        patch_size: int = 5,
        d_model: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.patch_size = patch_size
        self.d_model = d_model

        # Project flattened patch to d_model
        self.patch_proj = nn.Linear(patch_size * num_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1024, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tokenize input sequence into patches.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            (batch, num_patches, d_model)
        """
        B, T, F = x.shape
        assert F == self.num_features, f"Expected {self.num_features} features, got {F}"

        # Trim to be divisible by patch_size
        num_patches = T // self.patch_size
        x = x[:, : num_patches * self.patch_size, :]

        # Reshape into patches: (B, num_patches, patch_size, F) -> (B, num_patches, patch_size * F)
        x = x.reshape(B, num_patches, self.patch_size, F)
        x = x.reshape(B, num_patches, self.patch_size * F)

        # Project and add positional encoding
        tokens = self.patch_proj(x)
        tokens = self.layer_norm(tokens)
        tokens = self.pos_encoding(tokens)

        return tokens

    @property
    def num_patches(self) -> int:
        """Number of patches for a given sequence (set externally)."""
        return getattr(self, "_num_patches", 0)
