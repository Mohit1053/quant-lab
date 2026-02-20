"""Forecast decoder: pools encoder output into a fixed-size representation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForecastDecoder(nn.Module):
    """Extract a fixed-size representation from encoder output for prediction heads.

    Supports three pooling strategies:
    - cls: Use the [CLS] token output (position 0)
    - last: Use the last time-step output
    - mean: Mean pool over all time-step positions (excluding CLS)
    """

    def __init__(
        self,
        d_model: int,
        pooling: str = "cls",
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.pooling = pooling
        assert pooling in ("cls", "last", "mean"), f"Unknown pooling: {pooling}"

        # Optional projection layer between encoder and heads
        if projection_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(d_model, projection_dim),
                nn.GELU(),
                nn.LayerNorm(projection_dim),
            )
            self.output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = d_model

    def forward(self, encoder_output: torch.Tensor) -> torch.Tensor:
        """Pool encoder output to a single vector per sample.

        Args:
            encoder_output: (batch, seq_len + 1, d_model) with CLS at position 0

        Returns:
            (batch, output_dim)
        """
        if self.pooling == "cls":
            pooled = encoder_output[:, 0]  # CLS token
        elif self.pooling == "last":
            pooled = encoder_output[:, -1]  # Last time step
        elif self.pooling == "mean":
            pooled = encoder_output[:, 1:].mean(dim=1)  # Mean over time steps (skip CLS)

        return self.projection(pooled)
