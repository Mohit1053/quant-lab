"""Volatility prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolatilityHead(nn.Module):
    """Predict positive volatility scalar via Softplus activation.

    Target: absolute return |r| as a proxy for daily volatility.
    Loss: MSE.
    """

    def __init__(self, d_input: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.net = nn.Linear(d_input, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict volatility (always positive via softplus).

        Args:
            x: (batch, d_input)

        Returns:
            (batch,) positive volatility values
        """
        out = self.net(x).squeeze(-1)  # (batch,)
        return F.softplus(out)
