"""Distribution prediction head for return forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn


class GaussianHead(nn.Module):
    """Predict Gaussian distribution parameters (mean, log_variance) for returns.

    Loss: Gaussian negative log-likelihood.
    """

    def __init__(self, d_input: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),  # mean + log_var
            )
        else:
            self.net = nn.Linear(d_input, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict return distribution.

        Args:
            x: (batch, d_input)

        Returns:
            dict with 'mean' (batch,) and 'log_var' (batch,)
        """
        out = self.net(x)  # (batch, 2)
        return {
            "mean": out[:, 0],
            "log_var": out[:, 1],
        }


class StudentTHead(nn.Module):
    """Predict Student-t distribution parameters for heavy-tailed returns.

    Outputs: location (mu), log_scale, log_df (degrees of freedom).
    """

    def __init__(self, d_input: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 3),  # mu, log_scale, log_df
            )
        else:
            self.net = nn.Linear(d_input, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict return distribution.

        Args:
            x: (batch, d_input)

        Returns:
            dict with 'loc', 'log_scale', 'log_df'
        """
        out = self.net(x)
        return {
            "loc": out[:, 0],
            "log_scale": out[:, 1],
            "log_df": out[:, 2],
        }
