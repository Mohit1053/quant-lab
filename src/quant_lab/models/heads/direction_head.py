"""Direction prediction head for up/down/flat classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectionHead(nn.Module):
    """Classify return direction: down (0), flat (1), up (2).

    Loss: cross-entropy.
    """

    def __init__(self, d_input: int, num_classes: int = 3, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is not None:
            self.net = nn.Sequential(
                nn.Linear(d_input, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.net = nn.Linear(d_input, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict direction logits.

        Args:
            x: (batch, d_input)

        Returns:
            (batch, num_classes) logits
        """
        return self.net(x)
