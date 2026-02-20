"""Gated Residual Network (GRN) - core building block of TFT."""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit: sigmoid(Wx + b) * (Vx + c)."""

    def __init__(self, d_model: int, d_output: int | None = None):
        super().__init__()
        d_output = d_output or d_model
        self.fc = nn.Linear(d_model, d_output)
        self.gate = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.fc(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from the TFT paper.

    Architecture:
        eta_1 = ELU(W_1 * x + b_1)           (or ELU(W_1 * x + W_c * c + b_1) with context)
        eta_2 = W_2 * eta_1 + b_2
        GLU(eta_2) + residual -> LayerNorm

    Args:
        d_model: Input/output dimension.
        d_hidden: Hidden layer dimension.
        d_context: Optional context vector dimension (e.g., from static enrichment).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int | None = None,
        d_context: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_hidden = d_hidden or d_model

        self.fc1 = nn.Linear(d_model, d_hidden)
        self.context_proj = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (..., d_model) input tensor.
            context: Optional (..., d_context) context vector.

        Returns:
            (..., d_model) output tensor.
        """
        residual = x

        hidden = self.fc1(x)
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = torch.nn.functional.elu(hidden)

        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        hidden = self.glu(hidden)

        return self.layer_norm(hidden + residual)
