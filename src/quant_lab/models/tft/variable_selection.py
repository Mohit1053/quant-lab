"""Variable Selection Network for TFT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_lab.models.tft.gated_residual import GatedResidualNetwork


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network from the TFT paper.

    Learns which input features are most relevant via softmax-weighted
    combination of per-feature GRN outputs. Provides interpretable
    feature importance weights.

    Args:
        num_features: Number of input features.
        d_model: Model dimension (each feature is projected to this).
        d_hidden: Hidden dimension for GRN blocks.
        d_context: Optional context vector dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        d_hidden: int | None = None,
        d_context: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Per-feature input projection
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # Per-feature GRN transformations
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_hidden=d_hidden, dropout=dropout)
            for _ in range(num_features)
        ])

        # Variable selection weights: a GRN that produces softmax weights
        # Input: flattened projected features
        self.selection_grn = GatedResidualNetwork(
            num_features * d_model,
            d_hidden=d_hidden or d_model,
            d_context=d_context,
            dropout=dropout,
        )
        self.selection_head = nn.Linear(num_features * d_model, num_features)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (..., num_features) raw feature values.
            context: Optional context vector for selection weights.

        Returns:
            selected: (..., d_model) weighted combination of feature embeddings.
            weights: (..., num_features) softmax selection weights (interpretable).
        """
        # Project each feature to d_model
        # x shape: (..., num_features)
        batch_shape = x.shape[:-1]

        feature_embeddings = []
        for i in range(self.num_features):
            feat_i = x[..., i : i + 1]  # (..., 1)
            proj_i = self.feature_projections[i](feat_i)  # (..., d_model)
            transformed_i = self.feature_grns[i](proj_i)  # (..., d_model)
            feature_embeddings.append(transformed_i)

        # Stack: (..., num_features, d_model)
        stacked = torch.stack(feature_embeddings, dim=-2)

        # Compute selection weights
        # Flatten features for selection GRN input
        flat = stacked.reshape(*batch_shape, self.num_features * self.d_model)
        selection_input = self.selection_grn(flat, context)
        weights = F.softmax(self.selection_head(selection_input), dim=-1)  # (..., num_features)

        # Weighted combination: sum over features
        # weights: (..., num_features) -> (..., num_features, 1)
        # stacked: (..., num_features, d_model)
        selected = (stacked * weights.unsqueeze(-1)).sum(dim=-2)  # (..., d_model)

        return selected, weights
