"""Temporal Fusion Transformer (TFT) forecaster.

Full TFT with:
- Variable Selection Networks for interpretable feature importance
- LSTM encoder for local temporal patterns
- Interpretable multi-head attention for long-range dependencies
- Gated Residual Networks throughout
- Same multi-task heads as vanilla transformer (distribution, direction, volatility)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_lab.models.tft.gated_residual import GatedResidualNetwork, GatedLinearUnit
from quant_lab.models.tft.variable_selection import VariableSelectionNetwork
from quant_lab.models.heads.distribution_head import GaussianHead, StudentTHead
from quant_lab.models.heads.direction_head import DirectionHead
from quant_lab.models.heads.volatility_head import VolatilityHead


@dataclass
class TFTConfig:
    """Configuration for the Temporal Fusion Transformer."""

    # Input
    num_features: int = 10
    sequence_length: int = 63

    # Architecture
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 2
    lstm_layers: int = 1
    lstm_hidden: int = 128
    dropout: float = 0.1
    grn_hidden: int = 64

    # Heads
    distribution_type: str = "gaussian"
    direction_num_classes: int = 3
    direction_threshold: float = 0.005
    volatility_enabled: bool = True

    # Loss weights
    distribution_weight: float = 1.0
    direction_weight: float = 0.3
    volatility_weight: float = 0.3

    @classmethod
    def from_hydra(cls, cfg) -> TFTConfig:
        """Build config from Hydra OmegaConf dict."""
        arch = cfg.architecture
        return cls(
            num_features=cfg.input.get("num_features", 10),
            sequence_length=cfg.input.get("sequence_length", 63),
            d_model=arch.d_model,
            nhead=arch.nhead,
            num_encoder_layers=arch.num_encoder_layers,
            lstm_layers=arch.lstm_layers,
            lstm_hidden=arch.lstm_hidden,
            dropout=arch.dropout,
            grn_hidden=arch.grn_hidden,
            distribution_type=cfg.heads.distribution.type,
            direction_num_classes=cfg.heads.direction.num_classes,
            direction_threshold=cfg.heads.direction.threshold,
            volatility_enabled=cfg.heads.volatility.enabled,
            distribution_weight=cfg.loss.distribution_weight,
            direction_weight=cfg.loss.direction_weight,
            volatility_weight=cfg.loss.volatility_weight,
        )


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention from TFT paper.

    Unlike standard multi-head attention, this uses shared value projections
    and additive aggregation to produce interpretable attention weights.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: (B, T_q, d_model)
            key: (B, T_k, d_model)
            value: (B, T_k, d_model)

        Returns:
            output: (B, T_q, d_model)
            attn_weights: (B, nhead, T_q, T_k) interpretable attention weights
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project and reshape: (B, T, d_model) -> (B, nhead, T, d_k)
        q = self.q_proj(query).view(B, T_q, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.d_k ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, nhead, T_q, T_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)  # (B, nhead, T_q, d_k)

        # Reshape back: (B, nhead, T_q, d_k) -> (B, T_q, d_model)
        attended = attended.transpose(1, 2).contiguous().view(B, T_q, -1)
        output = self.out_proj(attended)

        return output, attn_weights


class TFTForecaster(nn.Module):
    """Temporal Fusion Transformer for financial time-series forecasting.

    Architecture flow:
    1. Variable Selection: select most relevant features
    2. LSTM Encoder: capture local temporal patterns
    3. Static Enrichment: GRN with context (final LSTM hidden state)
    4. Interpretable Multi-Head Attention: long-range dependencies
    5. Position-wise GRN: final non-linear processing
    6. Multi-task heads: distribution, direction, volatility
    """

    def __init__(self, config: TFTConfig):
        super().__init__()
        self.config = config

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            num_features=config.num_features,
            d_model=config.d_model,
            d_hidden=config.grn_hidden,
            dropout=config.dropout,
        )

        # 2. LSTM Encoder for local temporal processing
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        # Project LSTM output back to d_model
        self.lstm_proj = nn.Linear(config.lstm_hidden, config.d_model)

        # Gate + LayerNorm after LSTM (skip connection from VSN output)
        self.lstm_glu = GatedLinearUnit(config.d_model, config.d_model)
        self.lstm_norm = nn.LayerNorm(config.d_model)

        # 3. Static enrichment GRN (uses final LSTM hidden as context)
        self.enrichment_grn = GatedResidualNetwork(
            d_model=config.d_model,
            d_hidden=config.grn_hidden,
            d_context=config.lstm_hidden,
            dropout=config.dropout,
        )

        # 4. Interpretable Multi-Head Attention
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(
                d_model=config.d_model,
                nhead=config.nhead,
                dropout=config.dropout,
            )
            for _ in range(config.num_encoder_layers)
        ])
        # Gate + Norm after each attention layer
        self.attn_glu = nn.ModuleList([
            GatedLinearUnit(config.d_model, config.d_model)
            for _ in range(config.num_encoder_layers)
        ])
        self.attn_norm = nn.ModuleList([
            nn.LayerNorm(config.d_model)
            for _ in range(config.num_encoder_layers)
        ])

        # 5. Position-wise feed-forward with GRN
        self.ff_grn = GatedResidualNetwork(
            d_model=config.d_model,
            d_hidden=config.grn_hidden,
            dropout=config.dropout,
        )
        self.ff_glu = GatedLinearUnit(config.d_model, config.d_model)
        self.ff_norm = nn.LayerNorm(config.d_model)

        # 6. Multi-task prediction heads (same as vanilla transformer)
        self.heads = nn.ModuleDict()

        if config.distribution_type == "student_t":
            self.heads["distribution"] = StudentTHead(config.d_model)
        else:
            self.heads["distribution"] = GaussianHead(config.d_model)

        self.heads["direction"] = DirectionHead(
            config.d_model, num_classes=config.direction_num_classes
        )

        if config.volatility_enabled:
            self.heads["volatility"] = VolatilityHead(config.d_model)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | dict]:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_features)

        Returns:
            dict with head outputs + 'attention_weights' + 'feature_weights'
        """
        B, T, F_in = x.shape

        # 1. Variable Selection (applied per time step)
        # Reshape to (B*T, F) for VSN, then back
        x_flat = x.reshape(B * T, F_in)
        selected, feature_weights = self.vsn(x_flat)
        selected = selected.reshape(B, T, -1)  # (B, T, d_model)
        feature_weights = feature_weights.reshape(B, T, F_in)

        # 2. LSTM Encoder
        lstm_out, (h_n, _) = self.lstm(selected)  # (B, T, lstm_hidden)
        lstm_proj = self.lstm_proj(lstm_out)  # (B, T, d_model)

        # Gate + skip connection + LayerNorm
        gated = self.lstm_glu(lstm_proj)
        temporal = self.lstm_norm(gated + selected)

        # Extract context from final LSTM hidden state
        # h_n: (num_layers, B, lstm_hidden) -> take last layer
        context = h_n[-1]  # (B, lstm_hidden)

        # 3. Static enrichment: enrich temporal features with context
        # Expand context to match time dimension
        context_expanded = context.unsqueeze(1).expand(-1, T, -1)  # (B, T, lstm_hidden)
        enriched = self.enrichment_grn(
            temporal.reshape(B * T, -1),
            context_expanded.reshape(B * T, -1),
        ).reshape(B, T, -1)

        # 4. Interpretable Multi-Head Attention
        attn_input = enriched
        all_attn_weights = []
        for i, (attn, glu, norm) in enumerate(
            zip(self.attention_layers, self.attn_glu, self.attn_norm)
        ):
            attn_out, attn_w = attn(attn_input, attn_input, attn_input)
            all_attn_weights.append(attn_w)
            gated_attn = glu(attn_out)
            attn_input = norm(gated_attn + attn_input)

        # 5. Position-wise feed-forward
        ff_out = self.ff_grn(attn_input.reshape(B * T, -1)).reshape(B, T, -1)
        ff_gated = self.ff_glu(ff_out)
        processed = self.ff_norm(ff_gated + attn_input)

        # Pool: use last time step for prediction
        pooled = processed[:, -1, :]  # (B, d_model)

        # 6. Multi-task heads
        outputs: dict[str, torch.Tensor | dict] = {}
        for name, head in self.heads.items():
            outputs[name] = head(pooled)

        # Store interpretability outputs
        outputs["attention_weights"] = all_attn_weights
        outputs["feature_weights"] = feature_weights.mean(dim=1)  # (B, F_in) avg over time

        return outputs

    def predict_returns(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expected returns."""
        outputs = self.forward(x)
        dist = outputs["distribution"]
        if isinstance(dist, dict):
            return dist.get("mean", dist.get("loc"))
        return dist

    def get_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature importance weights for interpretation.

        Returns:
            (batch, num_features) softmax weights indicating feature importance.
        """
        outputs = self.forward(x)
        return outputs["feature_weights"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"config": self.config, "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> TFTForecaster:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
