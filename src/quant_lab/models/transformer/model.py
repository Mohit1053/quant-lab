"""Full transformer forecaster: encoder + decoder + multi-task heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_lab.models.transformer.encoder import TransformerEncoder
from quant_lab.models.transformer.decoder import ForecastDecoder
from quant_lab.models.heads.distribution_head import GaussianHead, StudentTHead
from quant_lab.models.heads.direction_head import DirectionHead
from quant_lab.models.heads.volatility_head import VolatilityHead


@dataclass
class TransformerConfig:
    """Full configuration for the transformer forecaster."""

    # Architecture
    num_features: int = 10
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    activation: str = "gelu"

    # Decoder
    pooling: str = "cls"  # cls, last, mean

    # Heads
    distribution_type: str = "gaussian"  # gaussian, student_t
    direction_num_classes: int = 3
    direction_threshold: float = 0.005
    volatility_enabled: bool = True

    # Loss weights
    distribution_weight: float = 1.0
    direction_weight: float = 0.3
    volatility_weight: float = 0.3

    @classmethod
    def from_hydra(cls, cfg) -> TransformerConfig:
        """Build config from Hydra OmegaConf dict."""
        arch = cfg.architecture
        return cls(
            num_features=cfg.input.get("num_features", 10),
            d_model=arch.d_model,
            nhead=arch.nhead,
            num_encoder_layers=arch.num_encoder_layers,
            dim_feedforward=arch.dim_feedforward,
            dropout=arch.dropout,
            activation=arch.activation,
            distribution_type=cfg.heads.distribution.type,
            direction_num_classes=cfg.heads.direction.num_classes,
            direction_threshold=cfg.heads.direction.threshold,
            volatility_enabled=cfg.heads.volatility.enabled,
            distribution_weight=cfg.loss.distribution_weight,
            direction_weight=cfg.loss.direction_weight,
            volatility_weight=cfg.loss.volatility_weight,
        )


class TransformerForecaster(nn.Module):
    """Multi-task transformer model for financial forecasting.

    Outputs:
    - distribution: predicted return distribution (Gaussian or Student-t)
    - direction: up/down/flat classification logits
    - volatility: predicted absolute return magnitude
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Encoder: features -> contextualized representations
        self.encoder = TransformerEncoder(
            num_features=config.num_features,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        # Decoder: pooled representation
        self.decoder = ForecastDecoder(
            d_model=config.d_model,
            pooling=config.pooling,
        )

        d_out = self.decoder.output_dim

        # Prediction heads
        self.heads = nn.ModuleDict()

        if config.distribution_type == "student_t":
            self.heads["distribution"] = StudentTHead(d_out)
        else:
            self.heads["distribution"] = GaussianHead(d_out)

        self.heads["direction"] = DirectionHead(
            d_out, num_classes=config.direction_num_classes
        )

        if config.volatility_enabled:
            self.heads["volatility"] = VolatilityHead(d_out)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        """Forward pass.

        Args:
            x: (batch, seq_len, num_features) raw feature sequences

        Returns:
            dict with head outputs:
              - 'distribution': dict with 'mean', 'log_var' (Gaussian)
              - 'direction': (batch, num_classes) logits
              - 'volatility': (batch,) positive values
        """
        encoded = self.encoder(x)  # (B, seq_len+1, d_model)
        pooled = self.decoder(encoded)  # (B, d_model)

        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(pooled)

        return outputs

    def predict_returns(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expected returns (convenience method for backtesting)."""
        outputs = self.forward(x)
        dist = outputs["distribution"]
        if isinstance(dist, dict):
            return dist.get("mean", dist.get("loc"))
        return dist

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
    def load(cls, path: str | Path, map_location: str = "cpu") -> TransformerForecaster:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss for the transformer forecaster.

    Combines:
    - Gaussian NLL for return distribution
    - Cross-entropy for direction classification
    - MSE for volatility prediction
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.distribution_weight = config.distribution_weight
        self.direction_weight = config.direction_weight
        self.volatility_weight = config.volatility_weight
        self.direction_threshold = config.direction_threshold
        self.volatility_enabled = config.volatility_enabled
        self.distribution_type = config.distribution_type

    def forward(
        self,
        outputs: dict[str, torch.Tensor | dict],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            outputs: model outputs from TransformerForecaster.forward()
            targets: dict with 'returns' (batch,) tensor

        Returns:
            (total_loss, loss_dict) where loss_dict has per-task losses
        """
        returns = targets["returns"]
        losses = {}
        total_loss = torch.tensor(0.0, device=returns.device, dtype=returns.dtype)

        # Distribution loss: Gaussian NLL
        if "distribution" in outputs:
            dist = outputs["distribution"]
            if self.distribution_type == "student_t":
                total_loss, losses = self._student_t_nll(dist, returns, total_loss, losses)
            else:
                total_loss, losses = self._gaussian_nll(dist, returns, total_loss, losses)

        # Direction loss: cross-entropy
        if "direction" in outputs:
            direction_logits = outputs["direction"]
            direction_labels = self._returns_to_direction(returns)
            direction_loss = F.cross_entropy(direction_logits, direction_labels)
            losses["direction"] = direction_loss.item()
            total_loss = total_loss + self.direction_weight * direction_loss

        # Volatility loss: MSE with |return| as target
        if "volatility" in outputs and self.volatility_enabled:
            vol_pred = outputs["volatility"]
            vol_target = returns.abs()
            vol_loss = F.mse_loss(vol_pred, vol_target)
            losses["volatility"] = vol_loss.item()
            total_loss = total_loss + self.volatility_weight * vol_loss

        losses["total"] = total_loss.item()
        return total_loss, losses

    def _gaussian_nll(self, dist, returns, total_loss, losses):
        mean = dist["mean"]
        log_var = dist["log_var"]
        # Gaussian NLL: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
        nll = 0.5 * (log_var + (returns - mean) ** 2 / (log_var.exp() + 1e-8))
        nll = nll.mean()
        losses["distribution"] = nll.item()
        total_loss = total_loss + self.distribution_weight * nll
        return total_loss, losses

    def _student_t_nll(self, dist, returns, total_loss, losses):
        loc = dist["loc"]
        log_scale = dist["log_scale"]
        log_df = dist["log_df"]
        scale = log_scale.exp() + 1e-8
        df = log_df.exp() + 2.0  # Ensure df > 2 for finite variance
        # Simplified Student-t NLL
        z = (returns - loc) / scale
        nll = (
            0.5 * (df + 1) * torch.log1p(z ** 2 / df)
            + log_scale
            + 0.5 * torch.log(df)
        )
        nll = nll.mean()
        losses["distribution"] = nll.item()
        total_loss = total_loss + self.distribution_weight * nll
        return total_loss, losses

    def _returns_to_direction(self, returns: torch.Tensor) -> torch.Tensor:
        """Convert continuous returns to direction labels.

        Returns:
            (batch,) long tensor: 0=down, 1=flat, 2=up
        """
        labels = torch.ones_like(returns, dtype=torch.long)  # flat by default
        labels[returns > self.direction_threshold] = 2  # up
        labels[returns < -self.direction_threshold] = 0  # down
        return labels
