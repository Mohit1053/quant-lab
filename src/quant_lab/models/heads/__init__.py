"""Prediction heads for multi-task forecasting."""

from quant_lab.models.heads.distribution_head import GaussianHead, StudentTHead
from quant_lab.models.heads.direction_head import DirectionHead
from quant_lab.models.heads.volatility_head import VolatilityHead

__all__ = ["GaussianHead", "StudentTHead", "DirectionHead", "VolatilityHead"]
