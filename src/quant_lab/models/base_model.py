"""Abstract base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Interface for all forecasting models (DL and sklearn)."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Load model from disk."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Compute evaluation metrics on a dataset."""
        preds = self.predict(X)
        mse = float(np.mean((preds - y) ** 2))
        mae = float(np.mean(np.abs(preds - y)))

        # Directional accuracy: did we predict the sign correctly?
        direction_correct = np.sign(preds) == np.sign(y)
        direction_accuracy = float(np.mean(direction_correct))

        # IC (Information Coefficient): rank correlation
        from scipy.stats import spearmanr

        ic, ic_pval = spearmanr(preds, y)

        return {
            "mse": mse,
            "mae": mae,
            "direction_accuracy": direction_accuracy,
            "ic": float(ic),
            "ic_pval": float(ic_pval),
        }
