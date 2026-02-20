"""Linear baseline model using Ridge regression."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from quant_lab.models.base_model import BaseForecaster

logger = structlog.get_logger(__name__)


class RidgeBaseline(BaseForecaster):
    """Ridge regression baseline for return prediction.

    This is the Phase 1 model: fast, no GPU needed, validates the pipeline.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit Ridge regression with feature scaling."""
        # Remove any rows with NaN
        mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
        X_clean = X_train[mask]
        y_clean = y_train[mask]

        X_scaled = self.scaler.fit_transform(X_clean)
        self.model.fit(X_scaled, y_clean)
        self._is_fitted = True

        logger.info(
            "ridge_fit",
            samples=len(X_clean),
            features=X_clean.shape[1],
            alpha=self.alpha,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate return predictions."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, path: str | Path) -> None:
        """Save model and scaler to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "alpha": self.alpha}, f)
        logger.info("model_saved", path=str(path))

    def load(self, path: str | Path) -> None:
        """Load model and scaler from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.alpha = data["alpha"]
        self._is_fitted = True
        logger.info("model_loaded", path=str(path))

    def get_feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return feature importance based on coefficient magnitudes."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        coefs = self.model.coef_
        return dict(zip(feature_names, coefs.tolist()))
