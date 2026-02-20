"""Regime clustering: KMeans, GMM, HDBSCAN on embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan


class ClusterMethod(str, Enum):
    KMEANS = "kmeans"
    GMM = "gmm"
    HDBSCAN = "hdbscan"


@dataclass
class ClusterConfig:
    """Clustering configuration."""

    method: str = "kmeans"
    n_clusters: int = 4
    min_cluster_size: int = 50  # HDBSCAN specific
    min_samples: int = 10  # HDBSCAN specific
    random_state: int = 42
    scale_features: bool = True


class RegimeClusterer:
    """Cluster market embeddings into discrete regimes.

    Supports KMeans, GMM, and HDBSCAN. All methods operate on
    standardized embedding vectors.

    Args:
        config: Clustering configuration.
    """

    def __init__(self, config: ClusterConfig | None = None):
        self.config = config or ClusterConfig()
        self.scaler = StandardScaler() if self.config.scale_features else None
        self._model = None

    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering model and return labels.

        Args:
            embeddings: (n_samples, embedding_dim) array.

        Returns:
            (n_samples,) integer labels. -1 for noise (HDBSCAN only).
        """
        X = self._preprocess(embeddings, fit=True)

        method = ClusterMethod(self.config.method)

        if method == ClusterMethod.KMEANS:
            self._model = KMeans(
                n_clusters=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=10,
            )
            labels = self._model.fit_predict(X)

        elif method == ClusterMethod.GMM:
            self._model = GaussianMixture(
                n_components=self.config.n_clusters,
                random_state=self.config.random_state,
                n_init=3,
            )
            self._model.fit(X)
            labels = self._model.predict(X)

        elif method == ClusterMethod.HDBSCAN:
            self._model = hdbscan.HDBSCAN(
                min_cluster_size=self.config.min_cluster_size,
                min_samples=self.config.min_samples,
            )
            labels = self._model.fit_predict(X)

        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        return labels

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data.

        Args:
            embeddings: (n_samples, embedding_dim) array.

        Returns:
            (n_samples,) integer labels.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._preprocess(embeddings, fit=False)

        method = ClusterMethod(self.config.method)
        if method == ClusterMethod.KMEANS:
            return self._model.predict(X)
        elif method == ClusterMethod.GMM:
            return self._model.predict(X)
        elif method == ClusterMethod.HDBSCAN:
            # HDBSCAN approximate_predict for new data
            labels, _ = hdbscan.approximate_predict(self._model, X)
            return labels
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def get_cluster_probabilities(self, embeddings: np.ndarray) -> np.ndarray | None:
        """Get soft cluster membership probabilities (GMM only).

        Returns:
            (n_samples, n_clusters) probability matrix, or None if not available.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._preprocess(embeddings, fit=False)

        method = ClusterMethod(self.config.method)
        if method == ClusterMethod.GMM:
            return self._model.predict_proba(X)
        return None

    def _preprocess(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """Optionally scale features."""
        if self.scaler is not None:
            if fit:
                return self.scaler.fit_transform(X)
            return self.scaler.transform(X)
        return X
