"""Regime detection orchestrator.

Combines clustering and HMM methods, assigns labels, and produces
regime time series for downstream use in backtesting and RL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import structlog

from quant_lab.regime.clustering import RegimeClusterer, ClusterConfig
from quant_lab.regime.hmm import RegimeHMM, HMMConfig
from quant_lab.regime.labels import label_regimes, RegimeCharacteristics, regime_summary_table

logger = structlog.get_logger(__name__)


@dataclass
class DetectorConfig:
    """Regime detector configuration."""

    method: str = "kmeans"  # kmeans, gmm, hdbscan, hmm
    n_regimes: int = 4
    cluster_config: ClusterConfig | None = None
    hmm_config: HMMConfig | None = None


class RegimeDetector:
    """Orchestrates regime detection using multiple methods.

    Supports both clustering-based (on embeddings) and HMM-based
    (on returns/volatility features) approaches.

    Usage:
        detector = RegimeDetector(config)
        result = detector.fit(embeddings, returns, volatility)
        # result contains labels, label_map, and summary table
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self._clusterer: RegimeClusterer | None = None
        self._hmm: RegimeHMM | None = None

    def fit(
        self,
        embeddings: np.ndarray | None = None,
        returns: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
    ) -> dict:
        """Fit regime detection model and return results.

        For clustering methods (kmeans/gmm/hdbscan): requires embeddings.
        For HMM: requires returns and volatility.

        Args:
            embeddings: (n_samples, embedding_dim) for clustering methods.
            returns: (n_samples,) return series.
            volatility: (n_samples,) volatility series.

        Returns:
            Dict with:
                - 'labels': (n_samples,) integer regime IDs
                - 'label_map': dict mapping ID -> RegimeCharacteristics
                - 'summary': DataFrame with regime summary
        """
        if self.config.method == "hmm":
            labels = self._fit_hmm(returns, volatility)
        else:
            labels = self._fit_clustering(embeddings)

        # Warn if no valid regimes found (e.g., HDBSCAN all noise)
        n_valid = len(set(labels[labels >= 0]))
        if n_valid == 0:
            logger.warning(
                "no_valid_regimes_found",
                method=self.config.method,
                total_samples=len(labels),
                noise_samples=int((labels < 0).sum()),
            )

        # Assign interpretable labels
        if returns is not None and volatility is not None and n_valid > 0:
            label_map = label_regimes(labels, returns, volatility)
        else:
            label_map = {}

        summary = regime_summary_table(label_map) if label_map else pd.DataFrame()

        logger.info(
            "regime_detection_complete",
            method=self.config.method,
            n_regimes=n_valid,
            label_map={k: v.display_name for k, v in label_map.items()},
        )

        return {
            "labels": labels,
            "label_map": label_map,
            "summary": summary,
        }

    def predict(
        self,
        embeddings: np.ndarray | None = None,
        returns: np.ndarray | None = None,
        volatility: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict regimes for new data."""
        if self.config.method == "hmm":
            if self._hmm is None:
                raise RuntimeError("HMM not fitted. Call fit() first.")
            features = np.column_stack([returns, volatility])
            return self._hmm.predict(features)
        else:
            if self._clusterer is None:
                raise RuntimeError("Clusterer not fitted. Call fit() first.")
            return self._clusterer.predict(embeddings)

    def _fit_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering-based regime detection."""
        if embeddings is None:
            raise ValueError("Embeddings required for clustering methods.")

        cluster_config = self.config.cluster_config or ClusterConfig(
            method=self.config.method,
            n_clusters=self.config.n_regimes,
        )
        cluster_config.method = self.config.method
        cluster_config.n_clusters = self.config.n_regimes

        self._clusterer = RegimeClusterer(cluster_config)
        return self._clusterer.fit(embeddings)

    def _fit_hmm(
        self,
        returns: np.ndarray | None,
        volatility: np.ndarray | None,
    ) -> np.ndarray:
        """Fit HMM-based regime detection."""
        if returns is None or volatility is None:
            raise ValueError("Returns and volatility required for HMM method.")

        hmm_config = self.config.hmm_config or HMMConfig(
            n_regimes=self.config.n_regimes,
        )
        hmm_config.n_regimes = self.config.n_regimes

        self._hmm = RegimeHMM(hmm_config)
        features = np.column_stack([returns, volatility])
        return self._hmm.fit(features)
