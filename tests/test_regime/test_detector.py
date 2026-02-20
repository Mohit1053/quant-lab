"""Tests for regime detector orchestrator."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.regime.detector import RegimeDetector, DetectorConfig
from quant_lab.regime.hmm import HMMConfig


def _make_data(n_samples=200, embed_dim=8):
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, embed_dim)
    returns = np.random.randn(n_samples) * 0.01
    volatility = np.abs(np.random.randn(n_samples)) * 0.02
    return embeddings, returns, volatility


class TestRegimeDetector:
    def test_kmeans_detection(self):
        embeddings, returns, volatility = _make_data()
        detector = RegimeDetector(DetectorConfig(method="kmeans", n_regimes=3))
        result = detector.fit(embeddings=embeddings, returns=returns, volatility=volatility)
        assert "labels" in result
        assert "label_map" in result
        assert "summary" in result
        assert result["labels"].shape == (200,)

    def test_gmm_detection(self):
        embeddings, returns, volatility = _make_data()
        detector = RegimeDetector(DetectorConfig(method="gmm", n_regimes=3))
        result = detector.fit(embeddings=embeddings, returns=returns, volatility=volatility)
        assert result["labels"].shape == (200,)

    def test_hmm_detection(self):
        _, returns, volatility = _make_data()
        hmm_cfg = HMMConfig(n_regimes=3, covariance_type="diag")
        detector = RegimeDetector(DetectorConfig(method="hmm", n_regimes=3, hmm_config=hmm_cfg))
        result = detector.fit(returns=returns, volatility=volatility)
        assert result["labels"].shape == (200,)

    def test_predict_after_fit(self):
        embeddings, returns, volatility = _make_data()
        detector = RegimeDetector(DetectorConfig(method="kmeans", n_regimes=3))
        detector.fit(embeddings=embeddings, returns=returns, volatility=volatility)
        new_labels = detector.predict(embeddings=embeddings[:10])
        assert new_labels.shape == (10,)

    def test_hmm_predict_after_fit(self):
        _, returns, volatility = _make_data()
        hmm_cfg = HMMConfig(n_regimes=3, covariance_type="diag")
        detector = RegimeDetector(DetectorConfig(method="hmm", n_regimes=3, hmm_config=hmm_cfg))
        detector.fit(returns=returns, volatility=volatility)
        new_labels = detector.predict(returns=returns[:10], volatility=volatility[:10])
        assert new_labels.shape == (10,)

    def test_missing_embeddings_raises(self):
        detector = RegimeDetector(DetectorConfig(method="kmeans"))
        with pytest.raises(ValueError, match="Embeddings required"):
            detector.fit(embeddings=None)

    def test_missing_returns_for_hmm_raises(self):
        detector = RegimeDetector(DetectorConfig(method="hmm"))
        with pytest.raises(ValueError, match="Returns and volatility required"):
            detector.fit(returns=None, volatility=None)

    def test_result_has_summary_table(self):
        embeddings, returns, volatility = _make_data()
        detector = RegimeDetector(DetectorConfig(method="kmeans", n_regimes=3))
        result = detector.fit(embeddings=embeddings, returns=returns, volatility=volatility)
        assert len(result["summary"]) > 0
