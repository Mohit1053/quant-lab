"""Tests for HMM regime detection."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.regime.hmm import RegimeHMM, HMMConfig


def _make_regime_features(n_samples=500, n_regimes=3):
    """Generate synthetic features with regime structure."""
    np.random.seed(42)
    regime_means = [[-0.01, 0.02], [0.01, 0.01], [-0.03, 0.04]]
    features = []
    for i in range(n_samples):
        regime = i % n_regimes
        features.append(np.random.multivariate_normal(
            regime_means[regime],
            np.eye(2) * 0.001,
        ))
    return np.array(features)


class TestRegimeHMM:
    def test_fit_returns_labels(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        labels = hmm.fit(features)
        assert labels.shape == (len(features),)
        assert len(set(labels)) >= 2  # At least 2 distinct regimes detected

    def test_predict_after_fit(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        hmm.fit(features)
        new_labels = hmm.predict(features[:20])
        assert new_labels.shape == (20,)

    def test_predict_proba(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        hmm.fit(features)
        probs = hmm.predict_proba(features[:10])
        assert probs.shape == (10, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_transition_matrix(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        hmm.fit(features)
        trans = hmm.transition_matrix
        assert trans.shape == (3, 3)
        # Each row should sum to ~1
        assert np.allclose(trans.sum(axis=1), 1.0, atol=1e-5)

    def test_regime_means(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        hmm.fit(features)
        means = hmm.regime_means
        assert means.shape == (3, 2)  # 3 regimes, 2 features

    def test_predict_before_fit_raises(self):
        hmm = RegimeHMM()
        with pytest.raises(RuntimeError, match="not fitted"):
            hmm.predict(np.random.randn(10, 2))

    def test_stationary_distribution(self):
        hmm = RegimeHMM(HMMConfig(n_regimes=3))
        features = _make_regime_features()
        hmm.fit(features)
        stationary = hmm.stationary_distribution
        assert stationary.shape == (3,)
        assert abs(stationary.sum() - 1.0) < 1e-5
