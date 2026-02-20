"""Tests for regime clustering."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.regime.clustering import RegimeClusterer, ClusterConfig


def _make_clusterable_data(n_samples=200, n_clusters=3, dim=8):
    """Generate clearly separable clusters."""
    np.random.seed(42)
    centers = np.random.randn(n_clusters, dim) * 5
    data = []
    for center in centers:
        data.append(center + np.random.randn(n_samples // n_clusters, dim) * 0.5)
    return np.vstack(data)


class TestKMeansClustering:
    def test_fit_returns_labels(self):
        config = ClusterConfig(method="kmeans", n_clusters=3)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data()
        labels = clusterer.fit(data)
        assert labels.shape == (len(data),)
        assert len(set(labels)) == 3

    def test_predict_after_fit(self):
        config = ClusterConfig(method="kmeans", n_clusters=3)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data()
        clusterer.fit(data)
        new_labels = clusterer.predict(data[:10])
        assert new_labels.shape == (10,)

    def test_labels_in_expected_range(self):
        config = ClusterConfig(method="kmeans", n_clusters=4)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data(n_clusters=4)
        labels = clusterer.fit(data)
        assert all(0 <= l < 4 for l in labels)


class TestGMMClustering:
    def test_fit_returns_labels(self):
        config = ClusterConfig(method="gmm", n_clusters=3)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data()
        labels = clusterer.fit(data)
        assert labels.shape == (len(data),)

    def test_cluster_probabilities(self):
        config = ClusterConfig(method="gmm", n_clusters=3)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data()
        clusterer.fit(data)
        probs = clusterer.get_cluster_probabilities(data[:5])
        assert probs is not None
        assert probs.shape == (5, 3)
        # Each row should sum to ~1
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestHDBSCANClustering:
    def test_fit_returns_labels(self):
        config = ClusterConfig(method="hdbscan", min_cluster_size=20)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data(n_samples=300)
        labels = clusterer.fit(data)
        assert labels.shape == (len(data),)
        # HDBSCAN may produce -1 for noise
        assert len(set(labels)) >= 1


class TestClustererCommon:
    def test_predict_before_fit_raises(self):
        clusterer = RegimeClusterer()
        with pytest.raises(RuntimeError, match="not fitted"):
            clusterer.predict(np.random.randn(10, 8))

    def test_no_scaling(self):
        config = ClusterConfig(method="kmeans", n_clusters=2, scale_features=False)
        clusterer = RegimeClusterer(config)
        data = _make_clusterable_data(n_clusters=2)
        labels = clusterer.fit(data)
        assert labels.shape == (len(data),)
