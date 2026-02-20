"""Tests for feature registry."""

from __future__ import annotations

import pytest

from quant_lab.features.registry import (
    FEATURE_REGISTRY,
    get_feature_func,
    list_features,
)


class TestFeatureRegistry:
    def test_registered_features_not_empty(self):
        assert len(FEATURE_REGISTRY) > 0

    def test_get_feature_func_returns_callable(self):
        func = get_feature_func("log_returns")
        assert callable(func)

    def test_get_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            get_feature_func("totally_fake_feature")

    def test_list_features_returns_dicts(self):
        features = list_features()
        assert isinstance(features, list)
        assert len(features) > 0
        assert "name" in features[0]
        assert "description" in features[0]

    def test_known_features_registered(self):
        expected = ["log_returns", "realized_volatility", "momentum", "max_drawdown"]
        for name in expected:
            assert name in FEATURE_REGISTRY, f"Feature '{name}' not registered"
