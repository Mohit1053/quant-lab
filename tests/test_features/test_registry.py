"""Tests for feature registry."""

from __future__ import annotations

import pytest

from quant_lab.features.registry import (
    FEATURE_REGISTRY,
    get_feature_func,
    list_features,
    register_feature,
)

# Import to trigger registration of built-in features
import quant_lab.features.price_features  # noqa: F401


class TestRegistry:
    def test_builtin_features_registered(self):
        """All Phase 1 features should be registered on import."""
        expected = {"log_returns", "realized_volatility", "momentum", "max_drawdown"}
        registered = set(FEATURE_REGISTRY.keys())
        assert expected.issubset(registered)

    def test_get_feature_func_returns_callable(self):
        func = get_feature_func("log_returns")
        assert callable(func)

    def test_get_unknown_feature_raises(self):
        with pytest.raises(ValueError, match="Unknown feature"):
            get_feature_func("nonexistent_feature_xyz")

    def test_list_features_returns_dicts(self):
        features = list_features()
        assert len(features) > 0
        assert all("name" in f and "description" in f for f in features)

    def test_custom_registration(self):
        @register_feature("test_feature_custom", "A test feature")
        def my_feature(df, **kwargs):
            return df

        assert "test_feature_custom" in FEATURE_REGISTRY
        func = get_feature_func("test_feature_custom")
        assert func is my_feature

        # Cleanup
        del FEATURE_REGISTRY["test_feature_custom"]
