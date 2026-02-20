"""Tests for RidgeBaseline model."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.models.linear_baseline import RidgeBaseline


class TestRidgeBaseline:
    def test_fit_and_predict(self):
        model = RidgeBaseline(alpha=1.0)
        X = np.random.randn(100, 5).astype(np.float32)
        y = X @ np.array([0.1, -0.2, 0.3, -0.1, 0.05]) + np.random.randn(100) * 0.01
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_predict_before_fit_raises(self):
        model = RidgeBaseline()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.random.randn(10, 5))

    def test_fit_with_nan(self):
        model = RidgeBaseline()
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        # Inject NaN
        X[5, 0] = np.nan
        y[10] = np.nan
        model.fit(X, y)
        # Should fit on clean rows only
        preds = model.predict(X[:5])  # First 5 rows are clean
        assert preds.shape == (5,)

    def test_save_and_load(self, tmp_path):
        model = RidgeBaseline(alpha=2.0)
        X = np.random.randn(50, 4)
        y = np.random.randn(50)
        model.fit(X, y)
        preds_before = model.predict(X)

        path = tmp_path / "ridge.pkl"
        model.save(path)

        loaded = RidgeBaseline()
        loaded.load(path)
        preds_after = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_before, preds_after)
        assert loaded.alpha == 2.0

    def test_feature_importance(self):
        model = RidgeBaseline()
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        model.fit(X, y)
        importance = model.get_feature_importance(["f1", "f2", "f3"])
        assert len(importance) == 3
        assert "f1" in importance

    def test_feature_importance_before_fit_raises(self):
        model = RidgeBaseline()
        with pytest.raises(RuntimeError):
            model.get_feature_importance(["f1"])
