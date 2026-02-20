"""Tests for ensemble signal combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.models.ensemble import (
    EnsembleStrategy,
    EnsembleConfig,
    CombinationMethod,
)


@pytest.fixture
def sample_signals():
    """Two model signal DataFrames: 3 dates x 2 tickers = 6 rows each."""
    dates = pd.bdate_range("2023-01-02", periods=3)
    tickers = ["A", "B"]
    records = []
    for d in dates:
        for t in tickers:
            records.append({"date": d, "ticker": t})
    base = pd.DataFrame(records)

    sig_a = base.copy()
    sig_a["signal"] = [0.1, 0.2, 0.3, 0.4, -0.1, -0.2]

    sig_b = base.copy()
    sig_b["signal"] = [0.2, 0.1, 0.1, 0.2, 0.0, 0.0]

    return {"model_a": sig_a, "model_b": sig_b}


class TestSimpleAverage:
    def test_equal_weight(self, sample_signals):
        cfg = EnsembleConfig(method=CombinationMethod.SIMPLE_AVERAGE)
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals)

        assert "signal" in result.columns
        assert len(result) == 6
        # First row: (0.1 + 0.2) / 2 = 0.15
        assert abs(result.iloc[0]["signal"] - 0.15) < 1e-6

    def test_output_columns(self, sample_signals):
        ens = EnsembleStrategy()
        result = ens.combine(sample_signals)
        assert list(result.columns) == ["date", "ticker", "signal"]

    def test_symmetric(self, sample_signals):
        """Order of models shouldn't matter for simple average."""
        ens = EnsembleStrategy()
        result1 = ens.combine(sample_signals)
        reversed_signals = dict(reversed(list(sample_signals.items())))
        result2 = ens.combine(reversed_signals)
        pd.testing.assert_series_equal(
            result1["signal"].reset_index(drop=True),
            result2["signal"].reset_index(drop=True),
        )


class TestWeightedAverage:
    def test_custom_weights(self, sample_signals):
        cfg = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            weights={"model_a": 0.8, "model_b": 0.2},
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals)
        # First row: 0.1*0.8 + 0.2*0.2 = 0.12
        assert abs(result.iloc[0]["signal"] - 0.12) < 1e-6

    def test_zero_weight_excludes_model(self, sample_signals):
        cfg = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            weights={"model_a": 1.0, "model_b": 0.0},
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals)
        # Should equal model_a signals exactly
        expected = sample_signals["model_a"]["signal"].values
        np.testing.assert_allclose(result["signal"].values, expected, atol=1e-6)

    def test_empty_weights_defaults_equal(self, sample_signals):
        cfg = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            weights={},
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals)
        # Same as simple average
        expected = (
            sample_signals["model_a"]["signal"].values
            + sample_signals["model_b"]["signal"].values
        ) / 2
        np.testing.assert_allclose(result["signal"].values, expected, atol=1e-6)


class TestRegimeConditional:
    def test_regime_weights_applied(self, sample_signals):
        dates = pd.bdate_range("2023-01-02", periods=3)
        regime_labels = pd.Series([0, 0, 1], index=dates)

        cfg = EnsembleConfig(
            method=CombinationMethod.REGIME_CONDITIONAL,
            regime_weights={
                0: {"model_a": 1.0, "model_b": 0.0},
                1: {"model_a": 0.0, "model_b": 1.0},
            },
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals, regime_labels=regime_labels)

        # Regime 0 (dates[0], dates[1]): only model_a
        r0_rows = result[result["date"] == dates[0]]
        assert abs(r0_rows.iloc[0]["signal"] - 0.1) < 1e-6  # model_a: A=0.1

        # Regime 1 (dates[2]): only model_b
        r1_rows = result[result["date"] == dates[2]]
        assert abs(r1_rows.iloc[0]["signal"] - 0.0) < 1e-6  # model_b: A=-0.1→0.0

    def test_falls_back_without_labels(self, sample_signals):
        cfg = EnsembleConfig(
            method=CombinationMethod.REGIME_CONDITIONAL,
            weights={"model_a": 0.5, "model_b": 0.5},
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals)  # no regime_labels
        assert len(result) == 6

    def test_unknown_regime_uses_default(self, sample_signals):
        dates = pd.bdate_range("2023-01-02", periods=3)
        regime_labels = pd.Series([99, 99, 99], index=dates)  # regime 99 not in map

        cfg = EnsembleConfig(
            method=CombinationMethod.REGIME_CONDITIONAL,
            weights={"model_a": 0.5, "model_b": 0.5},
            regime_weights={0: {"model_a": 1.0, "model_b": 0.0}},
        )
        ens = EnsembleStrategy(cfg)
        result = ens.combine(sample_signals, regime_labels=regime_labels)
        # Should use default 50/50 weights
        expected = (
            sample_signals["model_a"]["signal"].values
            + sample_signals["model_b"]["signal"].values
        ) / 2
        np.testing.assert_allclose(result["signal"].values, expected, atol=1e-6)


class TestSingleModel:
    def test_passthrough(self, sample_signals):
        single = {"model_a": sample_signals["model_a"]}
        ens = EnsembleStrategy()
        result = ens.combine(single)
        np.testing.assert_allclose(
            result["signal"].values,
            sample_signals["model_a"]["signal"].values,
        )


class TestModelContributions:
    def test_contributions_dataframe(self, sample_signals):
        ens = EnsembleStrategy()
        contrib = ens.get_model_contributions(sample_signals)
        assert len(contrib) == 2
        assert set(contrib.columns) >= {
            "model",
            "mean_signal",
            "std_signal",
            "correlation_with_ensemble",
        }
        assert set(contrib["model"]) == {"model_a", "model_b"}

    def test_correlations_valid(self, sample_signals):
        ens = EnsembleStrategy()
        contrib = ens.get_model_contributions(sample_signals)
        for _, row in contrib.iterrows():
            assert -1.0 <= row["correlation_with_ensemble"] <= 1.0


class TestNoSignalsRaises:
    def test_empty_dict(self):
        ens = EnsembleStrategy()
        with pytest.raises(ValueError, match="No signals"):
            ens.combine({})
