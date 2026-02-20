"""Tests for regime labeling."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.regime.labels import label_regimes, regime_summary_table, _avg_consecutive_run


class TestLabelRegimes:
    def test_returns_label_map(self):
        np.random.seed(42)
        regime_ids = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 0])
        returns = np.array([0.01, 0.02, -0.01, -0.02, 0.03, 0.04, 0.01, -0.01, 0.02, 0.01])
        volatility = np.array([0.01, 0.01, 0.03, 0.03, 0.02, 0.02, 0.01, 0.03, 0.02, 0.01])

        label_map = label_regimes(regime_ids, returns, volatility)
        assert isinstance(label_map, dict)
        assert len(label_map) == 3
        for cid, chars in label_map.items():
            assert chars.display_name != ""
            assert 0 <= chars.frequency <= 1

    def test_handles_noise_label(self):
        """HDBSCAN produces -1 for noise."""
        regime_ids = np.array([-1, 0, 0, 1, 1, -1])
        returns = np.array([0.0, 0.01, 0.02, -0.01, -0.02, 0.0])
        volatility = np.array([0.02, 0.01, 0.01, 0.03, 0.03, 0.02])

        label_map = label_regimes(regime_ids, returns, volatility)
        assert -1 not in label_map
        assert len(label_map) == 2

    def test_empty_regimes(self):
        label_map = label_regimes(np.array([]), np.array([]), np.array([]))
        assert label_map == {}


class TestAvgConsecutiveRun:
    def test_single_run(self):
        labels = np.array([0, 0, 0, 1, 1])
        assert _avg_consecutive_run(labels, 0) == 3.0

    def test_multiple_runs(self):
        labels = np.array([0, 0, 1, 0, 0, 0, 1])
        assert _avg_consecutive_run(labels, 0) == 2.5  # (2 + 3) / 2

    def test_no_occurrences(self):
        labels = np.array([1, 1, 1])
        assert _avg_consecutive_run(labels, 0) == 0.0


class TestRegimeSummaryTable:
    def test_produces_dataframe(self):
        from quant_lab.regime.labels import RegimeCharacteristics

        label_map = {
            0: RegimeCharacteristics(
                label="low_vol_bull", display_name="Low-Vol Bull",
                mean_return=0.01, mean_volatility=0.005,
                frequency=0.4, avg_duration=10.0,
            ),
            1: RegimeCharacteristics(
                label="bear", display_name="Bear",
                mean_return=-0.01, mean_volatility=0.02,
                frequency=0.3, avg_duration=5.0,
            ),
        }
        df = regime_summary_table(label_map)
        assert len(df) == 2
        assert "label" in df.columns
        assert "mean_return" in df.columns
