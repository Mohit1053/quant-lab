"""Tests for action space processor."""

from __future__ import annotations

import numpy as np
import pytest

from quant_lab.rl.environments.action_space import ActionProcessor


class TestActionProcessor:
    def test_weights_sum_to_max_leverage(self):
        proc = ActionProcessor(num_assets=5, max_leverage=1.0)
        raw = np.random.randn(5)
        weights = proc.process(raw)
        assert np.sum(weights) <= 1.0 + 1e-6

    def test_weights_non_negative(self):
        proc = ActionProcessor(num_assets=5, min_weight=0.0)
        raw = np.random.randn(5)
        weights = proc.process(raw)
        assert (weights >= 0).all()

    def test_max_weight_constraint(self):
        proc = ActionProcessor(num_assets=5, max_weight=0.20)
        raw = np.ones(5) * 10.0  # All very positive
        weights = proc.process(raw)
        assert (weights <= 0.20 + 1e-6).all()

    def test_turnover_computation(self):
        proc = ActionProcessor(num_assets=3)
        old = np.array([0.3, 0.3, 0.4])
        new = np.array([0.5, 0.2, 0.3])
        turnover = proc.compute_turnover(old, new)
        expected = abs(0.5 - 0.3) + abs(0.2 - 0.3) + abs(0.3 - 0.4)
        assert abs(turnover - expected) < 1e-6

    def test_zero_turnover_same_weights(self):
        proc = ActionProcessor(num_assets=3)
        w = np.array([0.3, 0.3, 0.4])
        assert proc.compute_turnover(w, w) == 0.0

    def test_cash_weight(self):
        proc = ActionProcessor(num_assets=3)
        weights = np.array([0.2, 0.3, 0.1])
        cash = proc.get_cash_weight(weights)
        assert abs(cash - 0.4) < 1e-6

    def test_full_investment_no_cash(self):
        proc = ActionProcessor(num_assets=3)
        weights = np.array([0.3, 0.3, 0.4])
        cash = proc.get_cash_weight(weights)
        assert abs(cash) < 1e-6

    def test_process_returns_correct_shape(self):
        proc = ActionProcessor(num_assets=10)
        raw = np.random.randn(10)
        weights = proc.process(raw)
        assert weights.shape == (10,)
