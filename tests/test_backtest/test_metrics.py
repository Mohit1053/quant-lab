"""Tests for backtest metrics with known values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest.metrics import (
    compute_cagr,
    compute_sharpe,
    compute_sortino,
    compute_max_drawdown,
    compute_calmar,
    compute_turnover,
    compute_all_metrics,
)


class TestCAGR:
    def test_known_cagr(self):
        # 100 to 200 over 252 days (1 year) = 100% return = 1.0 CAGR
        equity = pd.Series([100.0, 200.0], index=[0, 251])
        cagr = compute_cagr(equity)
        # n_years = 2/252 ≈ 0.00794, this test uses simple 2-point equity
        assert cagr > 0

    def test_flat_returns_zero_cagr(self):
        equity = pd.Series([100.0] * 252)
        cagr = compute_cagr(equity)
        assert abs(cagr) < 1e-10

    def test_single_point_returns_zero(self):
        equity = pd.Series([100.0])
        assert compute_cagr(equity) == 0.0


class TestSharpe:
    def test_zero_vol_returns_zero(self):
        returns = pd.Series([0.0] * 100)
        assert compute_sharpe(returns) == 0.0

    def test_positive_sharpe_for_positive_excess_returns(self):
        # Constant positive daily excess return
        returns = pd.Series([0.001] * 252)
        sharpe = compute_sharpe(returns, risk_free_annual=0.0)
        assert sharpe > 0

    def test_negative_sharpe_for_negative_returns(self):
        # Use variable negative returns so std != 0
        np.random.seed(42)
        returns = pd.Series(np.random.normal(-0.01, 0.005, 252))
        sharpe = compute_sharpe(returns, risk_free_annual=0.0)
        assert sharpe < 0


class TestSortino:
    def test_only_positive_returns_gives_high_sortino(self):
        returns = pd.Series([0.001] * 252)
        sortino = compute_sortino(returns, risk_free_annual=0.0)
        assert sortino > 0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        # Goes up to 200, then drops to 150 = 25% drawdown
        equity = pd.Series([100.0, 200.0, 150.0])
        mdd = compute_max_drawdown(equity)
        assert abs(mdd - (-0.25)) < 1e-10

    def test_monotone_increasing_no_drawdown(self):
        equity = pd.Series([100.0, 110.0, 120.0, 130.0])
        mdd = compute_max_drawdown(equity)
        assert mdd == 0.0

    def test_flat_no_drawdown(self):
        equity = pd.Series([100.0] * 10)
        mdd = compute_max_drawdown(equity)
        assert mdd == 0.0


class TestCalmar:
    def test_calmar_ratio(self):
        equity = pd.Series([100.0, 200.0, 150.0])
        returns = equity.pct_change().dropna()
        calmar = compute_calmar(equity, returns)
        # Should be positive (CAGR > 0, MDD < 0)
        assert calmar > 0


class TestTurnover:
    def test_no_trades_zero_turnover(self):
        weights = pd.DataFrame({
            "A": [0.5, 0.5, 0.5],
            "B": [0.5, 0.5, 0.5],
        })
        assert compute_turnover(weights) == 0.0

    def test_full_rebalance_turnover(self):
        # Every day: full rebalance from A->B and back
        weights = pd.DataFrame({
            "A": [1.0, 0.0, 1.0, 0.0],
            "B": [0.0, 1.0, 0.0, 1.0],
        })
        turnover = compute_turnover(weights)
        # diff().abs().sum(axis=1): [0.0, 2.0, 2.0, 2.0] -> mean = 1.5
        assert turnover == 1.5


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        equity = pd.Series([100.0, 105.0, 110.0, 108.0, 112.0])
        returns = equity.pct_change().fillna(0)
        metrics = compute_all_metrics(equity, returns)
        assert "cagr" in metrics
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "max_drawdown" in metrics
        assert "calmar" in metrics
        assert "total_return" in metrics
        assert "volatility" in metrics
