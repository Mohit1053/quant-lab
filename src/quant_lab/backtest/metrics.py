"""Backtest performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_cagr(equity_curve: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_years = len(equity_curve) / 252.0
    if n_years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1.0 / n_years) - 1.0)


def compute_sharpe(returns: pd.Series, risk_free_annual: float = 0.05) -> float:
    """Annualized Sharpe Ratio."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    return float(excess.mean() / excess.std() * np.sqrt(252))


def compute_sortino(returns: pd.Series, risk_free_annual: float = 0.05) -> float:
    """Annualized Sortino Ratio (downside risk only)."""
    if len(returns) < 2:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 999.0 if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a negative fraction (e.g., -0.15 = -15%)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def compute_calmar(equity_curve: pd.Series, returns: pd.Series) -> float:
    """Calmar Ratio = CAGR / |Max Drawdown|."""
    cagr = compute_cagr(equity_curve)
    mdd = compute_max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return float(cagr / abs(mdd))


def compute_turnover(weights_history: pd.DataFrame) -> float:
    """Average daily turnover as sum of absolute weight changes."""
    if len(weights_history) < 2:
        return 0.0
    daily_turnover = weights_history.diff().abs().sum(axis=1)
    return float(daily_turnover.mean())


def compute_annual_turnover(weights_history: pd.DataFrame) -> float:
    """Annualized turnover."""
    return compute_turnover(weights_history) * 252


def compute_all_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    weights_history: pd.DataFrame | None = None,
    risk_free_annual: float = 0.05,
) -> dict[str, float]:
    """Compute all backtest metrics and return as a dict."""
    metrics = {
        "cagr": compute_cagr(equity_curve),
        "sharpe": compute_sharpe(returns, risk_free_annual),
        "sortino": compute_sortino(returns, risk_free_annual),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "calmar": compute_calmar(equity_curve, returns),
        "total_return": float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
        if len(equity_curve) > 0
        else 0.0,
        "volatility": float(returns.std() * np.sqrt(252)),
    }

    if weights_history is not None:
        metrics["avg_daily_turnover"] = compute_turnover(weights_history)
        metrics["annual_turnover"] = compute_annual_turnover(weights_history)

    return metrics
