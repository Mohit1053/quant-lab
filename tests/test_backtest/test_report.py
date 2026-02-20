"""Tests for HTML backtest report generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_lab.backtest.report import BacktestReport, ReportConfig


class TestBacktestReport:
    def test_generate_basic_report(self, tmp_path):
        config = ReportConfig(
            title="Test Report",
            output_dir=str(tmp_path / "reports"),
        )
        report = BacktestReport(config)

        portfolio_values = np.cumsum(np.random.randn(100) * 0.01) + 100
        portfolio_values = np.maximum(portfolio_values, 50)  # Ensure positive

        path = report.generate(
            portfolio_values=portfolio_values,
            metrics={"cagr": 0.12, "sharpe": 1.5, "max_drawdown": -0.15},
        )

        assert path.endswith(".html")
        with open(path) as f:
            content = f.read()
        assert "Test Report" in content
        assert "12.00%" in content  # CAGR
        assert "1.50" in content  # Sharpe

    def test_generate_with_dates(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path / "reports"))
        report = BacktestReport(config)

        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        values = np.cumsum(np.random.randn(50) * 0.01) + 100

        path = report.generate(portfolio_values=values, dates=dates)
        assert path.endswith(".html")

    def test_generate_with_benchmark(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path / "reports"))
        report = BacktestReport(config)

        values = np.cumsum(np.random.randn(50) * 0.01) + 100
        benchmark = np.cumsum(np.random.randn(50) * 0.008) + 100

        path = report.generate(portfolio_values=values, benchmark_values=benchmark)
        with open(path) as f:
            content = f.read()
        assert "Benchmark" in content

    def test_generate_with_weights(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path / "reports"))
        report = BacktestReport(config)

        values = np.cumsum(np.random.randn(50) * 0.01) + 100
        weights = np.random.dirichlet(np.ones(5), size=50)

        path = report.generate(portfolio_values=values, weights_history=weights)
        with open(path) as f:
            content = f.read()
        assert "Portfolio Allocation" in content

    def test_generate_with_regime_summary(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path / "reports"))
        report = BacktestReport(config)

        values = np.cumsum(np.random.randn(50) * 0.01) + 100
        regime_summary = pd.DataFrame({
            "label": ["Bull", "Bear"],
            "mean_return": ["0.01", "-0.01"],
            "frequency": ["60%", "40%"],
        })

        path = report.generate(
            portfolio_values=values,
            regime_summary=regime_summary,
        )
        with open(path) as f:
            content = f.read()
        assert "Regime Analysis" in content

    def test_generate_with_regime_labels(self, tmp_path):
        config = ReportConfig(output_dir=str(tmp_path / "reports"))
        report = BacktestReport(config)

        values = np.cumsum(np.random.randn(50) * 0.01) + 100
        regime_labels = np.random.choice([0, 1, 2], size=50)

        path = report.generate(
            portfolio_values=values,
            regime_labels=regime_labels,
        )
        with open(path) as f:
            content = f.read()
        assert "Regime" in content
