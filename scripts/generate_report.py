"""Generate HTML backtest report from saved backtest results.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py report.title="Custom Report"
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.backtest.report import BacktestReport, ReportConfig

logger = structlog.get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate HTML report from backtest artifacts."""
    set_global_seed(cfg.project.seed)

    output_dir = Path("outputs/backtests")

    # Load backtest results
    equity_path = output_dir / "equity_curve.parquet"
    metrics_path = output_dir / "metrics.parquet"

    if equity_path.exists():
        equity_df = pd.read_parquet(equity_path)
        portfolio_values = equity_df["equity"].values
        dates = pd.to_datetime(equity_df.index) if "date" not in equity_df.columns else pd.to_datetime(equity_df["date"])
    else:
        # Try to reconstruct from any available data
        logger.warning("no_saved_equity_curve, generating sample report")
        np.random.seed(42)
        portfolio_values = np.cumsum(np.random.randn(252) * 0.01) + 100
        portfolio_values = np.maximum(portfolio_values, 50)
        dates = pd.bdate_range("2023-01-01", periods=252)

    # Load metrics
    metrics = {}
    if metrics_path.exists():
        metrics_df = pd.read_parquet(metrics_path)
        metrics = metrics_df.iloc[0].to_dict()
    else:
        # Compute basic metrics from equity curve
        from quant_lab.backtest.metrics import compute_all_metrics

        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns_series = pd.Series(returns, index=dates[1:] if len(dates) > len(returns) else range(len(returns)))
        equity_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
        metrics = compute_all_metrics(equity_series, returns_series)

    # Load weights if available
    weights_history = None
    weights_path = output_dir / "weights_history.parquet"
    if weights_path.exists():
        weights_df = pd.read_parquet(weights_path)
        weights_history = weights_df.values

    # Load benchmark if available
    benchmark_values = None
    benchmark_path = output_dir / "benchmark.parquet"
    if benchmark_path.exists():
        benchmark_df = pd.read_parquet(benchmark_path)
        benchmark_values = benchmark_df.iloc[:, 0].values

    # Load regime data if available
    regime_labels = None
    regime_summary = None
    regime_labels_path = Path("outputs/regimes/regime_labels.parquet")
    regime_summary_path = Path("outputs/regimes/regime_summary.parquet")

    if regime_labels_path.exists():
        regime_df = pd.read_parquet(regime_labels_path)
        regime_labels = regime_df["regime_label"].values[:len(portfolio_values)]

    if regime_summary_path.exists():
        regime_summary = pd.read_parquet(regime_summary_path)

    # Build report
    title = cfg.get("report", {}).get("title", "Backtest Report")
    report_dir = cfg.get("report", {}).get("output_dir", "outputs/backtests")

    report_config = ReportConfig(
        title=title,
        output_dir=report_dir,
    )
    report = BacktestReport(report_config)

    report_path = report.generate(
        portfolio_values=portfolio_values,
        dates=dates[:len(portfolio_values)],
        metrics=metrics,
        weights_history=weights_history,
        regime_labels=regime_labels,
        regime_summary=regime_summary,
        benchmark_values=benchmark_values,
    )

    print("\n" + "=" * 60)
    print("REPORT GENERATED")
    print("=" * 60)
    print(f"  Path: {report_path}")
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
