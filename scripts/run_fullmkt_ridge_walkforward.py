"""Run Ridge walk-forward on the full Indian market (1093 stocks).

No Hydra dependency — uses hardcoded config matching the standard pipeline.
Saves results to outputs/walk_forward/ridge_fullmkt/.

Usage:
    python scripts/run_fullmkt_ridge_walkforward.py
    python scripts/run_fullmkt_ridge_walkforward.py --start-fold 5   # resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.walk_forward import (
    WalkForwardEngine,
    WalkForwardConfig,
    WindowType,
)
from quant_lab.backtest.engine import BacktestConfig


UNIVERSE = "indian_market"
OUTPUT_DIR = Path("outputs/walk_forward/ridge_fullmkt")


def make_ridge_factory(target_col: str = "log_return_1d"):
    def factory(split, feature_df, feature_cols):
        datasets = create_flat_datasets(
            feature_df, feature_cols, split, target_col=target_col
        )
        X_train, y_train, _ = datasets["train"]
        X_test, y_test, meta_test = datasets["test"]

        model = RidgeBaseline(alpha=1.0)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        signals = meta_test.copy()
        signals["signal"] = preds
        return model, signals

    return factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-fold", type=int, default=0, help="Resume from fold N")
    args = parser.parse_args()

    set_global_seed(42)

    # Load features
    feature_store = FeatureStore("data/features")
    feature_name = f"{UNIVERSE}_features"
    if not feature_store.has_features(feature_name):
        print(f"ERROR: {feature_name} not found. Run run_indian_market_pipeline.py first.")
        sys.exit(1)

    df = feature_store.load_features(feature_name)

    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in df.columns if c not in base_cols]
    prices_df = df[["date", "ticker", "adj_close"]].copy()

    # Walk-forward config
    wf_config = WalkForwardConfig(
        window_type=WindowType.EXPANDING,
        train_days=756,
        val_days=126,
        test_days=126,
        step_days=126,
        min_train_days=504,
    )

    # Backtest: top_n=20 for broader universe
    backtest_config = BacktestConfig(
        initial_capital=1_000_000,
        rebalance_frequency=5,
        top_n=20,
        risk_free_rate=0.05,
    )

    print("=" * 60)
    print("  FULL MARKET RIDGE WALK-FORWARD")
    print("=" * 60)
    print(f"  Tickers:    {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Features:   {len(feature_cols)}")
    print(f"  Top-N:      {backtest_config.top_n}")
    if args.start_fold > 0:
        print(f"  Resuming from fold {args.start_fold}")
    print("=" * 60)

    factory = make_ridge_factory()

    start = time.time()
    engine = WalkForwardEngine(wf_config, backtest_config)
    result = engine.run(
        df, feature_cols, prices_df, factory,
        start_fold=args.start_fold,
    )
    elapsed = time.time() - start

    # Print results
    print(f"\nWalk-forward done in {elapsed/60:.1f} min ({len(result.fold_results)} folds)")
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    for metric, value in result.aggregate_metrics.items():
        if "return" in metric or "cagr" in metric or "drawdown" in metric:
            print(f"  {metric:25s}: {value:>10.2%}")
        else:
            print(f"  {metric:25s}: {value:>10.4f}")

    print(f"\nPer-fold breakdown:")
    display_cols = ["fold", "test_start", "test_end", "sharpe", "total_return", "max_drawdown"]
    available = [c for c in display_cols if c in result.per_fold_metrics.columns]
    print(result.per_fold_metrics[available].to_string(index=False))

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "per_fold_metrics.csv"

    if args.start_fold > 0 and csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["fold"] < args.start_fold]
        merged = pd.concat([existing, result.per_fold_metrics], ignore_index=True)
        merged.to_csv(csv_path, index=False)
        print(f"\nMerged {len(existing)} existing + {len(result.per_fold_metrics)} new folds")
    else:
        result.per_fold_metrics.to_csv(csv_path, index=False)

    result.aggregate_equity.to_frame("equity").to_parquet(OUTPUT_DIR / "aggregate_equity.parquet")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
