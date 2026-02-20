"""Run the full Indian market pipeline: ingest, clean, features for all NSE EQ-series stocks.

This is a long-running script (~30-60 min for ingestion, ~10 min for features).
Downloads 2100+ stocks in batches of 100 from yfinance, applies liquidity
filters, then computes features.

Usage:
    python scripts/run_indian_market_pipeline.py
    python scripts/run_indian_market_pipeline.py --skip-ingest  # reuse cached raw data
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.logging import setup_logging
from quant_lab.utils.seed import set_global_seed
from quant_lab.data.sources.yfinance_source import YFinanceSource
from quant_lab.data.cleaning.pipeline import CleaningPipeline, CleaningConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.data.universe import load_all_nse_tickers, get_universe
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore


UNIVERSE_NAME = "indian_market"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def main():
    parser = argparse.ArgumentParser(description="Full Indian market pipeline")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip data download, use cached raw/cleaned data")
    parser.add_argument("--skip-features", action="store_true", help="Skip feature computation")
    args = parser.parse_args()

    setup_logging()
    set_global_seed(42)

    import structlog
    logger = structlog.get_logger(__name__)

    data_dir = Path("data")
    raw_store = ParquetStore(data_dir / "raw")
    clean_store = ParquetStore(data_dir / "cleaned")
    feature_store = FeatureStore(str(data_dir / "features"))

    # ── Step 1: Load tickers ──
    print("=" * 60)
    print("  Full Indian Market Pipeline")
    print("=" * 60)

    cache_dir = str(data_dir / "universe_cache")
    tickers = load_all_nse_tickers(cache_dir=cache_dir)
    print(f"\nNSE EQ-series tickers: {len(tickers)}")

    # ── Step 2: Data Ingestion ──
    if not args.skip_ingest:
        if clean_store.exists(f"{UNIVERSE_NAME}_cleaned"):
            print("\nCleaned data already cached. Use --skip-ingest or delete cache to re-download.")
            df = clean_store.load(f"{UNIVERSE_NAME}_cleaned")
        else:
            print(f"\nDownloading {len(tickers)} stocks ({START_DATE} to {END_DATE})...")
            print(f"  Batch size: 100 | Estimated batches: {(len(tickers) + 99) // 100}")
            print(f"  Estimated time: ~30-60 min\n")

            source = YFinanceSource(batch_size=100)
            start = time.time()
            raw_df = source.fetch(tickers=tickers, start=START_DATE, end=END_DATE)
            elapsed = time.time() - start

            print(f"\nIngestion done in {elapsed/60:.1f} min")
            print(f"  Raw rows: {len(raw_df):,}")
            print(f"  Tickers with data: {raw_df['ticker'].nunique()}")

            # Save raw data
            raw_store.save(raw_df, f"{UNIVERSE_NAME}_raw")
            print(f"  Saved to {data_dir / 'raw' / f'{UNIVERSE_NAME}_raw.parquet'}")

            # ── Step 3: Clean with liquidity filters ──
            print("\nCleaning with liquidity filters...")
            cleaning_cfg = CleaningConfig(
                max_missing_pct=0.20,
                ffill_limit=5,
                outlier_sigma=10.0,
                min_history_days=504,         # 2 years minimum history
                min_avg_daily_volume=10_000,   # Avg 10K shares/day
                min_median_price=5.0,          # No penny stocks
                min_trading_days_pct=0.50,     # Must trade >=50% of days
            )
            pipeline = CleaningPipeline(cleaning_cfg)
            df = pipeline.run(raw_df)

            clean_store.save(df, f"{UNIVERSE_NAME}_cleaned")
            print(f"\nCleaned data: {len(df):,} rows, {df['ticker'].nunique()} tickers")
            print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
            print(f"  Saved to {data_dir / 'cleaned' / f'{UNIVERSE_NAME}_cleaned.parquet'}")
    else:
        if clean_store.exists(f"{UNIVERSE_NAME}_cleaned"):
            df = clean_store.load(f"{UNIVERSE_NAME}_cleaned")
            print(f"\nLoaded cached cleaned data: {len(df):,} rows, {df['ticker'].nunique()} tickers")
        else:
            print("\nERROR: No cached cleaned data found. Run without --skip-ingest first.")
            sys.exit(1)

    # ── Step 4: Feature Computation ──
    if not args.skip_features:
        feature_name = f"{UNIVERSE_NAME}_features"
        if feature_store.has_features(feature_name):
            print(f"\nFeatures already computed: {feature_name}")
        else:
            print("\nComputing features...")
            start = time.time()

            enabled_features = ["log_returns", "realized_volatility", "momentum", "max_drawdown"]
            windows = {"short": [1, 5], "medium": [21], "long": [63]}
            normalization = {"method": "rolling_zscore", "lookback": 252}

            engine = FeatureEngine(
                enabled_features=enabled_features,
                windows=windows,
                normalization=normalization,
            )
            features_df = engine.compute(df)
            features_df = engine.normalize(features_df)
            elapsed = time.time() - start

            feature_store.save_features(features_df, feature_name)
            print(f"\nFeatures done in {elapsed/60:.1f} min")
            print(f"  Feature rows: {len(features_df):,}")
            print(f"  Feature tickers: {features_df['ticker'].nunique()}")
            base_cols = {'date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'adj_close'}
            feat_cols = [c for c in features_df.columns if c not in base_cols]
            print(f"  Features per stock: {len(feat_cols)}")
            print(f"  Saved to {data_dir / 'features' / f'{feature_name}.parquet'}")
    else:
        print("\nSkipping feature computation.")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)

    sizes = {}
    for path in [
        data_dir / "raw" / f"{UNIVERSE_NAME}_raw.parquet",
        data_dir / "cleaned" / f"{UNIVERSE_NAME}_cleaned.parquet",
        data_dir / "features" / f"{UNIVERSE_NAME}_features.parquet",
    ]:
        if path.exists():
            sizes[path.name] = path.stat().st_size / 1e6
            print(f"  {path.name}: {sizes[path.name]:.1f} MB")
        else:
            print(f"  {path.name}: not yet created")


if __name__ == "__main__":
    main()
