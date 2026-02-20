"""Detect market regimes from embeddings or return/volatility data.

Usage:
    python scripts/detect_regimes.py
    python scripts/detect_regimes.py regime.method=hmm regime.n_regimes=4
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.regime.detector import RegimeDetector, DetectorConfig
from quant_lab.regime.hmm import HMMConfig

logger = structlog.get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run regime detection and save results."""
    set_global_seed(cfg.project.seed)

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # Determine method from config (default to kmeans)
    method = cfg.get("regime", {}).get("method", "kmeans")
    n_regimes = cfg.get("regime", {}).get("n_regimes", 4)

    logger.info("regime_detection_start", method=method, n_regimes=n_regimes)

    # Load cleaned data for returns/volatility
    clean_store = ParquetStore(data_dir / "cleaned")
    if not clean_store.exists(f"{universe_name}_cleaned"):
        logger.error("No cleaned data. Run 'python scripts/ingest_data.py' first.")
        return

    clean_df = clean_store.load(f"{universe_name}_cleaned")

    # Compute returns and volatility per ticker
    clean_df = clean_df.sort_values(["ticker", "date"])
    clean_df["return"] = clean_df.groupby("ticker")["adj_close"].transform(
        lambda s: s.pct_change()
    )
    clean_df["volatility"] = clean_df.groupby("ticker")["return"].transform(
        lambda s: s.rolling(21, min_periods=5).std()
    )
    clean_df = clean_df.dropna(subset=["return", "volatility"])

    # Aggregate across tickers (market-level regime)
    market_df = clean_df.groupby("date").agg(
        mean_return=("return", "mean"),
        mean_volatility=("volatility", "mean"),
    ).reset_index().sort_values("date")

    returns = market_df["mean_return"].values
    volatility = market_df["mean_volatility"].values
    dates = market_df["date"].values

    # Try to load embeddings for clustering methods
    embeddings = None
    emb_store = ParquetStore(data_dir / "embeddings")
    if method != "hmm" and emb_store.exists("embeddings"):
        emb_df = emb_store.load("embeddings")
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        if emb_cols:
            embeddings = emb_df[emb_cols].values
            logger.info("embeddings_loaded", shape=embeddings.shape)

    # Build detector config
    hmm_config = None
    if method == "hmm":
        hmm_config = HMMConfig(n_regimes=n_regimes, covariance_type="diag")

    detector_config = DetectorConfig(
        method=method,
        n_regimes=n_regimes,
        hmm_config=hmm_config,
    )
    detector = RegimeDetector(detector_config)

    # Fit
    if method == "hmm":
        result = detector.fit(returns=returns, volatility=volatility)
    elif embeddings is not None:
        result = detector.fit(embeddings=embeddings, returns=returns, volatility=volatility)
    else:
        # Fall back to return/vol feature matrix for clustering
        logger.warning("no_embeddings_found, using return/vol features for clustering")
        feature_matrix = np.column_stack([returns, volatility])
        result = detector.fit(embeddings=feature_matrix, returns=returns, volatility=volatility)

    labels = result["labels"]
    summary = result["summary"]

    # Save results
    output_dir = Path("outputs/regimes")
    output_dir.mkdir(parents=True, exist_ok=True)

    regime_df = pd.DataFrame({
        "date": dates[:len(labels)],
        "regime_label": labels,
    })
    regime_df.to_parquet(output_dir / "regime_labels.parquet", index=False)

    if len(summary) > 0:
        summary.to_parquet(output_dir / "regime_summary.parquet", index=False)

    # Log to tracker
    try:
        from quant_lab.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name=cfg.experiment.tracking.get("experiment_name", "regime_detection"),
            tracking_uri=cfg.experiment.mlflow.get("tracking_uri", "mlruns"),
        )
        tracker.start_run(run_name=f"regime_{method}_{n_regimes}")
        tracker.log_params({"method": method, "n_regimes": n_regimes})
        tracker.log_metrics({"num_unique_regimes": len(set(labels[labels >= 0]))})
        tracker.end_run()
    except Exception as e:
        logger.warning("mlflow_failed", error=str(e))

    # Print summary
    print("\n" + "=" * 60)
    print("REGIME DETECTION COMPLETE")
    print("=" * 60)
    print(f"  Method: {method}")
    print(f"  Regimes found: {len(set(labels[labels >= 0]))}")
    print(f"  Samples: {len(labels)}")
    if len(summary) > 0:
        print(f"\n{summary.to_string(index=False)}")
    print(f"\n  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
