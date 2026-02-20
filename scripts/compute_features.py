"""Script: Compute features from cleaned data."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.logging import setup_logging
from quant_lab.utils.seed import set_global_seed
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_global_seed(cfg.project.seed)

    import structlog

    logger = structlog.get_logger(__name__)
    logger.info("=== Feature Computation Pipeline ===")

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # Load cleaned data
    clean_store = ParquetStore(data_dir / "cleaned")
    df = clean_store.load(f"{universe_name}_cleaned")
    logger.info("data_loaded", rows=len(df), tickers=df["ticker"].nunique())

    # Compute features
    windows = OmegaConf.to_container(cfg.features.windows, resolve=True)
    normalization = OmegaConf.to_container(cfg.features.normalization, resolve=True)

    engine = FeatureEngine(
        enabled_features=list(cfg.features.enabled_features),
        windows=windows,
        normalization=normalization,
    )
    df = engine.compute(df)

    # Normalize features
    df = engine.normalize(df)

    feature_cols = engine.get_feature_columns(df)
    logger.info("features_computed", num_features=len(feature_cols), columns=feature_cols)

    # Save features
    feature_store = FeatureStore(data_dir / "features")
    feature_store.save_features(df, f"{universe_name}_features")

    logger.info("=== Feature Computation Complete ===")


if __name__ == "__main__":
    main()
