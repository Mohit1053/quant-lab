"""Script: Download and clean market data."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.logging import setup_logging
from quant_lab.utils.seed import set_global_seed
from quant_lab.data.sources.yfinance_source import YFinanceSource
from quant_lab.data.cleaning.pipeline import CleaningPipeline, CleaningConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.data.universe import get_universe


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    setup_logging()
    set_global_seed(cfg.project.seed)

    import structlog

    logger = structlog.get_logger(__name__)
    logger.info("=== Data Ingestion Pipeline ===")

    # Get universe
    universe = get_universe(cfg.data.universe.name)
    tickers = cfg.data.universe.tickers or universe.tickers
    logger.info("universe", name=universe.name, num_tickers=len(tickers))

    # Fetch data
    source = YFinanceSource()
    raw_df = source.fetch(
        tickers=list(tickers),
        start=cfg.data.date_range.start,
        end=cfg.data.date_range.end,
    )

    # Save raw data
    data_dir = Path(cfg.paths.data_dir)
    raw_store = ParquetStore(data_dir / "raw")
    raw_store.save(raw_df, f"{universe.name}_raw")

    # Clean data
    cleaning_cfg = CleaningConfig(
        max_missing_pct=cfg.data.cleaning.max_missing_pct,
        ffill_limit=cfg.data.cleaning.ffill_limit,
        outlier_sigma=cfg.data.cleaning.outlier_sigma,
        min_history_days=cfg.data.cleaning.min_history_days,
    )
    pipeline = CleaningPipeline(cleaning_cfg)
    clean_df = pipeline.run(raw_df)

    # Save cleaned data
    clean_store = ParquetStore(data_dir / "cleaned")
    clean_store.save(clean_df, f"{universe.name}_cleaned")

    logger.info("=== Ingestion Complete ===")


if __name__ == "__main__":
    main()
