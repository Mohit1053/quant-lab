"""Train RL portfolio allocation agent.

Usage:
    python scripts/train_rl.py
    python scripts/train_rl.py rl=sac
    python scripts/train_rl.py rl.agent.learning_rate=1e-4
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
from quant_lab.utils.device import get_device, get_device_info
from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.features.engine import FeatureEngine
from quant_lab.rl.environments.portfolio_env import PortfolioEnvConfig
from quant_lab.rl.environments.reward import RewardConfig
from quant_lab.rl.training import train_rl, RLTrainingConfig

logger = structlog.get_logger(__name__)


def _build_feature_tensor(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    split_start: str,
    split_end: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (T, N, D) features and (T, N) returns from feature DataFrame.

    Groups by date to create true multi-asset time steps.
    """
    df = feature_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] > split_start) & (df["date"] <= split_end)]

    # Ensure target column
    if "log_return_1d" not in df.columns:
        df["log_return_1d"] = df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    tickers = sorted(df["ticker"].unique())
    dates = sorted(df["date"].unique())
    n_assets = len(tickers)
    n_dates = len(dates)
    n_feats = len(feature_cols)

    # Pivot features and returns to (T, N, D) and (T, N)
    features_3d = np.zeros((n_dates, n_assets, n_feats), dtype=np.float32)
    returns_2d = np.zeros((n_dates, n_assets), dtype=np.float32)

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    date_to_idx = {d: i for i, d in enumerate(dates)}

    for _, row in df.iterrows():
        t_idx = date_to_idx[row["date"]]
        a_idx = ticker_to_idx[row["ticker"]]
        features_3d[t_idx, a_idx, :] = row[feature_cols].values.astype(np.float32)
        ret = row.get("log_return_1d", 0.0)
        returns_2d[t_idx, a_idx] = 0.0 if pd.isna(ret) else float(ret)

    # Replace NaN with 0
    features_3d = np.nan_to_num(features_3d, nan=0.0)

    return features_3d, returns_2d


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train RL agent for portfolio allocation."""
    set_global_seed(cfg.project.seed)
    device = get_device()
    device_info = get_device_info()
    logger.info("device_info", **device_info)
    logger.info("config", rl=OmegaConf.to_yaml(cfg.rl))

    # Load feature data
    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name
    store = ParquetStore(base_dir=str(data_dir / "features"))
    feature_name = f"{universe_name}_features"
    if not store.exists(feature_name):
        logger.error("No feature data. Run 'python scripts/compute_features.py' first.")
        return

    feature_df = store.load(feature_name)
    logger.info("data_loaded", rows=len(feature_df), tickers=feature_df["ticker"].nunique())

    # Detect feature columns
    engine = FeatureEngine(
        enabled_features=list(cfg.features.enabled_features),
        windows={k: list(v) for k, v in cfg.features.windows.items()},
    )
    feature_cols = engine.get_feature_columns(feature_df)
    logger.info("features_detected", num_features=len(feature_cols))

    # Create DataModule
    split = TemporalSplit(
        train_end=cfg.data.date_range.train_end,
        val_end=cfg.data.date_range.val_end,
    )
    dm_config = DataModuleConfig(
        sequence_length=cfg.model.input.sequence_length,
        target_col="log_return_1d",
        batch_size=cfg.rl.agent.get("batch_size", 256),
        num_workers=0,
    )
    dm = QuantDataModule(feature_df, feature_cols, split, dm_config)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if train_loader is None:
        logger.error("No training data. Check split dates.")
        return

    # Build feature tensors from DataFrame directly (multi-asset)
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in feature_df.columns if c not in base_cols]

    train_features, train_returns = _build_feature_tensor(
        feature_df, feature_cols,
        split_start="1900-01-01",
        split_end=cfg.data.date_range.train_end,
    )
    logger.info(
        "train_data",
        features_shape=train_features.shape,
        returns_shape=train_returns.shape,
    )

    val_features, val_returns = None, None
    val_end = cfg.data.date_range.val_end
    train_end = cfg.data.date_range.train_end
    val_features, val_returns = _build_feature_tensor(
        feature_df, feature_cols,
        split_start=train_end,
        split_end=val_end,
    )
    if val_features.shape[0] == 0:
        val_features, val_returns = None, None
    else:
        logger.info("val_data", features_shape=val_features.shape)

    # Build configs from Hydra
    rl_cfg = cfg.rl
    env_config = PortfolioEnvConfig(
        initial_cash=rl_cfg.environment.get("initial_cash", 1_000_000),
        max_weight=rl_cfg.environment.get("max_weight", 0.20),
        rebalance_frequency=rl_cfg.environment.get("rebalance_frequency", 5),
    )
    reward_config = RewardConfig(
        lambda_mdd=rl_cfg.reward.get("lambda_mdd", 0.5),
        lambda_turnover=rl_cfg.reward.get("lambda_turnover", 0.01),
        commission_bps=rl_cfg.reward.get("commission_bps", 10.0),
        slippage_bps=rl_cfg.reward.get("slippage_bps", 5.0),
        spread_bps=rl_cfg.reward.get("spread_bps", 5.0),
    )
    training_config = RLTrainingConfig(
        algorithm=rl_cfg.algorithm,
        total_timesteps=rl_cfg.training.get("total_timesteps", 100_000),
        eval_freq=rl_cfg.training.get("eval_freq", 10_000),
        n_eval_episodes=rl_cfg.training.get("n_eval_episodes", 5),
        checkpoint_dir=f"outputs/models/rl/{rl_cfg.algorithm}",
    )

    # Setup tracker
    tracker = None
    try:
        from quant_lab.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name=cfg.experiment.tracking.get("experiment_name", "rl_portfolio"),
            tracking_uri=cfg.experiment.mlflow.get("tracking_uri", "mlruns"),
        )
        tracker.start_run(run_name=f"rl_{rl_cfg.algorithm}")
        tracker.log_config(OmegaConf.to_container(cfg.rl, resolve=True))
    except Exception as e:
        logger.warning("mlflow_setup_failed", error=str(e))

    # Train
    result = train_rl(
        train_features=train_features,
        train_returns=train_returns,
        val_features=val_features,
        val_returns=val_returns,
        config=training_config,
        env_config=env_config,
        reward_config=reward_config,
        device="auto",
    )

    # Log metrics
    if tracker is not None:
        tracker.log_metrics({f"train_{k}": v for k, v in result["train_metrics"].items()})
        if "val_metrics" in result:
            tracker.log_metrics({f"val_{k}": v for k, v in result["val_metrics"].items()})
        tracker.end_run()

    # Print summary
    print("\n" + "=" * 60)
    print(f"RL TRAINING COMPLETE ({rl_cfg.algorithm.upper()})")
    print("=" * 60)
    for k, v in result["train_metrics"].items():
        print(f"  Train {k}: {v:.4f}")
    if "val_metrics" in result:
        for k, v in result["val_metrics"].items():
            print(f"  Val   {k}: {v:.4f}")
    print(f"  Agent saved to: outputs/models/rl/{rl_cfg.algorithm}")
    print("=" * 60)


if __name__ == "__main__":
    main()
