"""Run walk-forward analysis with any model type.

Usage:
    python scripts/run_walk_forward.py                              # Ridge, expanding
    python scripts/run_walk_forward.py walk_forward.window_type=rolling
    python scripts/run_walk_forward.py walk_forward.model_type=transformer
    python scripts/run_walk_forward.py walk_forward.step_days=63
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
import structlog
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.data.datasets import TemporalSplit, create_flat_datasets
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.linear_baseline import RidgeBaseline
from quant_lab.backtest.walk_forward import (
    WalkForwardEngine,
    WalkForwardConfig,
    WindowType,
)
from quant_lab.backtest.engine import BacktestConfig
from quant_lab.backtest.execution import ExecutionModel

logger = structlog.get_logger(__name__)


def make_ridge_factory(target_col: str = "log_return_1d"):
    """Create a model factory for Ridge baseline."""

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


def make_transformer_factory(cfg: DictConfig, device: torch.device):
    """Create a model factory for Transformer."""

    def factory(split, feature_df, feature_cols):
        from quant_lab.models.transformer.model import (
            TransformerForecaster,
            TransformerConfig,
            MultiTaskLoss,
        )
        from quant_lab.training.trainer import Trainer, TrainerConfig

        dm = QuantDataModule(
            feature_df,
            feature_cols,
            split,
            DataModuleConfig(
                sequence_length=cfg.model.input.sequence_length,
                target_col="log_return_1d",
                batch_size=cfg.model.training.batch_size,
                num_workers=0,
            ),
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        if train_loader is None:
            return None, pd.DataFrame()

        model_cfg = TransformerConfig(
            num_features=dm.num_features,
            d_model=cfg.model.architecture.d_model,
            nhead=cfg.model.architecture.nhead,
            num_encoder_layers=cfg.model.architecture.num_encoder_layers,
            dim_feedforward=cfg.model.architecture.dim_feedforward,
            dropout=cfg.model.architecture.dropout,
            direction_weight=cfg.model.loss.direction_weight,
            volatility_weight=cfg.model.loss.volatility_weight,
        )
        model = TransformerForecaster(model_cfg)
        loss_fn = MultiTaskLoss(model_cfg)

        wf_cfg = cfg.get("walk_forward", {})
        trainer_config = TrainerConfig(
            epochs=wf_cfg.get("epochs_per_fold", 50),
            learning_rate=cfg.model.training.learning_rate,
            weight_decay=cfg.model.training.weight_decay,
            warmup_steps=cfg.model.training.warmup_steps,
            patience=wf_cfg.get("patience_per_fold", 8),
            mixed_precision=cfg.project.mixed_precision,
            checkpoint_dir="outputs/walk_forward/transformer",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device)
        trainer.fit(train_loader, val_loader)

        # Generate test predictions
        test_loader = dm.test_dataloader()
        if test_loader is None:
            return model, pd.DataFrame()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                preds = model.predict_returns(x)
                all_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(all_preds)

        datasets = create_flat_datasets(
            feature_df, feature_cols, split, target_col="log_return_1d"
        )
        _, _, meta_test = datasets["test"]
        meta_test = meta_test.iloc[-len(test_preds):]

        signals = meta_test.copy()
        signals["signal"] = test_preds
        return model, signals

    return factory


def make_tft_factory(cfg: DictConfig, device: torch.device):
    """Create a model factory for TFT."""

    def factory(split, feature_df, feature_cols):
        from quant_lab.models.tft.model import TFTForecaster, TFTConfig
        from quant_lab.models.transformer.model import MultiTaskLoss, TransformerConfig
        from quant_lab.training.trainer import Trainer, TrainerConfig

        dm = QuantDataModule(
            feature_df,
            feature_cols,
            split,
            DataModuleConfig(
                sequence_length=cfg.model.input.sequence_length,
                target_col="log_return_1d",
                batch_size=cfg.model.training.batch_size,
                num_workers=0,
            ),
        )
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        if train_loader is None:
            return None, pd.DataFrame()

        tft_cfg = TFTConfig(
            num_features=dm.num_features,
            d_model=cfg.model.architecture.d_model,
            nhead=cfg.model.architecture.nhead,
            num_encoder_layers=cfg.model.architecture.num_encoder_layers,
            lstm_layers=cfg.model.architecture.get("lstm_layers", 1),
            lstm_hidden=cfg.model.architecture.get("lstm_hidden", 128),
            grn_hidden=cfg.model.architecture.get("grn_hidden", 64),
            dropout=cfg.model.architecture.dropout,
            direction_weight=cfg.model.loss.direction_weight,
            volatility_weight=cfg.model.loss.volatility_weight,
        )
        model = TFTForecaster(tft_cfg)

        loss_cfg = TransformerConfig(
            num_features=dm.num_features,
            direction_weight=cfg.model.loss.direction_weight,
            volatility_weight=cfg.model.loss.volatility_weight,
        )
        loss_fn = MultiTaskLoss(loss_cfg)

        wf_cfg = cfg.get("walk_forward", {})
        trainer_config = TrainerConfig(
            epochs=wf_cfg.get("epochs_per_fold", 50),
            learning_rate=cfg.model.training.learning_rate,
            weight_decay=cfg.model.training.weight_decay,
            warmup_steps=cfg.model.training.warmup_steps,
            patience=wf_cfg.get("patience_per_fold", 8),
            mixed_precision=cfg.project.mixed_precision,
            checkpoint_dir="outputs/walk_forward/tft",
        )
        trainer = Trainer(model, loss_fn, trainer_config, device)
        trainer.fit(train_loader, val_loader)

        test_loader = dm.test_dataloader()
        if test_loader is None:
            return model, pd.DataFrame()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                preds = model.predict_returns(x)
                all_preds.append(preds.cpu().numpy())

        test_preds = np.concatenate(all_preds)

        datasets = create_flat_datasets(
            feature_df, feature_cols, split, target_col="log_return_1d"
        )
        _, _, meta_test = datasets["test"]
        meta_test = meta_test.iloc[-len(test_preds):]

        signals = meta_test.copy()
        signals["signal"] = test_preds
        return model, signals

    return factory


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run walk-forward analysis."""
    set_global_seed(cfg.project.seed)
    device = get_device()

    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name

    # Load feature data
    feature_store = FeatureStore(data_dir / "features")
    feature_name = f"{universe_name}_features"
    if not feature_store.has_features(feature_name):
        logger.error("No feature data. Run 'python scripts/compute_features.py' first.")
        return

    feature_df = feature_store.load_features(feature_name)

    # Feature columns
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in feature_df.columns if c not in base_cols]

    target_col = "log_return_1d"
    if target_col not in feature_df.columns:
        feature_df[target_col] = feature_df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    # Prices
    prices_df = feature_df[["date", "ticker", "adj_close"]].copy()

    # Walk-forward config
    wf_cfg = cfg.get("walk_forward", {})
    wf_config = WalkForwardConfig(
        window_type=WindowType(wf_cfg.get("window_type", "expanding")),
        train_days=wf_cfg.get("train_days", 756),
        val_days=wf_cfg.get("val_days", 126),
        test_days=wf_cfg.get("test_days", 126),
        step_days=wf_cfg.get("step_days", 126),
        min_train_days=wf_cfg.get("min_train_days", 504),
    )

    # Backtest config
    backtest_config = BacktestConfig(
        initial_capital=cfg.backtest.portfolio.initial_capital,
        rebalance_frequency=cfg.backtest.portfolio.rebalance_frequency,
        max_position_size=cfg.backtest.portfolio.max_position_size,
        top_n=cfg.backtest.portfolio.top_n,
        risk_free_rate=cfg.backtest.metrics.risk_free_rate,
    )

    # Select model factory
    model_type = wf_cfg.get("model_type", "ridge")
    if model_type == "ridge":
        factory = make_ridge_factory(target_col)
    elif model_type == "transformer":
        factory = make_transformer_factory(cfg, device)
    elif model_type == "tft":
        factory = make_tft_factory(cfg, device)
    else:
        logger.error(f"Unknown model_type: {model_type}")
        return

    # Regime labels
    regime_labels = None
    regime_path = Path("outputs/regimes/regime_labels.parquet")
    if regime_path.exists():
        regime_df = pd.read_parquet(regime_path)
        regime_df["date"] = pd.to_datetime(regime_df["date"])
        regime_labels = regime_df.set_index("date")["regime_label"]

    print("=" * 60)
    print(f"WALK-FORWARD ANALYSIS: {model_type.upper()}")
    print("=" * 60)
    print(f"  Window type:    {wf_config.window_type.value}")
    print(f"  Train days:     {wf_config.train_days}")
    print(f"  Val days:       {wf_config.val_days}")
    print(f"  Test days:      {wf_config.test_days}")
    print(f"  Step days:      {wf_config.step_days}")
    print(f"  Tickers:        {feature_df['ticker'].nunique()}")
    print(f"  Date range:     {feature_df['date'].min()} to {feature_df['date'].max()}")
    print("=" * 60)

    start_fold = int(wf_cfg.get("start_fold", 0))
    if start_fold > 0:
        print(f"  Resuming from fold {start_fold}")

    engine = WalkForwardEngine(wf_config, backtest_config)
    result = engine.run(
        feature_df, feature_cols, prices_df, factory,
        regime_labels=regime_labels,
        start_fold=start_fold,
    )

    # Print results
    print("\n" + "=" * 60)
    print("WALK-FORWARD RESULTS")
    print("=" * 60)

    print(f"\nAggregate ({len(result.fold_results)} folds):")
    for metric, value in result.aggregate_metrics.items():
        if "return" in metric or "cagr" in metric or "drawdown" in metric:
            print(f"  {metric:25s}: {value:>10.2%}")
        else:
            print(f"  {metric:25s}: {value:>10.4f}")

    print(f"\nPer-fold breakdown:")
    display_cols = ["fold", "test_start", "test_end", "sharpe", "total_return", "max_drawdown"]
    available = [c for c in display_cols if c in result.per_fold_metrics.columns]
    print(result.per_fold_metrics[available].to_string(index=False))
    print("=" * 60)

    # Save results
    output_dir = Path("outputs/walk_forward") / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # If resuming, merge with existing results
    csv_path = output_dir / "per_fold_metrics.csv"
    if start_fold > 0 and csv_path.exists():
        existing = pd.read_csv(csv_path)
        existing = existing[existing["fold"] < start_fold]
        merged = pd.concat([existing, result.per_fold_metrics], ignore_index=True)
        merged.to_csv(csv_path, index=False)
        print(f"\nMerged {len(existing)} existing + {len(result.per_fold_metrics)} new folds")
    else:
        result.per_fold_metrics.to_csv(csv_path, index=False)

    result.aggregate_equity.to_frame("equity").to_parquet(output_dir / "aggregate_equity.parquet")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
