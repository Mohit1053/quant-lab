"""Optuna hyperparameter sweep for Transformer / TFT forecasters.

Usage:
    python scripts/sweep_hyperparams.py                        # Transformer, 50 trials
    python scripts/sweep_hyperparams.py --model tft            # TFT sweep
    python scripts/sweep_hyperparams.py --n-trials 100         # More trials
    python scripts/sweep_hyperparams.py --timeout 3600         # 1-hour budget
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import optuna
from omegaconf import DictConfig, OmegaConf
import structlog
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.features.engine import FeatureEngine
from quant_lab.features.feature_store import FeatureStore
from quant_lab.models.transformer.model import (
    TransformerForecaster,
    TransformerConfig,
    MultiTaskLoss,
)
from quant_lab.models.tft.model import TFTForecaster, TFTConfig
from quant_lab.training.trainer import Trainer, TrainerConfig

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Suggest helpers
# ---------------------------------------------------------------------------

def _suggest_transformer(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for the Transformer."""
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])

    # Ensure d_model % nhead == 0
    while d_model % nhead != 0:
        nhead = trial.suggest_categorical("nhead", [4, 8])

    return {
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 2, 6),
        "dim_feedforward": trial.suggest_categorical("dim_feedforward", [256, 512, 1024]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [200, 500, 1000]),
        "direction_weight": trial.suggest_float("direction_weight", 0.1, 0.5),
        "volatility_weight": trial.suggest_float("volatility_weight", 0.1, 0.5),
    }


def _suggest_tft(trial: optuna.Trial) -> dict:
    """Suggest hyperparameters for the TFT."""
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    nhead = trial.suggest_categorical("nhead", [2, 4, 8])

    while d_model % nhead != 0:
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])

    return {
        "d_model": d_model,
        "nhead": nhead,
        "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
        "lstm_layers": trial.suggest_int("lstm_layers", 1, 2),
        "lstm_hidden": trial.suggest_categorical("lstm_hidden", [64, 128, 256]),
        "grn_hidden": trial.suggest_categorical("grn_hidden", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.3),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [200, 500, 1000]),
        "direction_weight": trial.suggest_float("direction_weight", 0.1, 0.5),
        "volatility_weight": trial.suggest_float("volatility_weight", 0.1, 0.5),
    }


# ---------------------------------------------------------------------------
# Build model from suggested params
# ---------------------------------------------------------------------------

def _build_transformer(hp: dict, num_features: int):
    cfg = TransformerConfig(
        num_features=num_features,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        direction_weight=hp["direction_weight"],
        volatility_weight=hp["volatility_weight"],
    )
    model = TransformerForecaster(cfg)
    loss_fn = MultiTaskLoss(cfg)
    return model, loss_fn, cfg


def _build_tft(hp: dict, num_features: int):
    cfg = TFTConfig(
        num_features=num_features,
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_encoder_layers=hp["num_encoder_layers"],
        lstm_layers=hp["lstm_layers"],
        lstm_hidden=hp["lstm_hidden"],
        grn_hidden=hp["grn_hidden"],
        dropout=hp["dropout"],
        direction_weight=hp["direction_weight"],
        volatility_weight=hp["volatility_weight"],
    )
    model = TFTForecaster(cfg)
    # TFT reuses the same MultiTaskLoss structure (same loss weight fields)
    loss_cfg = TransformerConfig(
        num_features=num_features,
        direction_weight=hp["direction_weight"],
        volatility_weight=hp["volatility_weight"],
    )
    loss_fn = MultiTaskLoss(loss_cfg)
    return model, loss_fn, cfg


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(
    trial: optuna.Trial,
    *,
    model_type: str,
    feature_df,
    feature_cols: list[str],
    split: TemporalSplit,
    device: torch.device,
    epochs_per_trial: int,
    patience: int,
) -> float:
    """Optuna objective: train one trial and return best val loss."""

    # 1. Suggest hyperparameters
    if model_type == "transformer":
        hp = _suggest_transformer(trial)
    else:
        hp = _suggest_tft(trial)

    batch_size = hp["batch_size"]

    # 2. Build DataModule with suggested batch size
    dm_config = DataModuleConfig(
        sequence_length=63,
        target_col="log_return_1d",
        batch_size=batch_size,
        num_workers=0,
    )
    dm = QuantDataModule(feature_df, feature_cols, split, dm_config)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if train_loader is None or val_loader is None:
        return float("inf")

    # 3. Build model
    num_features = dm.num_features
    if model_type == "transformer":
        model, loss_fn, _ = _build_transformer(hp, num_features)
    else:
        model, loss_fn, _ = _build_tft(hp, num_features)

    trial.set_user_attr("parameters", model.count_parameters())

    # 4. Trainer with shortened schedule
    trainer_config = TrainerConfig(
        epochs=epochs_per_trial,
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_steps=hp["warmup_steps"],
        patience=patience,
        mixed_precision=device.type != "cpu",
        checkpoint_dir=f"outputs/sweep/{model_type}/trial_{trial.number}",
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=trainer_config,
        device=device,
    )

    # 5. Train
    history = trainer.fit(train_loader, val_loader)

    # 6. Report intermediate values for pruning
    for epoch_idx, val_loss in enumerate(history["val_loss"]):
        trial.report(val_loss, epoch_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")

    # 7. Clean up GPU memory
    del model, trainer, loss_fn, dm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweep")
    parser.add_argument("--model", choices=["transformer", "tft"], default="transformer")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=7200, help="Max seconds")
    parser.add_argument("--epochs-per-trial", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = get_device()

    # Load feature data (reuse Hydra config for paths)
    from omegaconf import OmegaConf

    cfg_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(cfg_path) as f:
        raw = OmegaConf.load(f)

    data_dir = Path("data")
    universe_name = raw.data.universe.name if hasattr(raw, "data") else "nifty50"

    feature_store = FeatureStore(data_dir / "features")
    feature_name = f"{universe_name}_features"
    if not feature_store.has_features(feature_name):
        print(f"ERROR: No feature data at {data_dir / 'features'}.")
        print("Run 'python scripts/compute_features.py' first.")
        sys.exit(1)

    feature_df = feature_store.load_features(feature_name)

    # Detect feature columns
    from quant_lab.features.engine import FeatureEngine

    feat_cfg = raw.get("features", {})
    enabled = list(feat_cfg.get("enabled_features", []))
    windows = {k: list(v) for k, v in feat_cfg.get("windows", {}).items()} if "windows" in feat_cfg else {}
    engine = FeatureEngine(enabled_features=enabled, windows=windows)
    feature_cols = engine.get_feature_columns(feature_df)

    # Ensure target
    if "log_return_1d" not in feature_df.columns:
        feature_df["log_return_1d"] = feature_df.groupby("ticker")["adj_close"].transform(
            lambda s: np.log(s / s.shift(1))
        )

    # Temporal split
    date_cfg = raw.get("data", {}).get("date_range", {})
    split = TemporalSplit(
        train_end=str(date_cfg.get("train_end", "2023-06-30")),
        val_end=str(date_cfg.get("val_end", "2024-03-31")),
    )

    print("=" * 60)
    print(f"OPTUNA SWEEP: {args.model.upper()}")
    print("=" * 60)
    print(f"  Device:           {device}")
    print(f"  Trials:           {args.n_trials}")
    print(f"  Epochs/trial:     {args.epochs_per_trial}")
    print(f"  Patience:         {args.patience}")
    print(f"  Timeout:          {args.timeout}s")
    print(f"  Features:         {len(feature_cols)}")
    print(f"  Tickers:          {feature_df['ticker'].nunique()}")
    print("=" * 60)

    # Create Optuna study
    study = optuna.create_study(
        study_name=f"{args.model}_sweep",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
        ),
    )

    study.optimize(
        lambda trial: objective(
            trial,
            model_type=args.model,
            feature_df=feature_df,
            feature_cols=feature_cols,
            split=split,
            device=device,
            epochs_per_trial=args.epochs_per_trial,
            patience=args.patience,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Report results
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Best trial:       #{study.best_trial.number}")
    print(f"  Best val loss:    {study.best_trial.value:.6f}")
    print(f"  Completed trials: {len(study.trials)}")
    pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    print(f"  Pruned trials:    {pruned}")
    print("\nBest hyperparameters:")
    for k, v in study.best_trial.params.items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    # Save results
    output_dir = Path("outputs/sweep") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best params as JSON
    best_params = study.best_trial.params
    best_params["best_val_loss"] = study.best_trial.value
    best_params["trial_number"] = study.best_trial.number

    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to: {output_dir / 'best_params.json'}")

    # Save full trial history
    trials_data = []
    for t in study.trials:
        trial_info = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name,
            "params": t.params,
        }
        trials_data.append(trial_info)

    with open(output_dir / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2, default=str)
    print(f"All trials saved to: {output_dir / 'all_trials.json'}")

    # Retrain best model with full schedule if requested
    print(f"\nTo retrain best model with full schedule:")
    params_str = " ".join(f"{k}={v}" for k, v in study.best_trial.params.items()
                          if k not in ("batch_size", "warmup_steps"))
    print(f"  python scripts/train_forecaster.py {params_str}")


if __name__ == "__main__":
    main()
