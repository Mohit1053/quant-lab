"""Train transformer forecaster on feature data.

Usage:
    python scripts/train_forecaster.py
    python scripts/train_forecaster.py model.architecture.d_model=256
    python scripts/train_forecaster.py model.training.epochs=50 model.training.batch_size=128
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import structlog
import torch

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device, get_device_info
from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.features.engine import FeatureEngine
from quant_lab.models.transformer.model import TransformerForecaster, TransformerConfig, MultiTaskLoss
from quant_lab.training.trainer import Trainer, TrainerConfig

logger = structlog.get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train transformer forecaster."""
    # Setup
    set_global_seed(cfg.project.seed)
    device = get_device()
    device_info = get_device_info()
    logger.info("device_info", **device_info)
    logger.info("config", config=OmegaConf.to_yaml(cfg))

    # Load feature data
    data_dir = Path(cfg.paths.data_dir)
    universe_name = cfg.data.universe.name
    store = ParquetStore(base_dir=str(data_dir / "features"))
    feature_name = f"{universe_name}_features"
    if not store.exists(feature_name):
        logger.error("No feature data found. Run 'python scripts/compute_features.py' first.")
        return

    feature_df = store.load(feature_name)
    logger.info("data_loaded", rows=len(feature_df), tickers=feature_df["ticker"].nunique())

    # Detect feature columns
    engine = FeatureEngine(
        enabled_features=list(cfg.features.enabled_features),
        windows={k: list(v) for k, v in cfg.features.windows.items()},
    )
    feature_cols = engine.get_feature_columns(feature_df)
    logger.info("features_detected", num_features=len(feature_cols), features=feature_cols[:10])

    # Create DataModule
    split = TemporalSplit(
        train_end=cfg.data.date_range.train_end,
        val_end=cfg.data.date_range.val_end,
    )
    dm_config = DataModuleConfig(
        sequence_length=cfg.model.input.sequence_length,
        target_col="log_return_1d",
        batch_size=cfg.model.training.batch_size,
        num_workers=0,  # Windows-safe; set higher on Linux
    )
    dm = QuantDataModule(feature_df, feature_cols, split, dm_config)
    dm.setup()

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    if train_loader is None:
        logger.error("No training data. Check split dates and feature data.")
        return

    # Build model
    model_cfg = TransformerConfig(
        num_features=dm.num_features,
        d_model=cfg.model.architecture.d_model,
        nhead=cfg.model.architecture.nhead,
        num_encoder_layers=cfg.model.architecture.num_encoder_layers,
        dim_feedforward=cfg.model.architecture.dim_feedforward,
        dropout=cfg.model.architecture.dropout,
        activation=cfg.model.architecture.activation,
        distribution_type=cfg.model.heads.distribution.type,
        direction_num_classes=cfg.model.heads.direction.num_classes,
        direction_threshold=cfg.model.heads.direction.threshold,
        volatility_enabled=cfg.model.heads.volatility.enabled,
        distribution_weight=cfg.model.loss.distribution_weight,
        direction_weight=cfg.model.loss.direction_weight,
        volatility_weight=cfg.model.loss.volatility_weight,
    )

    model = TransformerForecaster(model_cfg)
    loss_fn = MultiTaskLoss(model_cfg)

    logger.info(
        "model_built",
        parameters=model.count_parameters(),
        d_model=model_cfg.d_model,
        layers=model_cfg.num_encoder_layers,
    )

    # Setup tracker (optional)
    tracker = None
    try:
        from quant_lab.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(
            experiment_name=cfg.experiment.tracking.get("experiment_name", "transformer_forecaster"),
            tracking_uri=cfg.experiment.mlflow.get("tracking_uri", "mlruns"),
        )
        tracker.start_run(run_name="transformer_train")
        tracker.log_config(OmegaConf.to_container(cfg, resolve=True))
        tracker.log_params({"num_parameters": model.count_parameters()})
    except Exception as e:
        logger.warning("mlflow_setup_failed", error=str(e))

    # Train
    trainer_config = TrainerConfig(
        epochs=cfg.model.training.epochs,
        learning_rate=cfg.model.training.learning_rate,
        weight_decay=cfg.model.training.weight_decay,
        warmup_steps=cfg.model.training.warmup_steps,
        max_grad_norm=cfg.model.training.max_grad_norm,
        patience=cfg.model.training.patience,
        mixed_precision=cfg.project.mixed_precision,
        checkpoint_dir="outputs/models/transformer",
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=trainer_config,
        device=device,
        tracker=tracker,
    )

    history = trainer.fit(train_loader, val_loader)

    # Log final metrics
    if tracker is not None:
        tracker.log_metrics({
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else 0,
            "epochs_trained": len(history["train_loss"]),
        })
        tracker.end_run()

    # Save final model
    save_path = Path("outputs/models/transformer/final_model.pt")
    model.save(save_path)
    logger.info("model_saved", path=str(save_path))

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs trained: {len(history['train_loss'])}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    if history["val_loss"]:
        print(f"Final val loss:   {history['val_loss'][-1]:.6f}")
        print(f"Best val loss:    {min(history['val_loss']):.6f}")
    print(f"Model saved to:   {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
