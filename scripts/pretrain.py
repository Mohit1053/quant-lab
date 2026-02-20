"""Pre-train masked time-series encoder on full dataset.

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py representation.mask_ratio=0.2
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import structlog

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device, get_device_info
from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.features.engine import FeatureEngine
from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder, MaskedEncoderConfig
from quant_lab.representation.pretraining import PreTrainer, PretrainConfig
from quant_lab.representation.embedding_space import EmbeddingExtractor

logger = structlog.get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Pre-train masked encoder and extract embeddings."""
    set_global_seed(cfg.project.seed)
    device = get_device()
    logger.info("device_info", **get_device_info())

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

    # Create DataModule (use most of the data for pre-training)
    split = TemporalSplit(
        train_end=cfg.data.date_range.val_end,  # Use train + val for pre-training
        val_end=cfg.data.date_range.val_end,
    )
    rep_cfg = cfg.representation
    seq_len = rep_cfg.get("sequence_length", cfg.model.input.sequence_length)
    dm = QuantDataModule(
        feature_df, feature_cols, split,
        DataModuleConfig(sequence_length=seq_len, batch_size=64),
    )
    dm.setup()

    train_loader = dm.train_dataloader()
    if train_loader is None:
        logger.error("No training data for pre-training.")
        return

    # Build masked encoder
    encoder_config = MaskedEncoderConfig(
        num_features=dm.num_features,
        patch_size=rep_cfg.get("patch_size", 5),
        d_model=rep_cfg.get("d_model", cfg.model.architecture.d_model),
        nhead=rep_cfg.get("nhead", cfg.model.architecture.nhead),
        num_encoder_layers=rep_cfg.get("num_layers", cfg.model.architecture.num_encoder_layers),
        dim_feedforward=rep_cfg.get("dim_feedforward", cfg.model.architecture.dim_feedforward),
        dropout=rep_cfg.get("dropout", cfg.model.architecture.dropout),
        mask_ratio=rep_cfg.get("mask_ratio", 0.15),
    )
    model = MaskedTimeSeriesEncoder(encoder_config)
    logger.info("model_built", parameters=model.count_parameters())

    # Pre-train
    pt_config = PretrainConfig(
        epochs=rep_cfg.get("epochs", 50),
        learning_rate=rep_cfg.get("learning_rate", 1e-4),
        mask_ratio=rep_cfg.get("mask_ratio", 0.15),
        mixed_precision=cfg.project.mixed_precision,
        checkpoint_dir="outputs/models/pretrained",
    )
    pretrainer = PreTrainer(model, pt_config, device)
    history = pretrainer.fit(train_loader)

    # Save model
    save_path = Path("outputs/models/pretrained/masked_encoder.pt")
    model.save(save_path)
    logger.info("encoder_saved", path=str(save_path))

    # Extract embeddings
    logger.info("extracting_embeddings")
    extractor = EmbeddingExtractor(model, device)
    all_loader = dm.train_dataloader()
    if all_loader is not None:
        embeddings = extractor.extract(all_loader)
        logger.info("embeddings_extracted", shape=embeddings.shape)

    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs: {len(history['train_loss'])}")
    print(f"Final loss: {history['train_loss'][-1]:.6f}")
    print(f"Model saved: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
