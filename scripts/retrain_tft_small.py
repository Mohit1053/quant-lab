"""Retrain TFT with smaller architecture to fix mode collapse."""
from __future__ import annotations

import sys
import shutil
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quant_lab.utils.seed import set_global_seed
from quant_lab.utils.device import get_device
from quant_lab.features.feature_store import FeatureStore
from quant_lab.data.datasets import TemporalSplit
from quant_lab.data.datamodule import QuantDataModule, DataModuleConfig
from quant_lab.models.tft.model import TFTForecaster, TFTConfig
from quant_lab.models.transformer.model import MultiTaskLoss, TransformerConfig
from quant_lab.training.trainer import Trainer, TrainerConfig


def main():
    set_global_seed(42)
    device = get_device()

    store = FeatureStore("data/features")
    df = store.load_features("nifty50_features")
    base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
    feature_cols = [c for c in df.columns if c not in base_cols]
    split = TemporalSplit(train_end="2021-12-31", val_end="2023-06-30")

    dm = QuantDataModule(
        df, feature_cols, split,
        DataModuleConfig(sequence_length=63, target_col="log_return_1d", batch_size=64, num_workers=0),
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"Features: {dm.num_features}, Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Smaller TFT: 32 d_model (was 128), 1 layer (was 2), 0.3 dropout (was 0.1)
    tft_cfg = TFTConfig(
        num_features=dm.num_features,
        d_model=32,
        nhead=4,
        num_encoder_layers=1,
        lstm_layers=1,
        lstm_hidden=32,
        grn_hidden=16,
        dropout=0.3,
        direction_weight=0.3,
        volatility_weight=0.3,
    )
    model = TFTForecaster(tft_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,} (was 35,606,293)")

    loss_cfg = TransformerConfig(
        num_features=dm.num_features,
        direction_weight=0.3,
        volatility_weight=0.3,
    )
    loss_fn = MultiTaskLoss(loss_cfg)

    trainer_config = TrainerConfig(
        epochs=100,
        learning_rate=3e-4,
        weight_decay=1e-3,
        warmup_steps=500,
        patience=15,
        mixed_precision=True,
        checkpoint_dir="outputs/models/tft_small",
    )
    trainer = Trainer(model, loss_fn, trainer_config, device)
    trainer.fit(train_loader, val_loader)

    # Verify predictions
    model.eval()
    x = torch.randn(20, 63, dm.num_features).to(device)
    with torch.no_grad():
        preds = model.predict_returns(x)
    print(f"\nPrediction std: {preds.std():.6f} (was ~0)")
    print(f"Prediction range: [{preds.min():.6f}, {preds.max():.6f}]")

    # Save (backup old collapsed model)
    out = Path("outputs/models/tft")
    if (out / "final_model.pt").exists():
        shutil.copy(out / "final_model.pt", out / "final_model_collapsed.pt")
    model.save(out / "final_model.pt")
    model.save(out / "best.pt")
    print(f"Saved retrained TFT to {out}")


if __name__ == "__main__":
    main()
