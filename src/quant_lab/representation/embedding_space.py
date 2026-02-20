"""Extract and store market embeddings from pre-trained encoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import structlog

from quant_lab.representation.masked_encoder import MaskedTimeSeriesEncoder
from quant_lab.data.storage.parquet_store import ParquetStore
from quant_lab.utils.device import get_device, get_dtype

logger = structlog.get_logger(__name__)


class EmbeddingExtractor:
    """Extract market embeddings from a pre-trained masked encoder.

    Process:
    1. Load pre-trained MaskedTimeSeriesEncoder
    2. Run forward pass (encode only, no masking)
    3. Extract CLS token as the embedding for each (date, ticker) pair
    4. Store as Parquet: columns = [date, ticker, emb_0, ..., emb_{d_model-1}]
    """

    def __init__(
        self,
        model: MaskedTimeSeriesEncoder,
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.model.eval()

        self.use_amp = self.device.type != "cpu"
        self.amp_dtype = get_dtype() if self.use_amp else torch.float32

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> np.ndarray:
        """Extract embeddings from a DataLoader.

        Args:
            loader: DataLoader yielding (features, targets) batches

        Returns:
            (N, d_model) numpy array of embeddings
        """
        embeddings = []

        for features, _ in loader:
            features = features.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                emb = self.model.encode(features)  # (batch, d_model)

            embeddings.append(emb.float().cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def extract_and_save(
        self,
        loader: DataLoader,
        dates: np.ndarray,
        tickers: np.ndarray,
        save_dir: str = "data/embeddings",
        filename: str = "embeddings",
    ) -> pd.DataFrame:
        """Extract embeddings and save to Parquet with metadata.

        Args:
            loader: DataLoader
            dates: array of dates for each sample
            tickers: array of tickers for each sample
            save_dir: directory to save parquet file
            filename: parquet filename

        Returns:
            DataFrame with columns [date, ticker, emb_0, ..., emb_{d-1}]
        """
        embeddings = self.extract(loader)
        d_model = embeddings.shape[1]

        # Build DataFrame
        emb_cols = [f"emb_{i}" for i in range(d_model)]
        df = pd.DataFrame(embeddings, columns=emb_cols)
        df.insert(0, "date", dates[:len(df)])
        df.insert(1, "ticker", tickers[:len(df)])

        # Save
        store = ParquetStore(base_dir=save_dir)
        store.save(df, filename)

        logger.info(
            "embeddings_saved",
            samples=len(df),
            d_model=d_model,
            path=f"{save_dir}/{filename}",
        )

        return df
