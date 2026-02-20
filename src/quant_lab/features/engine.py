"""Feature computation orchestrator."""

from __future__ import annotations

import pandas as pd
import numpy as np
import structlog

from quant_lab.features.registry import get_feature_func, FEATURE_REGISTRY

logger = structlog.get_logger(__name__)

# Import to trigger registration
import quant_lab.features.price_features  # noqa: F401
import quant_lab.features.cross_asset_features  # noqa: F401
import quant_lab.features.regime_features  # noqa: F401


class FeatureEngine:
    """Computes features by running registered feature functions."""

    def __init__(
        self,
        enabled_features: list[str],
        windows: dict[str, list[int]] | None = None,
        normalization: dict | None = None,
    ):
        self.enabled_features = enabled_features
        self.windows = windows or {}
        self.normalization = normalization or {"method": "rolling_zscore", "lookback": 252}

        # Validate all requested features exist
        for name in enabled_features:
            if name not in FEATURE_REGISTRY:
                available = ", ".join(sorted(FEATURE_REGISTRY.keys()))
                raise ValueError(f"Unknown feature '{name}'. Available: {available}")

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all enabled features on the input DataFrame."""
        logger.info("feature_compute_start", features=self.enabled_features)

        df = df.copy()

        # Build window lists from config
        all_windows = []
        for key in ["short", "medium", "long"]:
            all_windows.extend(self.windows.get(key, []))
        if not all_windows:
            all_windows = [1, 5, 21, 63]

        for feature_name in self.enabled_features:
            func = get_feature_func(feature_name)
            logger.debug("computing_feature", feature=feature_name)
            df = func(df, windows=all_windows)

        # Get feature columns (everything except the base OHLCV columns)
        base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
        feature_cols = [c for c in df.columns if c not in base_cols]

        logger.info(
            "feature_compute_complete",
            num_features=len(feature_cols),
            feature_names=feature_cols,
        )

        return df

    def normalize(self, df: pd.DataFrame, target_col: str = "log_return_1d") -> pd.DataFrame:
        """Apply rolling z-score normalization to feature columns (excludes target)."""
        base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
        # Exclude target column from normalization to preserve raw scale for backtesting
        exclude_cols = base_cols | {target_col}
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        if not feature_cols:
            return df

        method = self.normalization.get("method", "rolling_zscore")
        lookback = self.normalization.get("lookback", 252)

        if method == "rolling_zscore":
            for col in feature_cols:
                mean = df.groupby("ticker")[col].transform(
                    lambda s: s.rolling(window=lookback, min_periods=lookback // 2).mean()
                )
                std = df.groupby("ticker")[col].transform(
                    lambda s: s.rolling(window=lookback, min_periods=lookback // 2).std()
                )
                zscore = (df[col] - mean) / std.clip(lower=1e-8)
                # Zero out z-scores where std is near-zero (constant features)
                df[col] = zscore.where(std > 1e-6, 0.0)
        elif method == "rank":
            for col in feature_cols:
                df[col] = df.groupby("date")[col].rank(pct=True)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the list of computed feature column names."""
        base_cols = {"date", "ticker", "open", "high", "low", "close", "volume", "adj_close"}
        return [c for c in df.columns if c not in base_cols]
