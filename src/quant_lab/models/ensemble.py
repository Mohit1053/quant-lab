"""Ensemble strategies for combining multi-model signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class CombinationMethod(str, Enum):
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    REGIME_CONDITIONAL = "regime_conditional"


@dataclass
class EnsembleConfig:
    """Configuration for ensemble signal combination."""

    method: CombinationMethod = CombinationMethod.SIMPLE_AVERAGE
    weights: dict[str, float] = field(default_factory=dict)
    # regime_id -> {model_name -> weight}
    regime_weights: dict[int, dict[str, float]] = field(default_factory=dict)


class EnsembleStrategy:
    """Combines signals from multiple models.

    Operates at the signal-DataFrame level, not at the raw prediction
    level, because different model types produce predictions through
    different interfaces (flat vs. sequence input).
    """

    def __init__(self, config: EnsembleConfig | None = None):
        self.config = config or EnsembleConfig()

    def combine(
        self,
        signals: dict[str, pd.DataFrame],
        regime_labels: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Combine multiple signal DataFrames into one.

        Args:
            signals: Mapping of model_name -> DataFrame[date, ticker, signal].
            regime_labels: Optional Series indexed by date with regime IDs.

        Returns:
            Combined signal DataFrame[date, ticker, signal].
        """
        if not signals:
            raise ValueError("No signals to combine")

        model_names = list(signals.keys())

        if len(model_names) == 1:
            return signals[model_names[0]][["date", "ticker", "signal"]].copy()

        method = self.config.method

        if method == CombinationMethod.SIMPLE_AVERAGE:
            return self._simple_average(signals, model_names)
        elif method == CombinationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(signals, model_names)
        elif method == CombinationMethod.REGIME_CONDITIONAL:
            if regime_labels is None:
                logger.warning(
                    "regime_labels_missing_falling_back_to_weighted"
                )
                return self._weighted_average(signals, model_names)
            return self._regime_conditional(
                signals, model_names, regime_labels
            )
        else:
            raise ValueError(f"Unknown combination method: {method}")

    def _simple_average(
        self,
        signals: dict[str, pd.DataFrame],
        model_names: list[str],
    ) -> pd.DataFrame:
        """Equal-weight average of all model signals."""
        merged = self._merge_signals(signals, model_names)
        signal_cols = [f"signal_{name}" for name in model_names]
        merged["signal"] = merged[signal_cols].mean(axis=1)
        return merged[["date", "ticker", "signal"]].copy()

    def _weighted_average(
        self,
        signals: dict[str, pd.DataFrame],
        model_names: list[str],
    ) -> pd.DataFrame:
        """Weighted average using self.config.weights."""
        merged = self._merge_signals(signals, model_names)

        weights = self.config.weights
        if not weights:
            weights = {name: 1.0 / len(model_names) for name in model_names}

        total = sum(weights.get(name, 0) for name in model_names)
        if total == 0:
            total = 1.0

        merged["signal"] = sum(
            merged[f"signal_{name}"] * (weights.get(name, 0) / total)
            for name in model_names
        )
        return merged[["date", "ticker", "signal"]].copy()

    def _regime_conditional(
        self,
        signals: dict[str, pd.DataFrame],
        model_names: list[str],
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        """Per-regime weighted combination."""
        merged = self._merge_signals(signals, model_names)

        # Map regime labels to each row
        merged["regime"] = merged["date"].map(regime_labels)

        default_weights = self.config.weights or {
            name: 1.0 / len(model_names) for name in model_names
        }

        def _compute_row_signal(row):
            regime = row.get("regime")
            if pd.notna(regime) and int(regime) in self.config.regime_weights:
                w = self.config.regime_weights[int(regime)]
            else:
                w = default_weights
            total = sum(w.get(name, 0) for name in model_names) or 1.0
            return sum(
                row[f"signal_{name}"] * (w.get(name, 0) / total)
                for name in model_names
            )

        merged["signal"] = merged.apply(_compute_row_signal, axis=1)
        return merged[["date", "ticker", "signal"]].copy()

    def _merge_signals(
        self,
        signals: dict[str, pd.DataFrame],
        model_names: list[str],
    ) -> pd.DataFrame:
        """Merge signal DataFrames on (date, ticker)."""
        base = None
        for name in model_names:
            df = signals[name][["date", "ticker", "signal"]].copy()
            df = df.rename(columns={"signal": f"signal_{name}"})
            if base is None:
                base = df
            else:
                base = base.merge(df, on=["date", "ticker"], how="inner")

        logger.info(
            "signals_merged",
            models=model_names,
            rows=len(base) if base is not None else 0,
        )
        return base

    def get_model_contributions(
        self,
        signals: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Compute per-model contribution statistics.

        Returns DataFrame with columns:
            model, mean_signal, std_signal, correlation_with_ensemble
        """
        model_names = list(signals.keys())
        merged = self._merge_signals(signals, model_names)

        ensemble = self.combine(signals)
        merged = merged.merge(
            ensemble.rename(columns={"signal": "ensemble_signal"}),
            on=["date", "ticker"],
        )

        records = []
        for name in model_names:
            col = f"signal_{name}"
            records.append({
                "model": name,
                "mean_signal": float(merged[col].mean()),
                "std_signal": float(merged[col].std()),
                "correlation_with_ensemble": float(
                    merged[col].corr(merged["ensemble_signal"])
                ),
            })
        return pd.DataFrame(records)


def optimize_ensemble_weights(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    metric: str = "sharpe",
    n_trials: int = 100,
) -> dict[str, float]:
    """Find optimal ensemble weights via random Dirichlet sampling.

    Evaluates each weight combination with a quick backtest on the
    provided validation-period data.

    Args:
        signals: model_name -> signal DataFrame for validation period.
        prices: Price DataFrame for validation period.
        metric: Metric to maximize ("sharpe" or "calmar").
        n_trials: Number of random weight combinations to try.

    Returns:
        Optimal weights dict: {model_name: weight}.
    """
    from quant_lab.backtest.engine import BacktestEngine, BacktestConfig

    model_names = list(signals.keys())
    n_models = len(model_names)
    best_score = -np.inf
    best_weights = {name: 1.0 / n_models for name in model_names}

    config = BacktestConfig(
        initial_capital=1_000_000,
        rebalance_frequency=5,
        top_n=5,
    )
    engine = BacktestEngine(config=config)

    for _ in range(n_trials):
        raw = np.random.dirichlet(np.ones(n_models))
        trial_weights = dict(zip(model_names, raw))

        ensemble_cfg = EnsembleConfig(
            method=CombinationMethod.WEIGHTED_AVERAGE,
            weights=trial_weights,
        )
        ensemble = EnsembleStrategy(ensemble_cfg)
        combined = ensemble.combine(signals)

        try:
            result = engine.run(prices, combined)
            score = result.metrics.get(metric, 0.0)
            if score > best_score:
                best_score = score
                best_weights = trial_weights
        except Exception:
            continue

    logger.info(
        "weight_optimization_complete",
        best_score=f"{best_score:.4f}",
        best_weights={k: f"{v:.3f}" for k, v in best_weights.items()},
    )
    return best_weights
