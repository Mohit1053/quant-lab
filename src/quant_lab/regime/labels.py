"""Map cluster IDs to interpretable regime labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# Standard regime labels
REGIME_LABELS = {
    "low_vol_bull": "Low-Vol Bull",
    "high_vol_bull": "High-Vol Bull",
    "bear": "Bear",
    "crisis": "Crisis",
    "transition": "Transition",
}


@dataclass
class RegimeCharacteristics:
    """Characteristics of a detected regime."""

    label: str
    display_name: str
    mean_return: float
    mean_volatility: float
    frequency: float  # Fraction of time spent in this regime
    avg_duration: float  # Average consecutive days in regime


def label_regimes(
    regime_ids: np.ndarray,
    returns: np.ndarray,
    volatility: np.ndarray,
) -> dict[int, RegimeCharacteristics]:
    """Assign interpretable labels to cluster IDs based on return/vol characteristics.

    Logic:
    - Rank clusters by mean return and mean volatility
    - Highest return + low vol -> Low-Vol Bull
    - Highest return + high vol -> High-Vol Bull
    - Lowest return + high vol -> Crisis
    - Low return + moderate vol -> Bear
    - Anything else -> Transition

    Args:
        regime_ids: (n_samples,) integer cluster labels.
        returns: (n_samples,) return values.
        volatility: (n_samples,) volatility values.

    Returns:
        Dict mapping cluster ID to RegimeCharacteristics.
    """
    unique_ids = sorted(set(regime_ids[regime_ids >= 0]))  # Exclude noise (-1)

    cluster_stats = {}
    for cid in unique_ids:
        mask = regime_ids == cid
        cluster_stats[cid] = {
            "mean_return": float(np.mean(returns[mask])),
            "mean_vol": float(np.mean(volatility[mask])),
            "frequency": float(np.sum(mask)) / len(regime_ids),
            "avg_duration": _avg_consecutive_run(regime_ids, cid),
        }

    if len(unique_ids) == 0:
        return {}

    # Sort by mean return
    sorted_by_return = sorted(unique_ids, key=lambda c: cluster_stats[c]["mean_return"])
    median_vol = np.median([cluster_stats[c]["mean_vol"] for c in unique_ids])

    label_map = {}
    assigned_labels = set()

    for cid in unique_ids:
        stats = cluster_stats[cid]
        ret_rank = sorted_by_return.index(cid) / max(len(sorted_by_return) - 1, 1)
        is_high_vol = stats["mean_vol"] > median_vol

        if ret_rank >= 0.75 and not is_high_vol and "low_vol_bull" not in assigned_labels:
            label_key = "low_vol_bull"
        elif ret_rank >= 0.75 and is_high_vol and "high_vol_bull" not in assigned_labels:
            label_key = "high_vol_bull"
        elif ret_rank <= 0.25 and is_high_vol and "crisis" not in assigned_labels:
            label_key = "crisis"
        elif ret_rank <= 0.25 and "bear" not in assigned_labels:
            label_key = "bear"
        else:
            label_key = "transition"

        assigned_labels.add(label_key)
        label_map[cid] = RegimeCharacteristics(
            label=label_key,
            display_name=REGIME_LABELS[label_key],
            mean_return=stats["mean_return"],
            mean_volatility=stats["mean_vol"],
            frequency=stats["frequency"],
            avg_duration=stats["avg_duration"],
        )

    return label_map


def _avg_consecutive_run(labels: np.ndarray, target: int) -> float:
    """Compute average length of consecutive runs of a given label."""
    runs = []
    current_run = 0
    for lab in labels:
        if lab == target:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)

    return float(np.mean(runs)) if runs else 0.0


def regime_summary_table(
    label_map: dict[int, RegimeCharacteristics],
) -> pd.DataFrame:
    """Create a summary DataFrame of regime characteristics."""
    rows = []
    for cid, chars in sorted(label_map.items()):
        rows.append({
            "cluster_id": cid,
            "label": chars.display_name,
            "mean_return": f"{chars.mean_return:.4f}",
            "mean_volatility": f"{chars.mean_volatility:.4f}",
            "frequency": f"{chars.frequency:.1%}",
            "avg_duration": f"{chars.avg_duration:.1f} days",
        })
    return pd.DataFrame(rows)
