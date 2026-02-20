"""Parse walk-forward log files into per_fold_metrics.csv.

Extracts fold splits and metrics from structlog output and saves
them as CSVs compatible with consolidate_results.py.

Usage:
    python scripts/parse_wf_logs.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def parse_wf_log(log_path: Path) -> pd.DataFrame:
    """Parse a walk-forward log file into a DataFrame of per-fold metrics."""
    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Parse fold split info: fold=N test_end=DATE train_start=DATE ...
    splits = {}
    for m in re.finditer(
        r"walk_forward_fold\s+"
        r"fold=(\d+)\s+"
        r"test_end=([\d-]+)\s+"
        r"train_days=(\d+)\s+"
        r"train_end=([\d-]+)\s+"
        r"train_start=([\d-]+)\s+"
        r"val_end=([\d-]+)",
        text,
    ):
        fold = int(m.group(1))
        splits[fold] = {
            "test_end": m.group(2),
            "train_days": int(m.group(3)),
            "train_start": m.group(5),
            "train_end": m.group(4),
            "val_end": m.group(6),
        }

    # Parse fold completion: fold=N sharpe=X total_return=Y
    completions = {}
    for m in re.finditer(
        r"walk_forward_fold_complete\s+"
        r"fold=(\d+)\s+"
        r"sharpe=([\d.e+-]+)\s+"
        r"total_return=([\d.e+-]+)",
        text,
    ):
        fold = int(m.group(1))
        completions[fold] = {
            "sharpe": float(m.group(2)),
            "total_return": float(m.group(3)),
        }

    if not completions:
        return pd.DataFrame()

    rows = []
    for fold in sorted(completions.keys()):
        row = {"fold": fold, **completions[fold]}
        if fold in splits:
            # Compute test_start from val_end (next business day)
            val_end = pd.Timestamp(splits[fold]["val_end"])
            test_start = val_end + pd.offsets.BDay(1)
            row["test_start"] = str(test_start.date())
            row["test_end"] = splits[fold]["test_end"]
            row["train_start"] = splits[fold]["train_start"]
            row["train_days"] = splits[fold]["train_days"]
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns
    col_order = ["fold", "test_start", "test_end", "sharpe", "total_return"]
    extra = [c for c in df.columns if c not in col_order]
    return df[[c for c in col_order if c in df.columns] + extra]


def main():
    wf_dir = Path("outputs/walk_forward")

    configs = [
        ("Transformer NIFTY 50", wf_dir / "transformer_wf_log.txt", wf_dir / "transformer"),
        ("Ridge NIFTY 500", wf_dir / "ridge_nifty500_log.txt", wf_dir / "ridge_nifty500"),
    ]

    for label, log_path, out_dir in configs:
        if not log_path.exists():
            print(f"  [skip] {label}: log not found at {log_path}")
            continue

        df = parse_wf_log(log_path)
        if df.empty:
            print(f"  [skip] {label}: no completed folds found in log")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "per_fold_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [ok] {label}: {len(df)} folds -> {csv_path}")

        # Summary stats
        avg_sharpe = df["sharpe"].mean()
        std_sharpe = df["sharpe"].std()
        pos = (df["sharpe"] > 0).sum()
        neg = (df["sharpe"] < 0).sum()
        print(f"       Avg Sharpe: {avg_sharpe:.4f} +/- {std_sharpe:.4f}")
        print(f"       Win rate: {pos}/{pos+neg} ({pos/(pos+neg)*100:.0f}%)")
        print(f"       Best: {df['sharpe'].max():.4f}, Worst: {df['sharpe'].min():.4f}")
        print()


if __name__ == "__main__":
    main()
