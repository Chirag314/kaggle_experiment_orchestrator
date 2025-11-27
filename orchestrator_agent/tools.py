"""
plain Python functions that:
- Load an experiments CSV
- Compute simple statistics
- Produce a human-readable summary string
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_experiments(csv_path: str | Path) -> pd.DataFrame:
    """
    Load an experiments CSV into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the experiments.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiments file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    expected_cols = {
        "experiment_id",
        "model_type",
        "features_desc",
        "params_summary",
        "cv_metric",
        "holdout_metric",
        "train_time_seconds",
        "notes",
    }

    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    return df


def compute_model_family_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute aggregated stats per model_type.

    For each model_type, return:
      - n_runs
      - best_cv
      - mean_cv
      - mean_gap (cv - holdout)
      - mean_train_time
    """
    # Ensure gap column exists
    if "cv_holdout_gap" not in df.columns:
        df = df.copy()
        df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]

    stats: Dict[str, Dict[str, float]] = {}

    for model, group in df.groupby("model_type"):
        stats[model] = {
            "n_runs": int(len(group)),
            "best_cv": float(group["cv_metric"].max()),
            "mean_cv": float(group["cv_metric"].mean()),
            "mean_gap": float(group["cv_holdout_gap"].mean()),
            "mean_train_time": float(group["train_time_seconds"].mean()),
        }

    return stats


def summarize_experiments(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics and useful views over experiment results.

    Returns a dictionary you can later feed into an agent or a report-writer.
    """
    # Basic counts
    n_experiments = len(df)
    model_counts = df["model_type"].value_counts().to_dict()

    # Best experiments by cv_metric
    df_sorted_cv = df.sort_values("cv_metric", ascending=False).reset_index(drop=True)
    best_cv_row = df_sorted_cv.iloc[0].to_dict()

    # Overfitting indicator: large gap between cv and holdout
    df = df.copy()
    df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]
    df_sorted_gap = df.sort_values("cv_holdout_gap", ascending=False)
    worst_gap_row = df_sorted_gap.iloc[0].to_dict()

    # Time stats
    time_stats = {
        "min_train_time": float(df["train_time_seconds"].min()),
        "max_train_time": float(df["train_time_seconds"].max()),
        "mean_train_time": float(df["train_time_seconds"].mean()),
    }

    # Model-family stats (per model_type)
    model_family_stats = compute_model_family_stats(df)

    summary = {
        "n_experiments": int(n_experiments),
        "model_counts": model_counts,
        "best_cv_experiment": best_cv_row,
        "worst_gap_experiment": worst_gap_row,
        "time_stats": time_stats,
        "model_family_stats": model_family_stats,
    }

    return summary


def format_summary_text(summary: Dict[str, Any]) -> str:
    """
    Turn the summary dictionary into a human-readable text block.
    """
    lines = []

    lines.append(f"Total experiments: {summary['n_experiments']}")
    lines.append("Models used (raw counts):")
    for model, count in summary["model_counts"].items():
        lines.append(f"  - {model}: {count} runs")

    # Per-model stats
    lines.append("\nPer-model summary:")
    for model, stats in summary["model_family_stats"].items():
        lines.append(f"  {model}:")
        lines.append(f"    runs:          {stats['n_runs']}")
        lines.append(f"    best CV:       {stats['best_cv']:.4f}")
        lines.append(f"    mean CV:       {stats['mean_cv']:.4f}")
        lines.append(f"    mean CV-gap:   {stats['mean_gap']:.4f}")
        lines.append(f"    mean train(s): {stats['mean_train_time']:.1f}")

    best = summary["best_cv_experiment"]
    lines.append("\nBest CV experiment:")
    lines.append(f"  ID: {best['experiment_id']}")
    lines.append(f"  Model: {best['model_type']}")
    lines.append(f"  CV metric: {best['cv_metric']:.4f}")
    lines.append(f"  Holdout metric: {best['holdout_metric']:.4f}")
    lines.append(f"  Features: {best['features_desc']}")
    lines.append(f"  Params: {best['params_summary']}")

    worst = summary["worst_gap_experiment"]
    lines.append("\nMost overfitted experiment (largest CV - holdout gap):")
    lines.append(f"  ID: {worst['experiment_id']}")
    lines.append(f"  Model: {worst['model_type']}")
    lines.append(f"  CV metric: {worst['cv_metric']:.4f}")
    lines.append(f"  Holdout metric: {worst['holdout_metric']:.4f}")
    lines.append(f"  Gap: {worst['cv_metric'] - worst['holdout_metric']:.4f}")
    lines.append(f"  Notes: {worst['notes']}")

    t = summary["time_stats"]
    lines.append("\nTraining time (seconds):")
    lines.append(f"  min:  {t['min_train_time']:.1f}")
    lines.append(f"  mean: {t['mean_train_time']:.1f}")
    lines.append(f"  max:  {t['max_train_time']:.1f}")

    return "\n".join(lines)
