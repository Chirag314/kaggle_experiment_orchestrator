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


def load_experiments(csv_path: str | path) -> pd.DataFrame:
    """Load experiments from a CSV file into a DataFrame."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    expected_columns = {
        "experiment_id",
        "model_type",
        "features_desc",
        "params_summary",
        "cv_metric",
        "holdout_metric",
        "train_time_seconds",
        "notes",
    }

    missing = expected_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing expected columns in csv: {missing}")
    return df


def summarize_experiments(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics and useful views over experiemtnt results.

    Returns a dictionary you can later feed into an agent or a repoprt-writier.
    """
    # Baseic statistics
    n_epxperiments = len(df)
    model_counts = df["model_type"].value_counts().to_dict()

    # beset experiemtn by cv_metric
    df_sorted_cv = df.sort_values("cv_metric", ascending=False).reset_index(drop=True)
    best_cv_row = df_sorted_cv.iloc[0].to_dict()

    # Overfitting indicator
    df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]
    df_sorted_gap = df.sort_values("cv_holdout_gap", ascending=False)
    worst_gap_row = df_sorted_gap.iloc[0].to_dict()

    # Time stats
    time_stats = {
        "min_train_time": float(df["train_time_seconds"].min()),
        "max_train_time": float(df["train_time_seconds"].max()),
        "mean_train_time": float(df["train_time_seconds"].mean()),
    }

    summary = {
        "n_experiments": int(n_epxperiments),
        "model_counts": model_counts,
        "best_cv_experiment": best_cv_row,
        "worst_gap_experiment": worst_gap_row,
        "time_stats": time_stats,
    }
    return summary


def format_summary_text(summary: Dict[str, Any]) -> str:
    """
    Turn the summary dictionary into a human-readable text block.
    """
    lines = []
    lines.append(f"Total experiments: {summary['n_experiments']}")

    lines.append("Model used:")
    for model, count in summary["model_counts"].items():
        lines.append(f"  - {model}:{count} runs")
    best = summary["best_cv_experiment"]
    lines.append("Best CV experiment:")
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
