# orchestrator_agent/viz.py

"""
Visualization helpers for the experiment portfolio.

These are plain matplotlib-based functions meant to be used in:
- local Jupyter notebooks
- Kaggle notebooks
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_gap_column(df: pd.DataFrame) -> pd.DataFrame:
    if "cv_holdout_gap" not in df.columns:
        df = df.copy()
        df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]
    return df


def plot_cv_vs_holdout(df: pd.DataFrame, title: str = "CV vs Holdout"):
    """
    Scatter plot: CV metric vs Holdout metric, colored by model_type.
    """
    df = _ensure_gap_column(df)

    # Simple scatter (matplotlib default colors)
    fig, ax = plt.subplots()
    for model, group in df.groupby("model_type"):
        ax.scatter(group["cv_metric"], group["holdout_metric"], label=model)

    ax.set_xlabel("CV metric")
    ax.set_ylabel("Holdout metric / LB estimate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_time_vs_cv(df: pd.DataFrame, title: str = "Train Time vs CV"):
    """
    Scatter plot: training time vs CV metric.
    """
    df = _ensure_gap_column(df)

    fig, ax = plt.subplots()
    ax.scatter(df["train_time_seconds"], df["cv_metric"])
    ax.set_xlabel("Train time (seconds)")
    ax.set_ylabel("CV metric")
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def plot_model_family_performance(
    summary: Dict[str, Any], title: str = "Model Family Mean CV"
):
    """
    Bar chart: mean CV by model_type using the summary dict.
    """
    stats = summary["model_family_stats"]
    models = list(stats.keys())
    mean_cvs = [stats[m]["mean_cv"] for m in models]

    fig, ax = plt.subplots()
    ax.bar(models, mean_cvs)
    ax.set_ylabel("Mean CV metric")
    ax.set_title(title)
    ax.set_xticklabels(models, rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
