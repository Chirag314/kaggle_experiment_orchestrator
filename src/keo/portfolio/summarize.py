from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from keo.core.schema import EXPECTED_EXPERIMENT_COLUMNS


def load_experiments(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Experiments file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = EXPECTED_EXPERIMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    return df


def compute_model_family_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if "cv_holdout_gap" not in df.columns:
        df = df.copy()
        df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]

    stats: dict[str, dict[str, float]] = {}
    for model, group in df.groupby("model_type"):
        stats[model] = {
            "n_runs": int(len(group)),
            "best_cv": float(group["cv_metric"].max()),
            "mean_cv": float(group["cv_metric"].mean()),
            "mean_gap": float(group["cv_holdout_gap"].mean()),
            "mean_train_time": float(group["train_time_seconds"].mean()),
        }
    return stats


def summarize_experiments(df: pd.DataFrame) -> dict[str, Any]:
    n_experiments = len(df)
    model_counts = df["model_type"].value_counts().to_dict()

    df_sorted_cv = df.sort_values("cv_metric", ascending=False).reset_index(drop=True)
    best_cv_row = df_sorted_cv.iloc[0].to_dict()

    df = df.copy()
    df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]

    df_sorted_gap = df.sort_values("cv_holdout_gap", ascending=False)
    worst_gap_row = df_sorted_gap.iloc[0].to_dict()

    time_stats = {
        "min_train_time": float(df["train_time_seconds"].min()),
        "max_train_time": float(df["train_time_seconds"].max()),
        "mean_train_time": float(df["train_time_seconds"].mean()),
    }

    model_family_stats = compute_model_family_stats(df)

    return {
        "n_experiments": int(n_experiments),
        "model_counts": model_counts,
        "best_cv_experiment": best_cv_row,
        "worst_gap_experiment": worst_gap_row,
        "time_stats": time_stats,
        "model_family_stats": model_family_stats,
    }


def format_summary_text(summary: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Total experiments: {summary['n_experiments']}")
    lines.append("Models used (raw counts):")
    for model, count in summary["model_counts"].items():
        lines.append(f"  - {model}: {count} runs")

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


def run_portfolio_analysis(experiments_path: str | Path, verbose: bool = True) -> dict[str, Any]:
    experiments_path = Path(experiments_path)
    df = load_experiments(experiments_path)
    summary = summarize_experiments(df)
    text_report = format_summary_text(summary)

    if verbose:
        print("\n===== EXPERIMENT PORTFOLIO SUMMARY =====\n")
        print(text_report)
        print("\n========================================\n")

    return {
        "experiments_path": str(experiments_path.resolve()),
        "summary": summary,
        "text_report": text_report,
    }
