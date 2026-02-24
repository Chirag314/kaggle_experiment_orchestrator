# orchestrator_agent/ranking.py

"""
Experiment ranking utilities.

Compute a composite score for experiments based on:
- CV metric
- CV vs holdout gap
- training time

Supports different strategies (balanced, leaderboard, stability, speed).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

Strategy = Literal["balanced", "leaderboard", "stability", "speed"]


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_v = s.min()
    max_v = s.max()
    if max_v == min_v:
        return pd.Series(0.0, index=s.index)
    return (s - min_v) / (max_v - min_v)


def rank_experiments(
    df: pd.DataFrame,
    strategy: Strategy = "balanced",
) -> pd.DataFrame:
    """
    Rank experiments by a composite score.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: cv_metric, holdout_metric, train_time_seconds, etc.
    strategy : {'balanced', 'leaderboard', 'stability', 'speed'}
        Ranking strategy.

    Returns
    -------
    pd.DataFrame
        Copy of df with an extra 'rank_score' column, sorted descending.
    """
    df = df.copy()
    if "cv_holdout_gap" not in df.columns:
        df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]

    cv_norm = _normalize(df["cv_metric"])
    gap_norm = _normalize(df["cv_holdout_gap"])  # higher = more overfitting (bad)
    time_norm = _normalize(df["train_time_seconds"])  # higher = slower (bad)

    if strategy == "leaderboard":
        w_cv, w_gap, w_time = 1.0, 0.3, 0.1
    elif strategy == "stability":
        w_cv, w_gap, w_time = 0.8, 0.7, 0.1
    elif strategy == "speed":
        w_cv, w_gap, w_time = 0.6, 0.2, 0.7
    else:  # balanced
        w_cv, w_gap, w_time = 1.0, 0.5, 0.3

    # higher cv_norm is good, gap_norm/time_norm are bad
    df["rank_score"] = +w_cv * cv_norm - w_gap * gap_norm - w_time * time_norm

    df_sorted = df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return df_sorted
