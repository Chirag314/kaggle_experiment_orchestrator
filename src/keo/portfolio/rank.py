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


def rank_experiments(df: pd.DataFrame, strategy: Strategy = "balanced") -> pd.DataFrame:
    df = df.copy()
    if "cv_holdout_gap" not in df.columns:
        df["cv_holdout_gap"] = df["cv_metric"] - df["holdout_metric"]

    cv_norm = _normalize(df["cv_metric"])
    gap_norm = _normalize(df["cv_holdout_gap"])
    time_norm = _normalize(df["train_time_seconds"])

    if strategy == "leaderboard":
        w_cv, w_gap, w_time = 1.0, 0.3, 0.1
    elif strategy == "stability":
        w_cv, w_gap, w_time = 0.8, 0.7, 0.1
    elif strategy == "speed":
        w_cv, w_gap, w_time = 0.6, 0.2, 0.7
    else:
        w_cv, w_gap, w_time = 1.0, 0.5, 0.3

    df["rank_score"] = +w_cv * cv_norm - w_gap * gap_norm - w_time * time_norm
    return df.sort_values("rank_score", ascending=False).reset_index(drop=True)
