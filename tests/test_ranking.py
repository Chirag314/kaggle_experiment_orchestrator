import pandas as pd

from keo.portfolio.rank import rank_experiments


def test_rank_experiments_adds_score():
    df = pd.DataFrame(
        {
            "cv_metric": [0.8, 0.9],
            "holdout_metric": [0.78, 0.7],
            "train_time_seconds": [10, 100],
        }
    )
    out = rank_experiments(df, "balanced")
    assert "rank_score" in out.columns
