import pandas as pd

from keo.portfolio.summarize import format_summary_text, summarize_experiments


def test_summary_and_format_smoke():
    df = pd.DataFrame(
        {
            "experiment_id": ["a", "b"],
            "model_type": ["LGBM", "XGB"],
            "features_desc": ["base", "base"],
            "params_summary": ["p1", "p2"],
            "cv_metric": [0.8, 0.9],
            "holdout_metric": [0.78, 0.7],
            "train_time_seconds": [10.0, 100.0],
            "notes": ["ok", "gap"],
        }
    )
    s = summarize_experiments(df)
    txt = format_summary_text(s)
    assert "Total experiments" in txt
