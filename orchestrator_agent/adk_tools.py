# orchestrator_agent/adk_tools.py

"""
Tools exposed to the LLM-based agent via google-genai's automatic
Python function calling.

We wrap our existing Python functions so that the agent can call them.
"""

from __future__ import annotations

from pathlib import Path

from .orchestrator import run_portfolio_analysis


def tool_run_portfolio_analysis(experiments_path=None):
    """
    Run portfolio analysis on the given experiments CSV.

    Parameters
    ----------
    experiments_path : str, optional
        Path to the experiments CSV. If not provided or empty, defaults to
        data/sample_experiments.csv in the project root.

    Returns
    -------
    dict
        Dictionary containing:
          - experiments_path
          - summary (dict)
          - text_report (str)
    """
    project_root = Path(__file__).resolve().parents[1]
    if not experiments_path:
        experiments_path = project_root / "data" / "sample_experiments.csv"
    else:
        experiments_path = Path(experiments_path)

    result = run_portfolio_analysis(experiments_path, verbose=False)
    return result


def tool_get_best_experiment(experiments_path=None):
    """
    Return a focused view on the best CV experiment.
    """
    result = tool_run_portfolio_analysis(experiments_path)
    best = result["summary"]["best_cv_experiment"]
    return best


def tool_get_overfitting_info(experiments_path=None):
    """
    Return information about the most overfitted experiment
    (largest cv_metric - holdout_metric gap).
    """
    result = tool_run_portfolio_analysis(experiments_path)
    worst = result["summary"]["worst_gap_experiment"]
    gap = worst["cv_metric"] - worst["holdout_metric"]
    return {
        "experiment": worst,
        "gap": gap,
    }


def tool_get_time_stats(experiments_path=None):
    """
    Return training time statistics and per-model time summary.
    """
    result = tool_run_portfolio_analysis(experiments_path)
    return {
        "time_stats": result["summary"]["time_stats"],
        "model_family_stats": result["summary"]["model_family_stats"],
    }
