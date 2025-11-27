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


def tool_suggest_next_experiments(experiments_path=None):
    """
    Suggest next experiments to try based on:
      - best CV model
      - overfitting gap
      - model_family_stats
      - training time distribution

    Returns
    -------
    dict
      {
        "suggestions": [list of recommended experiments],
        "rationale": explanation text
      }
    """
    result = tool_run_portfolio_analysis(experiments_path)
    summary = result["summary"]

    best = summary["best_cv_experiment"]
    worst_gap = summary["worst_gap_experiment"]
    model_stats = summary["model_family_stats"]
    time_stats = summary["time_stats"]

    suggestions = []

    # 1. Fine-tune the best model
    suggestions.append(
        f"📌 Hyperparameter refinement on best model ({best['model_type']}): "
        "try tuning learning rate, max_depth, num_leaves, and regularization parameters."
    )

    # 2. Address overfitting
    if (best["cv_metric"] - best["holdout_metric"]) > 0.02:
        suggestions.append(
            "⚠️ High CV–holdout gap detected → try stronger validation split (GroupKFold), "
            "feature regularization, or reduce model complexity."
        )
    else:
        suggestions.append(
            "👍 CV–holdout gap is reasonable → explore additional features safely."
        )

    # 3. Try an alternative model family with strong mean CV
    sorted_models = sorted(
        model_stats.items(), key=lambda x: x[1]["mean_cv"], reverse=True
    )
    if len(sorted_models) > 1:
        next_best_model = sorted_models[1][0]
        suggestions.append(
            f"🔁 Try second-best performing model family: {next_best_model} "
            "with improved feature engineering."
        )

    # 4. Speed-optimization experiment
    if time_stats["max_train_time"] > 2 * time_stats["mean_train_time"]:
        suggestions.append(
            "⚡ Some models are slow → build a 'fast baseline' (e.g., LightGBM with simple features) "
            "to accelerate iteration."
        )

    # 5. Feature engineering suggestion
    suggestions.append(
        "🧪 Add 1–2 new feature families (target encodings, polynomial interactions, "
        "frequency encodings, or domain-specific transformations)."
    )

    # 6. Ensemble recommendation
    suggestions.append(
        "🤝 Build a small ensemble of top 2 model families (e.g., LightGBM + CatBoost). "
        "This often improves LB stability."
    )

    rationale = (
        "Recommendations are based on: best CV score, overfitting behavior, "
        "relative performance of model families, and training time statistics."
    )

    return {
        "suggestions": suggestions,
        "rationale": rationale,
    }
