# orchestrator_agent/orchestrator.py

"""
Orchestrator module for Kaggle Experiment Orchestrator Lite.

This is our first "agent-like" layer:
- It decides what steps to run (load, summarize, format).
- It can later be replaced or wrapped by a real LLM-based agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from .tools import load_experiments, summarize_experiments, format_summary_text


def run_portfolio_analysis(
    experiments_path: str | Path,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    High-level function that:
      - Loads experiment results
      - Summarizes them
      - Optionally prints a human-readable report

    Returns a dictionary so other code (or agents) can use the raw data.
    """
    experiments_path = Path(experiments_path)

    df = load_experiments(experiments_path)
    summary = summarize_experiments(df)
    text_report = format_summary_text(summary)

    if verbose:
        print("\n===== EXPERIMENT PORTFOLIO SUMMARY =====\n")
        print(text_report)
        print("\n========================================\n")

    result: Dict[str, Any] = {
        "experiments_path": str(experiments_path.resolve()),
        "summary": summary,
        "text_report": text_report,
    }
    return result
