# orchestrator_agent/main.py

"""
Simple CLI entry point to test the early Orchestrator logic.

Usage (from project root):
    python -m orchestrator_agent.main
"""

from pathlib import Path
from .tools import load_experiments, summarize_experiments, format_summary_text


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiments_path = project_root / "data" / "sample_experiments.csv"

    print(f"Loading experiments from : {experiments_path}")

    df = load_experiments(experiments_path)
    summary = summarize_experiments(df)
    text = format_summary_text(summary)

    print("\n===== EXPERIMENT PORTFOLIO SUMMARY =====\n")
    print(text)
    print("\n========================================\n")


if __name__ == "__main__":
    main()
