# orchestrator_agent/main.py

"""
Simple CLI entry point to test the early Orchestrator logic.

Usage (from project root):
    python -m orchestrator_agent.main
"""

from pathlib import Path

from .orchestrator import run_portfolio_analysis


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    experiments_path = project_root / "data" / "sample_experiments.csv"

    print(f"Running portfolio analysis for: {experiments_path}")

    result = run_portfolio_analysis(experiments_path, verbose=True)

    # For now we don't do anything with `result` beyond printing it,
    # but later our "agent" can use this dictionary to plan next experiments.
    # This makes it easier to connect to an ADK / LLM agent later.
    _ = result


if __name__ == "__main__":
    main()
