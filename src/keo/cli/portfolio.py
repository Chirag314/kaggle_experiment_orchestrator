from __future__ import annotations

from pathlib import Path

from keo.portfolio.summarize import run_portfolio_analysis


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_csv = project_root / "keo" / "data" / "sample_experiments.csv"
    run_portfolio_analysis(default_csv, verbose=True)


if __name__ == "__main__":
    main()
