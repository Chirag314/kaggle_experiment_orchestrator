from __future__ import annotations

import argparse
from pathlib import Path

from keo.agents.rule_based import PortfolioAgent


def run_rule_agent(csv_path: Path) -> None:
    agent = PortfolioAgent(csv_path)
    print("=== KEO — Rule-based Portfolio Agent ===")
    print(f"CSV: {csv_path}")
    print("Type 'help' or 'exit'.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nagent> Goodbye!")
            break
        if user.lower() in {"exit", "quit"}:
            print("agent> Goodbye!")
            break
        if not user:
            continue
        print("\nagent>")
        print(agent.handle_message(user))
        print("")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="", help="Path to experiments CSV")
    p.add_argument("--mode", type=str, default="rule", choices=["rule"], help="Agent mode")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    default_csv = project_root / "keo" / "data" / "sample_experiments.csv"
    csv_path = Path(args.csv) if args.csv else default_csv

    run_rule_agent(csv_path)


if __name__ == "__main__":
    main()
