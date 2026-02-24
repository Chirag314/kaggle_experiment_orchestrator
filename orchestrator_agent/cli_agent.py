# orchestrator_agent/cli_agent.py

"""
CLI-based "agent" for Kaggle Experiment Orchestrator Lite.

This is our FIRST real agent layer:
- It takes natural language input from the user.
- It decides which actions to take (which tools to call).
- It calls the underlying orchestrator + tools and shows results.

Later, we can replace this with an LLM-powered ADK agent that behaves
similarly, but for now everything is explicit Python logic so you
can see the structure clearly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .orchestrator import run_portfolio_analysis


class PortfolioAgent:
    """
    A simple rule-based agent that understands a few intents:

    - "full" / "summary": run full portfolio analysis (default).
    - "best": show only the best experiment.
    - "overfit": highlight overfitting / CV-gap.
    - "time" / "speed": focus on training time stats.
    - "help": show available commands.

    This is intentionally simple so you can focus on the structure
    of an agent: interpret user request -> call tools -> respond.
    """

    def __init__(self, experiments_path: str | Path) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.experiments_path = Path(experiments_path)
        self._last_result: dict[str, Any] | None = None

    def ensure_analysis(self) -> dict[str, Any]:
        """
        Run portfolio analysis once and cache the result, so repeated
        queries don't re-load the CSV every time.
        """
        if self._last_result is None:
            self._last_result = run_portfolio_analysis(
                self.experiments_path,
                verbose=False,  # we will print selectively
            )
        return self._last_result

    # ----------- Intent handlers -----------

    def handle_full_summary(self) -> str:
        """
        Return the full text report from the orchestrator.
        """
        result = self.ensure_analysis()
        return result["text_report"]

    def handle_best_experiment(self) -> str:
        """
        Return a short description of the best CV experiment.
        """
        result = self.ensure_analysis()
        best = result["summary"]["best_cv_experiment"]

        lines = [
            "Best CV experiment:",
            f"  ID: {best['experiment_id']}",
            f"  Model: {best['model_type']}",
            f"  CV metric: {best['cv_metric']:.4f}",
            f"  Holdout metric: {best['holdout_metric']:.4f}",
            f"  Gap: {best['cv_metric'] - best['holdout_metric']:.4f}",
            f"  Features: {best['features_desc']}",
            f"  Params: {best['params_summary']}",
        ]
        return "\n".join(lines)

    def handle_overfitting(self) -> str:
        """
        Emphasize which experiment seems most overfitted.
        """
        result = self.ensure_analysis()
        worst = result["summary"]["worst_gap_experiment"]

        lines = [
            "Most overfitted experiment (largest CV - holdout gap):",
            f"  ID: {worst['experiment_id']}",
            f"  Model: {worst['model_type']}",
            f"  CV metric: {worst['cv_metric']:.4f}",
            f"  Holdout metric: {worst['holdout_metric']:.4f}",
            f"  Gap: {worst['cv_metric'] - worst['holdout_metric']:.4f}",
            f"  Notes: {worst['notes']}",
            "",
            "Tip: On Kaggle, a big CV > LB gap often means:",
            "- Data leakage",
            "- Bad CV split strategy, or",
            "- Overly complex features that don't generalize.",
        ]
        return "\n".join(lines)

    def handle_time_stats(self) -> str:
        """
        Focus on training time information and model tradeoffs.
        """
        result = self.ensure_analysis()
        time_stats = result["summary"]["time_stats"]
        per_model = result["summary"]["model_family_stats"]

        lines = [
            "Training time overview (seconds):",
            f"  min:  {time_stats['min_train_time']:.1f}",
            f"  mean: {time_stats['mean_train_time']:.1f}",
            f"  max:  {time_stats['max_train_time']:.1f}",
            "",
            "Per-model mean train time:",
        ]
        for model, stats in per_model.items():
            lines.append(
                f"  {model}: mean {stats['mean_train_time']:.1f}s over {stats['n_runs']} runs"
            )

        lines.append(
            "\nKaggle tip: use the fastest strong baseline as your daily workhorse,\n"
            "and run heavier models overnight or when experimenting less frequently."
        )
        return "\n".join(lines)

    def help_text(self) -> str:
        return (
            "I can help you understand your experiment portfolio.\n"
            "Try commands like:\n"
            "  - full / summary   → full report\n"
            "  - best             → best CV experiment\n"
            "  - overfit          → where CV >> holdout\n"
            "  - time / speed     → training time insights\n"
            "  - help             → show this message\n"
            "  - exit / quit      → leave the agent\n"
        )

    # ----------- Main entry point -----------

    def handle_message(self, message: str) -> str:
        """
        Decide which intent to trigger based on the user's message.
        """
        msg = message.strip().lower()

        if msg in {"help", "?", "h"}:
            return self.help_text()
        if any(word in msg for word in ["best", "top", "winner"]):
            return self.handle_best_experiment()
        if "overfit" in msg or "gap" in msg:
            return self.handle_overfitting()
        if any(word in msg for word in ["time", "speed", "fast", "slow"]):
            return self.handle_time_stats()

        # Default: full summary
        return self.handle_full_summary()


def main() -> None:
    """
    Simple REPL loop that lets you chat with PortfolioAgent.

    Usage (from project root):
        python -m orchestrator_agent.cli_agent
    """
    project_root = Path(__file__).resolve().parents[1]
    experiments_path = project_root / "data" / "sample_experiments.csv"

    agent = PortfolioAgent(experiments_path)

    print("=== Kaggle Experiment Orchestrator Lite – CLI Agent ===")
    print(f"Using experiments file: {experiments_path}")
    print("Type 'help' to see options, 'exit' to quit.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("agent> Goodbye, and good luck on Kaggle!")
            break

        if not user_input:
            continue

        response = agent.handle_message(user_input)
        print("\nagent>")
        print(response)
        print("")  # blank line for readability


if __name__ == "__main__":
    main()
