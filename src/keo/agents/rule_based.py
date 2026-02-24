from __future__ import annotations

from pathlib import Path
from typing import Any

from keo.portfolio.summarize import run_portfolio_analysis


class PortfolioAgent:
    def __init__(self, experiments_path: str | Path) -> None:
        self.experiments_path = Path(experiments_path)
        self._last_result: dict[str, Any] | None = None

    def ensure_analysis(self) -> dict[str, Any]:
        if self._last_result is None:
            self._last_result = run_portfolio_analysis(self.experiments_path, verbose=False)
        return self._last_result

    def help_text(self) -> str:
        return (
            "Commands:\n"
            "  - full / summary   → full report\n"
            "  - best             → best CV experiment\n"
            "  - overfit          → where CV >> holdout\n"
            "  - time / speed     → training time insights\n"
            "  - help             → show this message\n"
            "  - exit / quit      → leave the agent\n"
        )

    def handle_full_summary(self) -> str:
        return self.ensure_analysis()["text_report"]

    def handle_best_experiment(self) -> str:
        best = self.ensure_analysis()["summary"]["best_cv_experiment"]
        return "\n".join(
            [
                "Best CV experiment:",
                f"  ID: {best['experiment_id']}",
                f"  Model: {best['model_type']}",
                f"  CV metric: {best['cv_metric']:.4f}",
                f"  Holdout metric: {best['holdout_metric']:.4f}",
                f"  Gap: {best['cv_metric'] - best['holdout_metric']:.4f}",
                f"  Features: {best['features_desc']}",
                f"  Params: {best['params_summary']}",
            ]
        )

    def handle_overfitting(self) -> str:
        worst = self.ensure_analysis()["summary"]["worst_gap_experiment"]
        return "\n".join(
            [
                "Most overfitted experiment (largest CV - holdout gap):",
                f"  ID: {worst['experiment_id']}",
                f"  Model: {worst['model_type']}",
                f"  CV metric: {worst['cv_metric']:.4f}",
                f"  Holdout metric: {worst['holdout_metric']:.4f}",
                f"  Gap: {worst['cv_metric'] - worst['holdout_metric']:.4f}",
                f"  Notes: {worst['notes']}",
                "",
                "Tip: A big CV > LB gap often means leakage, bad split strategy, or over-complex features.",
            ]
        )

    def handle_time_stats(self) -> str:
        s = self.ensure_analysis()["summary"]
        t = s["time_stats"]
        per_model = s["model_family_stats"]
        lines = [
            "Training time overview (seconds):",
            f"  min:  {t['min_train_time']:.1f}",
            f"  mean: {t['mean_train_time']:.1f}",
            f"  max:  {t['max_train_time']:.1f}",
            "",
            "Per-model mean train time:",
        ]
        for model, stats in per_model.items():
            lines.append(
                f"  {model}: mean {stats['mean_train_time']:.1f}s over {stats['n_runs']} runs"
            )
        return "\n".join(lines)

    def handle_message(self, message: str) -> str:
        msg = message.strip().lower()
        if msg in {"help", "?", "h"}:
            return self.help_text()
        if any(w in msg for w in ["best", "top", "winner"]):
            return self.handle_best_experiment()
        if "overfit" in msg or "gap" in msg:
            return self.handle_overfitting()
        if any(w in msg for w in ["time", "speed", "fast", "slow"]):
            return self.handle_time_stats()
        return self.handle_full_summary()
