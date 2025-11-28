from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Optional
from IPython.display import Markdown
from google import genai
from google.genai import types as genai_types

from . import adk_tools


def _build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY must be set in the environment."
        )
    return genai.Client(api_key=api_key)


def answer_question(
    question: str,
    experiments_path: Optional[str] = None,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
) -> str:
    """
    Answer a single question about the experiment portfolio.

    Instead of using function tools, this:
      - runs tool_run_portfolio_analysis() directly in Python
      - packages the summary + text report into a prompt
      - lets Gemini answer based on that context

    This avoids google-genai's automatic tool-calling bugs in some environments.
    """
    client = _build_client()

    # Resolve default experiments path if needed
    if experiments_path is None:
        project_root = Path(__file__).resolve().parents[1]
        experiments_path = str(project_root / "data" / "sample_experiments.csv")

    # 1) Run your Python tools directly
    result = adk_tools.tool_run_portfolio_analysis(experiments_path)
    summary = result["summary"]
    text_report = result["text_report"]

    # 2) Build a rich prompt with JSON + human-readable summary
    context_json = json.dumps(summary, indent=2)
    from orchestrator_agent.ranking import rank_experiments

    # Add ranking strategies
    df = adk_tools.load_experiments(experiments_path)

    rank_balanced = rank_experiments(df, "balanced").head(5).to_dict(orient="records")
    rank_leaderboard = (
        rank_experiments(df, "leaderboard").head(5).to_dict(orient="records")
    )
    rank_stability = rank_experiments(df, "stability").head(5).to_dict(orient="records")
    rank_speed = rank_experiments(df, "speed").head(5).to_dict(orient="records")

    ranking_block = {
        "balanced_top5": rank_balanced,
        "leaderboard_top5": rank_leaderboard,
        "stability_top5": rank_stability,
        "speed_top5": rank_speed,
    }
    strategy_explanation = """
        Ranking strategies:
        - balanced: trade-off between CV, overfitting (small CV–holdout gap), and train speed.
        - leaderboard: prioritize highest CV, accept some overfitting and slower models.
        - stability: prioritize small CV–holdout gap, even if CV is slightly lower.
        - speed: prioritize faster models with acceptable CV for quick iteration.

        When the user does not state explicit goals, you should:
        - explain these trade-offs, and
        - suggest which strategy fits common Kaggle goals (maximize LB, stable scores, fast iteration).
        Never say you cannot answer; instead, explain options and what you recommend.
        """

    prompt = f"""
        You are a Kaggle experiment portfolio assistant.

        You are given:
        1) A JSON summary of experiments (per-model stats, best experiment, overfitting info, times).
        2) A human-readable text report.
        3) Ranking tables for four strategies (balanced, leaderboard, stability, speed).
        4) A description of what each strategy means.
        5) A user question.

        Use ALL of this information to answer clearly and concisely.
        If the user does not specify goals, explain the trade-offs between strategies
        and recommend which strategies typically match common Kaggle goals
        (e.g. 'maximize leaderboard score', 'fast iteration', 'stable public/private scores').
        Do NOT answer that you cannot help.

        === EXPERIMENT SUMMARY (JSON) ===
        {context_json}

        === EXPERIMENT REPORT (TEXT) ===
        {text_report}

        === RANKING TABLES (TOP 5 IN EACH STRATEGY) ===
        {json.dumps(ranking_block, indent=2)}

        === STRATEGY EXPLANATION ===
        {strategy_explanation}

        === USER QUESTION ===
        {question}
        """

    # 3) Ask Gemini to answer based on this context
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    return response.text.strip()
