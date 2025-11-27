# orchestrator_agent/notebook_agent.py

"""
Notebook-friendly entry points for the Gemini-powered agent.

Use these in:
  - local Jupyter notebooks
  - Kaggle notebooks

They avoid interactive input() loops and just expose a simple
function: answer_question(question, experiments_path).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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

    Parameters
    ----------
    question : str
        Natural language question (e.g. 'Which experiment is best?').
    experiments_path : str, optional
        Path to the experiments CSV. If None, use the default sample CSV
        under data/sample_experiments.csv.
    model : str
        Gemini model name.
    temperature : float
        Sampling temperature.

    Returns
    -------
    str
        The agent's answer as plain text.
    """
    client = _build_client()

    # Resolve default experiments path if needed
    if experiments_path is None:
        project_root = Path(__file__).resolve().parents[1]
        experiments_path = str(project_root / "data" / "sample_experiments.csv")

    tools = [
        adk_tools.tool_run_portfolio_analysis,
        adk_tools.tool_get_best_experiment,
        adk_tools.tool_get_overfitting_info,
        adk_tools.tool_get_time_stats,
    ]

    system_instruction = (
        "You are a Kaggle experiment portfolio assistant. "
        "You have access to Python tools that analyze a CSV of experiments. "
        "Use these tools whenever the question requires knowledge of the "
        "experiments, and answer clearly and concisely."
    )

    # We include the experiments_path in the user message so the model
    # understands which file to analyze if it decides to call tools.
    user_prompt = f"Experiments CSV path: {experiments_path}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            tools=tools,
            system_instruction=system_instruction,
            temperature=temperature,
        ),
    )

    return response.text
