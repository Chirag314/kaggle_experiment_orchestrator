from __future__ import annotations

import os
import json
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

    prompt = f"""
You are a Kaggle experiment portfolio assistant.

You are given:
1) A JSON summary of experiments (per-model stats, best experiment, overfitting info, times).
2) A human-readable text report.
3) A user question.

Use ONLY this information to answer the question clearly and concisely.
If something is not available in the data, say so.

=== EXPERIMENT SUMMARY (JSON) ===
{context_json}

=== EXPERIMENT REPORT (TEXT) ===
{text_report}

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
