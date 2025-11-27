# orchestrator_agent/template_agent.py

"""
Notebook generator agent.

Given some basic info about a Kaggle competition, ask Gemini to
generate a starter notebook template: section headings, code stubs,
and where to plug in the experiment orchestrator.
"""

from __future__ import annotations

import os
from typing import Optional

from google import genai
from google.genai import types as genai_types


def _build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY or GEMINI_API_KEY must be set in the environment."
        )
    return genai.Client(api_key=api_key)


def generate_notebook_template(
    competition_name: str,
    primary_metric: str,
    target_column: str,
    include_orchestrator_section: bool = True,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> str:
    """
    Generate a Kaggle notebook template (in Markdown + code-stub style).

    Parameters
    ----------
    competition_name : str
        Name of the Kaggle competition.
    primary_metric : str
        Main metric (e.g., 'AUC', 'RMSE', 'MAPE').
    target_column : str
        Name of the target column in the dataset.
    include_orchestrator_section : bool
        Whether to include a section that uses the Experiment Orchestrator.
    model : str
        Gemini model name.
    temperature : float
        Sampling temperature for creativity.

    Returns
    -------
    str
        The generated template text.
    """
    client = _build_client()

    orchestrator_note = ""
    if include_orchestrator_section:
        orchestrator_note = """
6. **Experiment Orchestrator Integration**
   - Load `experiments.csv` or your experiment tracking table.
   - Call portfolio analysis tools.
   - Use agent to suggest next runs.
"""

    prompt = f"""
You are helping a Kaggle competitor create a starter notebook template.

Competition: {competition_name}
Primary metric: {primary_metric}
Target column: {target_column}

Create a notebook outline in Markdown with embedded Python code blocks. Use
sections like:

1. Setup & Imports
2. Load Data
3. EDA (light)
4. Feature Engineering
5. Baseline Model
{orchestrator_note}
7. Training & Evaluation
8. Submission

Requirements:
- Use Python code blocks with ```python.
- Show a minimal LightGBM or XGBoost baseline.
- Show where to call an "experiment orchestrator" that analyzes experiments.csv.
- Keep it concise but realistic (something a Kaggle user could actually use).

Return ONLY the template, no extra commentary.
"""

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    return response.text.strip()
