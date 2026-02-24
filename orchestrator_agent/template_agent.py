# orchestrator_agent/template_agent.py

"""
Notebook generator agent.

Given some basic info about a Kaggle competition, ask Gemini to
generate a starter notebook template: section headings, code stubs,
and where to plug in the experiment orchestrator.
"""

from __future__ import annotations

import os

import nbformat
from google import genai
from google.genai import types as genai_types
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from orchestrator_agent import adk_tools
from orchestrator_agent.ranking import rank_experiments


def _build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in the environment.")
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


def _template_to_cells(template: str):
    """
    Convert a Markdown+```python``` template string into a list of
    nbformat cells (markdown + code).
    """
    lines = template.splitlines()
    cells = []

    current_block = []
    in_code = False
    code_lang = None

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            # Toggle code / markdown
            if not in_code:
                # starting code block
                in_code = True
                code_lang = stripped.strip("`").strip()
                # flush any markdown accumulated
                if current_block:
                    cells.append(new_markdown_cell("\n".join(current_block)))
                    current_block = []
            else:
                # ending code block
                in_code = False
                cells.append(new_code_cell("\n".join(current_block)))
                current_block = []
                code_lang = None
            continue

        # accumulate lines
        current_block.append(line)

    # Flush whatever is left
    if current_block:
        if in_code:
            cells.append(new_code_cell("\n".join(current_block)))
        else:
            cells.append(new_markdown_cell("\n".join(current_block)))

    return cells


def save_notebook_from_template(
    template: str, output_path: str = "portfolio_starter_notebook.ipynb"
):
    """
    Create a .ipynb file from a template string that mixes Markdown and
    ```python code fences.

    Parameters
    ----------
    template : str
        Full notebook content (Markdown + code fences).
    output_path : str
        Where to save the .ipynb file.
    """
    nb = new_notebook()
    nb.cells = _template_to_cells(template)

    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def select_models_for_goal(
    experiments_path: str,
    goal_description: str,
    primary_metric: str,
    max_models: int = 3,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> dict:
    """
    Model-selection agent.

    Uses your experiment portfolio + a natural-language goal description
    to choose which model families to focus on.

    Parameters
    ----------
    experiments_path : str
        Path to the experiments CSV.
    goal_description : str
        e.g. "maximize leaderboard score",
             "fast iteration with decent performance",
             "stable scores between CV and LB".
    primary_metric : str
        Name of the main metric (AUC, accuracy, RMSE, etc.).
    max_models : int
        Maximum number of model families to recommend.
    model : str
        Gemini model name.
    temperature : float
        Sampling temperature.

    Returns
    -------
    dict
        {
          "chosen_models": [list of model_type strings],
          "strategy_used": "leaderboard" / "stability" / "speed" / "balanced",
          "explanation": "natural language rationale"
        }
    """
    client = _build_client()

    # Load and rank experiments with all strategies
    df = adk_tools.load_experiments(experiments_path)

    ranked_balanced = rank_experiments(df, "balanced").head(10).to_dict(orient="records")
    ranked_lb = rank_experiments(df, "leaderboard").head(10).to_dict(orient="records")
    ranked_stable = rank_experiments(df, "stability").head(10).to_dict(orient="records")
    ranked_speed = rank_experiments(df, "speed").head(10).to_dict(orient="records")

    ranking_block = {
        "balanced_top10": ranked_balanced,
        "leaderboard_top10": ranked_lb,
        "stability_top10": ranked_stable,
        "speed_top10": ranked_speed,
    }

    prompt = f"""
    You are a Kaggle model-selection assistant.

    You are given:
    - An experiment portfolio (top runs for several ranking strategies).
    - The primary metric: {primary_metric}.
    - A user's goal description: "{goal_description}".

    Ranking strategies:
    - balanced: trade-off between CV, overfitting (small CV–holdout gap), and train speed.
    - leaderboard: prioritize highest CV, accept some overfitting and slower models.
    - stability: prioritize small CV–holdout gap, even if CV is slightly lower.
    - speed: prioritize faster models with acceptable CV for quick iteration.

    Data (JSON, top 10 rows per strategy):
    {json.dumps(ranking_block, indent=2)}

    TASK:
    1. Decide which ranking strategy (or mixture of strategies) best matches the goal.
    2. Pick up to {max_models} model families (model_type) to focus on.
    3. Explain your reasoning in a short paragraph.

    Return your answer as a compact JSON object with keys:
    - "chosen_models": list of model_type strings
    - "strategy_used": one of ["balanced", "leaderboard", "stability", "speed", "mixed"]
    - "explanation": short text
    """

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )

    # response.text should be JSON; but be defensive
    try:
        parsed = json.loads(response.text)
    except Exception:
        # fallback: wrap in dict
        parsed = {
            "raw_response": response.text,
        }

    return parsed
