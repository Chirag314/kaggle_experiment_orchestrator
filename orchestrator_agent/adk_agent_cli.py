# orchestrator_agent/adk_agent_cli.py

"""
LLM-powered agent using Gemini + our tools, using google-genai's
automatic Python function calling.

Usage (from project root):

    python -m orchestrator_agent.adk_agent_cli

This will:
  - Load Gemini using GOOGLE_API_KEY or GEMINI_API_KEY
  - Expose our portfolio tools as Python function tools
  - Start a chat loop where Gemini decides which tools to call
"""

from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from . import adk_tools


def build_client() -> genai.Client:
    """
    Create a Gemini client.

    The SDK will automatically pick up:
      - GOOGLE_API_KEY  or
      - GEMINI_API_KEY

    We still sanity-check that at least one is set so error messages
    are clearer for you.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in the environment.")
    # You can either pass api_key explicitly or let the client read from env.
    client = genai.Client(api_key=api_key)
    return client


def interactive_loop() -> None:
    """
    Start an interactive loop with the Gemini-powered agent.

    We pass our Python functions as tools. The SDK will:
      - read their docstrings + signatures
      - let the model decide when/how to call them
      - automatically execute the functions
      - include their results in response.text
    """
    client = build_client()

    # Our Python tools that the model can call
    tools = [
        adk_tools.tool_run_portfolio_analysis,
        adk_tools.tool_get_best_experiment,
        adk_tools.tool_get_overfitting_info,
        adk_tools.tool_get_time_stats,
    ]

    # Optional: give the model some instructions about its role
    system_instruction = (
        "You are a Kaggle experiment portfolio assistant. "
        "You help analyze a CSV of experiment results and answer questions "
        "about best experiments, overfitting (CV vs holdout), and training time. "
        "Use the provided tools when helpful, and explain your reasoning clearly."
    )

    print("=== Kaggle Experiment Orchestrator Lite – Gemini Agent ===")
    print("Type your questions about your experiment portfolio.")
    print("Examples:")
    print("  - Which experiment is the best?")
    print("  - Where am I overfitting?")
    print("  - Which models are fastest to train?")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("agent> Goodbye!")
            break
        if not user_input:
            continue

        # Call Gemini with automatic function calling + our tools
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input,
            config=genai_types.GenerateContentConfig(
                tools=tools,
                system_instruction=system_instruction,
                temperature=0.2,  # make it a bit more deterministic
            ),
        )

        # With automatic function calling, Gemini executes the Python tools
        # and we can just read response.text as the final answer.
        print("\nagent>")
        print(response.text)
        print("")


def main() -> None:
    """
    Entry point when running as a module.
    """
    # Optional: warn if default CSV missing
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / "sample_experiments.csv"
    if not default_csv.exists():
        print(f"WARNING: default experiments file not found at {default_csv}")
        print("The tools that read experiments may fail unless you pass a path.\n")

    interactive_loop()


if __name__ == "__main__":
    main()
