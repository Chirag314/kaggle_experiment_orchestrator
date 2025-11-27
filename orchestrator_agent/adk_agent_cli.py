# orchestrator_agent/adk_agent_cli.py

"""
LLM-powered agent using Gemini + our tools.

Usage (from project root):
    python -m orchestrator_agent.adk_agent_cli

This will:
  - Load Gemini using GOOGLE_API_KEY
  - Expose our portfolio tools as function tools
  - Start a chat loop where Gemini decides which tools to call
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from google import genai
from google.genai import types as genai_types

from . import adk_tools


def build_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)
    return client


def build_tool_schema() -> List[genai_types.Tool]:
    """
    Define tools that Gemini can call.

    We map our Python functions to function declarations that the model understands.
    """
    return [
        genai_types.Tool(
            function_declarations=[
                genai_types.FunctionDeclaration(
                    name="tool_run_portfolio_analysis",
                    description=(
                        "Analyze a Kaggle experiments CSV and return a summary, "
                        "including best experiment, overfitting, and timings."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "experiments_path": {
                                "type": "string",
                                "description": (
                                    "Path to the experiments CSV. "
                                    "If omitted, use the default sample file."
                                ),
                            }
                        },
                        "required": [],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="tool_get_best_experiment",
                    description="Return the best CV experiment's details.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "experiments_path": {
                                "type": "string",
                                "description": "Optional path to experiments CSV.",
                            }
                        },
                        "required": [],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="tool_get_overfitting_info",
                    description=(
                        "Return information about the most overfitted experiment, "
                        "where CV - holdout gap is largest."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "experiments_path": {
                                "type": "string",
                                "description": "Optional path to experiments CSV.",
                            }
                        },
                        "required": [],
                    },
                ),
                genai_types.FunctionDeclaration(
                    name="tool_get_time_stats",
                    description="Return training time statistics and per-model timings.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "experiments_path": {
                                "type": "string",
                                "description": "Optional path to experiments CSV.",
                            }
                        },
                        "required": [],
                    },
                ),
            ]
        )
    ]


def call_tool_by_name(
    name: str,
    args: Dict[str, Any],
) -> Any:
    """
    Dispatch from a tool call name to the corresponding Python function.
    """
    if name == "tool_run_portfolio_analysis":
        return adk_tools.tool_run_portfolio_analysis(**args)
    if name == "tool_get_best_experiment":
        return adk_tools.tool_get_best_experiment(**args)
    if name == "tool_get_overfitting_info":
        return adk_tools.tool_get_overfitting_info(**args)
    if name == "tool_get_time_stats":
        return adk_tools.tool_get_time_stats(**args)

    raise ValueError(f"Unknown tool name: {name}")


def interactive_loop() -> None:
    """
    Start an interactive loop with the LLM-powered agent.
    """
    client = build_client()
    tools = build_tool_schema()

    # We'll use a single ongoing "chat" session (stream of messages)
    history: List[genai_types.Content] = []

    print("=== Kaggle Experiment Orchestrator Lite – Gemini Agent ===")
    print("Type your questions about your experiment portfolio.")
    print("Example: 'Which experiment is the best?' or 'Where am I overfitting?'")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("agent> Goodbye!")
            break
        if not user_input:
            continue

        # Add user message to history
        history.append(
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(user_input)],
            )
        )

        # First call: let the model respond, possibly with tool calls
        response = client.models.generate_content(
            model="gemini-1.5-flash-latest",  # or another Gemini model
            contents=history,
            tools=tools,
        )

        # Check for tool calls
        tool_calls = []
        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_calls.append(part.function_call)

        # If there are no tool calls, just print the response
        if not tool_calls:
            text = response.candidates[0].content.parts[0].text
            print(f"\nagent> {text}\n")
            history.append(response.candidates[0].content)
            continue

        # If there ARE tool calls, execute them and send results back
        tool_results_parts: List[genai_types.Part] = []

        for fc in tool_calls:
            name = fc.name
            args = dict(fc.args or {})
            try:
                result = call_tool_by_name(name, args)
            except Exception as e:
                result = {"error": str(e)}

            # Each tool result is added as a function response part
            tool_results_parts.append(
                genai_types.Part.from_function_response(
                    name=name,
                    response={"result": result},
                )
            )

        # Add the tool responses as a new message from "tool" role
        history.append(
            genai_types.Content(
                role="tool",
                parts=tool_results_parts,
            )
        )

        # Now let the model see the tool results and produce a final answer
        followup = client.models.generate_content(
            model="gemini-1.5-flash-latest",
            contents=history,
        )

        final_text = followup.candidates[0].content.parts[0].text
        print(f"\nagent> {final_text}\n")

        # Add the final answer to history
        history.append(followup.candidates[0].content)


def main() -> None:
    """
    Entry point when running as a module.
    """
    # Ensure default file exists (helpful error if not)
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / "sample_experiments.csv"
    if not default_csv.exists():
        print(f"WARNING: default experiments file not found at {default_csv}")
        print("Some tool calls may fail unless you provide a path.")
    interactive_loop()


if __name__ == "__main__":
    main()
