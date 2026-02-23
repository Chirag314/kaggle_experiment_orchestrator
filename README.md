# 🤖 Orchestrator Agent — Autonomous Kaggle Experiment Intelligence Layer

A production-minded experiment intelligence system that transforms raw Kaggle experiment logs into structured insights, strategy-aware rankings, and agent-assisted decision making.

This is not just experiment tracking.  
It is an **intelligent orchestration layer** designed to simulate how high-performing Kaggle practitioners reason about trade-offs between performance, stability, and iteration speed.

It bridges:

- Structured ML experiment logging  
- Portfolio-level statistical analysis  
- Multi-objective ranking strategies  
- Rule-based agent systems  
- LLM-powered tool-calling agents  
- Notebook auto-generation  
- Goal-driven model selection  

This repository demonstrates **system-level ML thinking** beyond notebooks and ad-hoc experimentation.

---

## 🔖 Badges

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" />
  <img src="https://img.shields.io/badge/architecture-agent--driven-purple" />
  <img src="https://img.shields.io/badge/kaggle-experiment%20intelligence-orange" />
  <img src="https://img.shields.io/badge/google--genai-tool--calling-green" />
  <img src="https://img.shields.io/badge/design-reproducible%20ML-success" />
</p>

---

# 🏗 System Architecture

Below is the logical architecture of the Orchestrator Agent layer.

---

## 🏗 Architecture Diagram

<svg viewBox="0 0 980 520" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect x="0" y="0" width="980" height="520" fill="#ffffff"/>

  <!-- Title -->
  <text x="40" y="48" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="18" font-weight="600" fill="#0b0f19">
    Orchestrator Agent — Portfolio Intelligence Flow
  </text>
  <text x="40" y="72" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">
    experiments.csv → analysis → ranking → agents (CLI / Gemini / template)
  </text>

  <!-- Helpers: consistent styles (inline only) -->
  <!-- Boxes -->
  <!-- Column 1 -->
  <rect x="40" y="110" width="300" height="82" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="62" y="142" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">experiments.csv</text>
  <text x="62" y="166" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Source of truth for portfolio runs</text>

  <rect x="40" y="214" width="300" height="110" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="62" y="246" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">tools.py</text>
  <text x="62" y="268" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Load CSV + validate schema</text>
  <text x="62" y="288" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Compute portfolio stats (best, gap, time)</text>
  <text x="62" y="308" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Format human-readable summary</text>

  <!-- Column 2 -->
  <rect x="380" y="150" width="300" height="110" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="402" y="182" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">orchestrator.py</text>
  <text x="402" y="204" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">High-level pipeline:</text>
  <text x="402" y="224" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">load → summarize → report → return dict</text>
  <text x="402" y="244" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Agent-ready interface (no LLM required)</text>

  <rect x="380" y="282" width="300" height="110" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="402" y="314" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">ranking.py</text>
  <text x="402" y="336" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Composite scoring:</text>
  <text x="402" y="356" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">balanced • leaderboard • stability • speed</text>
  <text x="402" y="376" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">CV ↑, gap ↓, time ↓</text>

  <!-- Column 3: Agents -->
  <rect x="720" y="110" width="220" height="86" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="742" y="142" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">cli_agent.py</text>
  <text x="742" y="164" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Rule-based agent (REPL)</text>

  <rect x="720" y="214" width="220" height="110" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="742" y="246" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">adk_agent_cli.py</text>
  <text x="742" y="268" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Gemini agent w/ tool-calling</text>
  <text x="742" y="288" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Uses adk_tools.py wrappers</text>
  <text x="742" y="308" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Answers portfolio questions</text>

  <rect x="720" y="346" width="220" height="110" rx="14" fill="#ffffff" stroke="#101828" stroke-width="1.5"/>
  <text x="742" y="378" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
        font-size="13" fill="#101828" font-weight="600">template_agent.py</text>
  <text x="742" y="400" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Notebook template generator</text>
  <text x="742" y="420" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">Goal-based model selection</text>

  <!-- Connectors (simple lines + triangle arrows) -->
  <!-- experiments.csv -> tools.py -->
  <line x1="190" y1="192" x2="190" y2="214" stroke="#101828" stroke-width="2"/>
  <polygon points="190,214 184,204 196,204" fill="#101828"/>

  <!-- tools.py -> orchestrator.py -->
  <line x1="340" y1="250" x2="380" y2="205" stroke="#101828" stroke-width="2"/>
  <polygon points="380,205 368,205 375,215" fill="#101828"/>

  <!-- orchestrator.py -> ranking.py -->
  <line x1="530" y1="260" x2="530" y2="282" stroke="#101828" stroke-width="2"/>
  <polygon points="530,282 524,272 536,272" fill="#101828"/>

  <!-- ranking.py -> agents -->
  <line x1="680" y1="336" x2="720" y2="153" stroke="#101828" stroke-width="2"/>
  <polygon points="720,153 708,153 715,163" fill="#101828"/>

  <line x1="680" y1="350" x2="720" y2="270" stroke="#101828" stroke-width="2"/>
  <polygon points="720,270 708,270 715,280" fill="#101828"/>

  <line x1="680" y1="370" x2="720" y2="401" stroke="#101828" stroke-width="2"/>
  <polygon points="720,401 708,401 715,411" fill="#101828"/>

  <!-- Footer note -->
  <text x="40" y="492" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial"
        font-size="12" fill="#667085">
    Tip: For perfect GitHub rendering, you can also save this SVG as docs/architecture.svg and embed it via Markdown.
  </text>
</svg>

---


# 🌟 Core Capabilities

✔ Portfolio-level experiment intelligence  
✔ Multi-objective ranking (balanced / leaderboard / stability / speed)  
✔ CV vs Holdout overfitting diagnostics  
✔ Model-family aggregate analytics  
✔ Training-time trade-off modeling  
✔ Rule-based CLI assistant  
✔ Gemini-powered tool-calling agent  
✔ Kaggle notebook auto-generation  
✔ Goal-aware model-family selection  

---

# 📦 Project Structure

```text
orchestrator_agent/
│
├── tools.py               # CSV loading & portfolio statistics
├── ranking.py             # Multi-strategy ranking engine
├── orchestrator.py        # High-level portfolio analysis
├── cli_agent.py           # Rule-based CLI agent
├── adk_tools.py           # Tool wrappers for LLM function calling
├── adk_agent_cli.py       # Gemini-powered interactive agent
├── template_agent.py      # Notebook generator + model selection
├── notebook_agent.py      # LLM Q&A wrapper
├── viz.py                 # Matplotlib visualizations
├── main.py                # Basic CLI entry point
├── generate_dummy_experiments.py
└── __init__.py
```

---

# 🧠 Ranking Strategies

Experiments are scored via normalized composite scoring:

```
rank_score = w_cv * normalized_cv
           - w_gap * normalized_gap
           - w_time * normalized_time
```

Where:

- CV → performance objective  
- Gap → generalization penalty  
- Time → iteration cost  

Supported strategies:

| Strategy     | Objective Focus |
|--------------|-----------------|
| balanced     | Strong overall trade-off |
| leaderboard  | Maximize CV score |
| stability    | Minimize overfitting |
| speed        | Optimize iteration velocity |

This mirrors real-world ML trade-off reasoning.

---

# 🚀 Quick Start

### Install dependencies

```
pip install pandas numpy matplotlib nbformat google-genai
```

### Generate sample portfolio

```
python generate_dummy_experiments.py
```

### Run portfolio analysis

```
python -m orchestrator_agent.main
```

### Run rule-based agent

```
python -m orchestrator_agent.cli_agent
```

### Run Gemini-powered agent

```
export GOOGLE_API_KEY=your_key
python -m orchestrator_agent.adk_agent_cli
```

---

# 🎯 Strategic Workflow

1. Log experiments to CSV  
2. Run portfolio intelligence  
3. Diagnose overfitting & model-family stats  
4. Rank experiments using strategy  
5. Query agent for trade-off analysis  
6. Generate next experiment plan  
7. Iterate  

This introduces system-level experimentation discipline.

---

# ⚠ Constraints

- Requires structured `experiments.csv`
- Assumes higher metric is better
- LLM features require Google GenAI API key
- Designed for tabular ML portfolios
- Does not auto-submit to Kaggle

---

# 🔮 Roadmap

- Multi-objective optimization engine  
- Optuna / hyperparameter sweeps  
- MLflow integration  
- Kaggle API submission automation  
- Drift monitoring  
- Local LLM backend support  
- Autonomous experiment planner  

---

# 👤 Author

**Chirag Desai**   
Focused on reproducible ML systems, experiment intelligence, and agent-driven workflows.

---

### ⭐ If you find this project useful, consider starring the repository.
