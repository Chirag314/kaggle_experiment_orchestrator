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

## Architecture Diagram (SVG)

<svg viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg">
  <style>
    .box { fill:#f4f6f8; stroke:#333; stroke-width:1.5; rx:10; }
    .text { font-family:monospace; font-size:14px; fill:#111; }
    .arrow { stroke:#444; stroke-width:2; marker-end:url(#arrowhead); }
  </style>

  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7"
            refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#444"/>
    </marker>
  </defs>

  <!-- CSV -->
  <rect class="box" x="300" y="20" width="300" height="50"/>
  <text class="text" x="375" y="50">experiments.csv</text>

  <!-- Tools -->
  <rect class="box" x="250" y="100" width="400" height="60"/>
  <text class="text" x="300" y="135">tools.py → summarize_experiments()</text>

  <!-- Orchestrator -->
  <rect class="box" x="250" y="190" width="400" height="60"/>
  <text class="text" x="280" y="225">orchestrator.py → portfolio intelligence</text>

  <!-- Ranking -->
  <rect class="box" x="250" y="280" width="400" height="60"/>
  <text class="text" x="295" y="315">ranking.py → multi-strategy scoring</text>

  <!-- Agents -->
  <rect class="box" x="100" y="380" width="250" height="70"/>
  <text class="text" x="140" y="415">cli_agent.py</text>
  <text class="text" x="130" y="435">Rule-based agent</text>

  <rect class="box" x="375" y="380" width="250" height="70"/>
  <text class="text" x="410" y="415">adk_agent_cli.py</text>
  <text class="text" x="395" y="435">Gemini tool-calling agent</text>

  <rect class="box" x="650" y="380" width="220" height="70"/>
  <text class="text" x="675" y="415">template_agent.py</text>
  <text class="text" x="675" y="435">Notebook + model selection</text>

  <!-- Arrows -->
  <line class="arrow" x1="450" y1="70" x2="450" y2="100"/>
  <line class="arrow" x1="450" y1="160" x2="450" y2="190"/>
  <line class="arrow" x1="450" y1="250" x2="450" y2="280"/>
  <line class="arrow" x1="450" y1="340" x2="225" y2="380"/>
  <line class="arrow" x1="450" y1="340" x2="500" y2="380"/>
  <line class="arrow" x1="450" y1="340" x2="760" y2="380"/>
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
