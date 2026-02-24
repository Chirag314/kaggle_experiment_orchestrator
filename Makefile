.PHONY: help install dev lint format test run-summary run-cli-agent

help:
	@echo "make install       - install keo"
	@echo "make dev           - install dev + gemini extras"
	@echo "make lint          - ruff check"
	@echo "make format        - ruff format"
	@echo "make test          - pytest"
	@echo "make run-summary   - run portfolio analysis on sample csv"
	@echo "make run-cli-agent - run rule-based agent"

install:
	pip install -e .

dev:
	pip install -e ".[dev,gemini]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

run-summary:
	python -m keo.cli.portfolio

run-cli-agent:
	python -m keo.cli.agent --mode rule
