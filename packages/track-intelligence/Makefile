# Development tasks. `make help` lists them.
#
# uv rather than pip + venv activation: it resolves and installs the whole
# stack in seconds, and `uv run` means no target has to care whether a
# virtualenv is active.

.DEFAULT_GOAL := help
.PHONY: help install dev test test-fast lint format typecheck check serve analyze fixtures evaluate docker docker-run clean

UV ?= uv
FILE ?=
DATASET ?= fixtures/dataset.csv

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and its runtime dependencies
	$(UV) sync

dev:  ## Install with dev tooling (pytest, ruff, mypy)
	$(UV) sync --extra dev

test:  ## Run the whole test suite
	$(UV) run pytest

test-fast:  ## Unit tests only -- skips anything that decodes audio
	$(UV) run pytest -m "not integration"

lint:  ## Check style and common mistakes
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:  ## Apply formatting and safe fixes
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## Static type check
	$(UV) run mypy

check: lint typecheck test  ## Everything CI would run

serve:  ## Run the HTTP API on :8000
	$(UV) run dj-analyze serve

analyze:  ## Analyse a file: make analyze FILE=track.mp3
	@test -n "$(FILE)" || { echo "usage: make analyze FILE=track.mp3"; exit 2; }
	$(UV) run dj-analyze "$(FILE)"

fixtures:  ## Render the generated audio fixtures to fixtures/generated/
	$(UV) run python scripts/make_fixtures.py

evaluate:  ## Score against a labelled dataset: make evaluate DATASET=my.csv
	$(UV) run python scripts/evaluate_dataset.py "$(DATASET)"

docker:  ## Build the container image (this is where Essentia runs)
	docker build -t dj-track-intelligence .

docker-run:  ## Run the container on :8000
	docker run --rm -p 8000:8000 dj-track-intelligence

clean:  ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache dist build htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
