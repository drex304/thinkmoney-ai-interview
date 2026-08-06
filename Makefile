# thinkmoney AI Customer Service — developer commands
#
# Provider defaults to ollama because it needs no API key, so `make run`
# works on a clean checkout. Override per invocation:
#
#   make run PROVIDER=anthropic
#   make run PROVIDER=openai MODEL=gpt-4o

PROVIDER ?= ollama
MODEL    ?=

# Only pass --model through when one was actually supplied, so each provider
# keeps its own default from src/config.py.
MODEL_ARG := $(if $(MODEL),--model $(MODEL),)

.DEFAULT_GOAL := help
.PHONY: help install test test-quiet lint lint-fix format format-check check_all run check clean

help: ## Show this help
	@echo "thinkmoney AI Customer Service"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  PROVIDER=$(PROVIDER)$(if $(MODEL), MODEL=$(MODEL),)"

install: ## Install dependencies
	uv sync

test: ## Run the full test suite (verbose)
	uv run pytest -v

test-quiet: ## Run the full test suite (summary only)
	uv run pytest -q

lint: ## Lint with ruff (real defects only — layout is black's job)
	uv run ruff check .

lint-fix: ## Lint and apply the fixes ruff considers safe
	uv run ruff check . --fix

format: ## Format the code with black
	uv run black .

format-check: ## Fail if anything is unformatted (CI-shaped, changes nothing)
	uv run black --check .

# The full gate, in cheapest-first order: formatting and linting fail in
# under a second, so a broken build says why before the suite has started.
# Each step is a separate recipe line, so make stops at the first failure.
check_all: ## Format check, then lint, then the full test suite
	@echo "==> Formatting (black --check)"
	@$(MAKE) --no-print-directory format-check
	@echo
	@echo "==> Linting (ruff)"
	@$(MAKE) --no-print-directory lint
	@echo
	@echo "==> Tests (pytest)"
	@$(MAKE) --no-print-directory test-quiet
	@echo
	@echo "All checks passed."

run: ## Start the CLI (PROVIDER=ollama|openai|anthropic, optional MODEL=...)
	uv run thinkmoney --provider $(PROVIDER) $(MODEL_ARG)

check: test ## Run tests, then start the CLI
	@echo
	@echo "Tests green — starting CLI with provider '$(PROVIDER)'..."
	@echo
	@$(MAKE) --no-print-directory run PROVIDER=$(PROVIDER) MODEL=$(MODEL)

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
