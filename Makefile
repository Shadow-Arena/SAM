# ============================================================================
# SAM3 Segment Studio — interactive image segmentation (text / box / point / mixed)
# Run `make help` to list all targets.
# ============================================================================

SHELL        := /bin/bash
VENV         := .venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
HOST         ?= 0.0.0.0
PORT         ?= 7860
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------- help
.PHONY: help
help: ## Show this help
	@echo "SAM3 Segment Studio — available targets:"
	@grep -E '^[a-zA-Z_][a-zA-Z0-9_.-]*:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- setup
.PHONY: venv
venv: ## Create the virtualenv
	python3 -m venv $(VENV)

.PHONY: install
install: ## Install package + dependencies into .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

.PHONY: setup
setup: venv install env ## Full setup (venv + deps + .env)

.PHONY: env
env: ## Create .env from .env.example (does not overwrite)
	@if [ -f .env ]; then echo ".env already exists, keeping it."; \
	else cp .env.example .env && echo "Created .env from .env.example"; fi

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "Cleaned."

.PHONY: purge
purge: clean ## Clean + remove .venv and outputs
	rm -rf $(VENV) outputs
	@echo "Removed $(VENV) and outputs/."

# ------------------------------------------------------------------ run
.PHONY: run
run: ## Launch the interactive Gradio app (make run PORT=7861)
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(PYTHON) -m app.main --host $(HOST) --port $(PORT)

.PHONY: run-mock
run-mock: ## Launch the app WITHOUT downloading the model (synthetic mock engine)
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	SAM_MOCK=true $(PYTHON) -m app.main --host $(HOST) --port $(PORT)

.PHONY: segment
segment: ## One-shot CLI segmentation: make segment ARGS="--image a.jpg --text car"
	@if [ ! -x "$(PYTHON)" ]; then echo "Run 'make setup' first."; exit 1; fi
	$(PYTHON) -m app.cli $(ARGS)

.PHONY: config
config: ## Print the effective configuration (values from .env / env vars)
	$(PYTHON) -m app.main --config

# -------------------------------------------------------------- quality
.PHONY: test
test: ## Run the test suite
	$(PYTHON) -m pytest

.PHONY: lint
lint: ## Run ruff checks
	$(PYTHON) -m ruff check app tests

.PHONY: fmt
fmt: ## Auto-format with ruff
	$(PYTHON) -m ruff check --fix app tests
	$(PYTHON) -m ruff format app tests

.PHONY: check
check: lint test ## Lint + tests

.PHONY: typecheck
typecheck: ## Run pyright/mypy if installed (optional)
	@$(PYTHON) -c "import mypy" 2>/dev/null && $(PYTHON) -m mypy app || \
		echo "mypy not installed — run: pip install mypy"

# ------------------------------------------------------------------ misc
.PHONY: preload
preload: ## Download & cache the SAM3 model to ~/.cache/huggingface
	$(PYTHON) -c "from transformers import Sam3Model, Sam3Processor; \
		Sam3Model.from_pretrained('facebook/sam3'); \
		Sam3Processor.from_pretrained('facebook/sam3'); \
		print('SAM3 cached.')"

	@echo ""
	@echo "Next steps:"
	@echo "  make run          # interactive UI on http://localhost:$(PORT)"
	@echo "  make segment ARGS=\"--image photo.jpg --text car\""
	@echo "  make test / lint # quality checks"
