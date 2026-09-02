# ============================================================================
# SAM3 Segment Studio — interactive image segmentation (text / box / point / mixed)
# Dependency management: uv  (https://docs.astral.sh/uv/)
# Run `make help` to list all targets.
# ============================================================================

SHELL        := /bin/bash
UV           ?= uv
PYTHON       := $(UV) run python
HOST         ?= 0.0.0.0
PORT         ?= 7860
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------- help
.PHONY: help
help: ## Show this help
	@echo "SAM3 Segment Studio — available targets (uv):"
	@grep -E '^[a-zA-Z_][a-zA-Z0-9_.-]*:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------- prerequisites
.PHONY: check-uv
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "Error: 'uv' not found."; \
		echo "Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		echo "or:               pip install uv"; \
		exit 1; }

# ------------------------------------------------------------------ setup
.PHONY: setup
setup: check-uv ## Create .venv and install all deps from uv.lock (incl. dev group)
	$(UV) sync

.PHONY: env
env: ## Create .env from .env.example (does not overwrite)
	@if [ -f .env ]; then echo ".env already exists, keeping it."; \
	else cp .env.example .env && echo "Created .env from .env.example"; fi

.PHONY: lock
lock: check-uv ## (Re)generate uv.lock
	$(UV) lock

.PHONY: update
update: check-uv ## Upgrade dependencies and refresh uv.lock
	$(UV) sync --upgrade

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "Cleaned."

.PHONY: purge
purge: clean ## Clean + remove .venv and outputs
	rm -rf .venv outputs
	@echo "Removed .venv/ and outputs/. Re-run 'make setup'."

# ------------------------------------------------------------------ run
.PHONY: run
run: check-uv ## Launch the interactive Gradio app (make run PORT=7861)
	$(PYTHON) -m app.main --host $(HOST) --port $(PORT)

.PHONY: run-mock
run-mock: check-uv ## Launch the app WITHOUT downloading the model (synthetic mock engine)
	SAM_MOCK=true $(PYTHON) -m app.main --host $(HOST) --port $(PORT)

.PHONY: segment
segment: check-uv ## One-shot CLI segmentation: make segment ARGS="--image a.jpg --text car"
	$(PYTHON) -m app.cli $(ARGS)

.PHONY: config
config: check-uv ## Print the effective configuration (values from .env / env vars)
	$(PYTHON) -m app.main --config

# -------------------------------------------------------------- quality
.PHONY: test
test: check-uv ## Run the test suite
	$(UV) run pytest

.PHONY: lint
lint: check-uv ## Run ruff checks
	$(UV) run ruff check app tests

.PHONY: fmt
fmt: check-uv ## Auto-format with ruff
	$(UV) run ruff check --fix app tests
	$(UV) run ruff format app tests

.PHONY: check
check: lint test ## Lint + tests

# ------------------------------------------------------------------ misc
.PHONY: preload
preload: check-uv ## Download & cache the SAM3 model to ~/.cache/huggingface
	$(PYTHON) -c "from transformers import Sam3Model, Sam3Processor; \
		Sam3Model.from_pretrained('facebook/sam3'); \
		Sam3Processor.from_pretrained('facebook/sam3'); \
		print('SAM3 cached.')"

	@echo ""
	@echo "Next steps:"
	@echo "  make run          # interactive UI on http://localhost:$(PORT)"
	@echo "  make segment ARGS=\"--image photo.jpg --text car\""
	@echo "  make test / lint # quality checks"
