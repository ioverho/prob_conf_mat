.DEFAULT_GOAL := help

##@ Utility
.PHONY: help
help:  ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make <target>\033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: hello-world
hello-world: ## Tests uv and make
	@uv run python -c "import prob_conf_mat; print('Hello World!')"

.PHONY: clean
clean:  ## Clean up caches and build artifacts
	@rm -rf .venv/
	@rm -rf .__pycache__/
	@rm -rf .cache/
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -rf build/
	@rm -rf site/
	@rm -rf .coverage
# I don't trust this enough to actually run it
# 	@find . -type f -name '*.py[co]' -delete -or -type d -name __pycache__ -delete

##@ Environment
.PHONY: install
install: ## Install default dependencies
	@uv sync --no-dev --frozen

.PHONY: install-dev
install-dev: ## Install dev dependencies
	@uv sync --dev --frozen

.PHONY: upgrade
upgrade: ## Upgrade installed dependencies
	@uv lock --refresh --upgrade
	@uv cache prune

.PHONY: export
export: ## Export uv to requirements.txt file
	@uv export --no-dev --output-file ./requirements.txt --format requirements.txt

##@ Testing, Linting, Typing & Formatting
.PHONY: test
test: ## Runs all tests
	@uv run --dev pytest

.PHONY: coverage
coverage: ## Checks test coverage
	@uv run --dev coverage run -m pytest
	@uv run --dev coverage html

.PHONY: lint
lint: ## Run linting
	@uv run --dev ruff check ./src/prob_conf_mat ./tests

.PHONY: type
type: ## Run static typechecking
	@uv run --dev pyright > ./tests/logs/pyright/report

.PHONY: commit
commit: ## Run pre-commit checks
	@uv run --dev pre-commit run

##@ Documentation
.PHONY: docs-build
docs-build: ## Update the docs
	@uv run --dev python mkdocs.py
	@uv run --dev mkdocs build

.PHONY: docs-serve
docs-serve: ## Serve documentation site
	@uv run mkdocs serve --watch "./documentation" --watch "./src/prob_conf_mat"

#@ Profiling
#.PHONY: importtime
#importtime: ## Profile import time
#	@uv run --no-dev python -X importtime -c "from prob_conf_mat import Study" 2> ./tests/logs/import.log
#	@uv run --dev tuna ./tests/logs/import.log
