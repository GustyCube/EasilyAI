.PHONY: help install install-dev test test-unit test-integration test-cov lint format type-check security-check clean docs serve-docs build publish-test publish dev-setup pre-commit

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package with all development dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

test: ## Run all tests
	pytest

test-unit: ## Run unit tests only
	pytest -m unit

test-integration: ## Run integration tests only
	pytest -m integration

test-cov: ## Run tests with coverage report
	pytest --cov=easilyai --cov-report=html --cov-report=term

lint: ## Run all linting checks
	@echo "Running flake8..."
	flake8 easilyai tests
	@echo "Running mypy..."
	mypy easilyai
	@echo "Running bandit security check..."
	bandit -r easilyai -ll

format: ## Format code with black and isort
	@echo "Running isort..."
	isort easilyai tests
	@echo "Running black..."
	black easilyai tests

format-check: ## Check code formatting without making changes
	@echo "Checking isort..."
	isort --check-only easilyai tests
	@echo "Checking black..."
	black --check easilyai tests

type-check: ## Run type checking with mypy
	mypy easilyai

security-check: ## Run security checks
	bandit -r easilyai -ll
	safety check --json

clean: ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

docs: ## Build documentation
	mkdocs build

serve-docs: ## Serve documentation locally
	mkdocs serve --dev-addr 127.0.0.1:8000

build: clean ## Build distribution packages
	python -m build

publish-test: build ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	python -m twine upload dist/*

dev-setup: ## Complete development environment setup
	@echo "Setting up development environment..."
	pip install --upgrade pip setuptools wheel
	pip install -e ".[dev,test,docs]"
	pre-commit install
	@echo "Development environment setup complete!"

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

update-deps: ## Update all dependencies to latest versions
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev,test,docs]"
	pre-commit autoupdate

check: format-check lint type-check test ## Run all checks (format, lint, type, test)

ci: ## Run CI pipeline locally
	@echo "Running CI pipeline..."
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) type-check
	$(MAKE) test-cov
	@echo "CI pipeline complete!"

watch-test: ## Run tests in watch mode (requires pytest-watch)
	pip install pytest-watch
	ptw -- --tb=short

benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ -v --benchmark-only

profile: ## Profile the code (requires py-spy)
	pip install py-spy
	py-spy record -o profile.svg -- python -m pytest tests/

version: ## Show current version
	@python -c "from easilyai import __version__; print(__version__)"

release-patch: ## Create a patch release
	bump2version patch
	git push && git push --tags

release-minor: ## Create a minor release
	bump2version minor
	git push && git push --tags

release-major: ## Create a major release
	bump2version major
	git push && git push --tags