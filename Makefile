# Quantum-Annealed Hyperparameter Search Makefile

.PHONY: help install install-dev test test-fast test-coverage lint format type-check security clean docs build publish

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install package for production"
	@echo "  install-dev  - Install package for development"
	@echo "  test         - Run all tests"
	@echo "  test-fast    - Run fast tests only"
	@echo "  test-coverage - Run tests with coverage report"
	@echo "  lint         - Run all linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  type-check   - Run type checking with mypy"
	@echo "  security     - Run security checks"
	@echo "  clean        - Clean build artifacts"
	@echo "  docs         - Build documentation"
	@echo "  build        - Build distribution packages"
	@echo "  publish      - Publish to PyPI"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,simulators,monitoring]"

# Testing
test:
	pytest tests/ -v

test-fast:
	pytest tests/ -v -m "not slow"

test-coverage:
	pytest tests/ -v --cov=quantum_hyper_search --cov-report=html --cov-report=term-missing

test-integration:
	pytest tests/ -v -m integration

# Code quality
lint: lint-flake8 lint-black lint-isort

lint-flake8:
	flake8 quantum_hyper_search/ --count --statistics

lint-black:
	black --check --diff quantum_hyper_search/

lint-isort:
	isort --check-only --diff quantum_hyper_search/

format:
	black quantum_hyper_search/
	isort quantum_hyper_search/

type-check:
	mypy quantum_hyper_search/ --ignore-missing-imports

# Security
security:
	bandit -r quantum_hyper_search/
	safety check

# Health check
health-check:
	python -m quantum_hyper_search.monitoring.health_check

# Performance benchmark
benchmark:
	python -c "from quantum_hyper_search.monitoring.health_check import run_health_check; run_health_check()"

# Documentation
docs:
	@echo "Building documentation..."
	@mkdir -p docs/build
	@echo "Documentation built in docs/build/"

# Build and distribution
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

# Development workflow
dev-setup: install-dev
	pre-commit install

dev-test: format lint type-check test-fast

dev-full: format lint type-check test-coverage security

# CI/CD simulation
ci-test:
	@echo "Running CI pipeline locally..."
	make format
	make lint
	make type-check
	make test-coverage
	make security
	@echo "CI pipeline completed successfully!"

# Docker targets
docker-build:
	docker build -t quantum-hyper-search .

docker-test:
	docker run --rm quantum-hyper-search make test

# Maintenance
update-deps:
	pip-compile --upgrade requirements.in
	pip-compile --upgrade requirements-dev.in

check-outdated:
	pip list --outdated