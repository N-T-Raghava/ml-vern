#!/bin/bash
# Local test script for mlvern

set -e

echo "Running mlvern test suite..."
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "pytest not found. Installing dev dependencies..."
    pip install -e ".[dev]"
fi

# Ensure dev tools are installed (flake8, mypy, pytest, coverage)
echo "Checking for dev dependencies: pytest, flake8, mypy, coverage"
missing=0
command -v pytest >/dev/null 2>&1 || missing=1
command -v flake8 >/dev/null 2>&1 || missing=1
python -c "import importlib; exit(0) if importlib.util.find_spec('mypy') else exit(1)" || missing=1
command -v coverage >/dev/null 2>&1 || missing=1
if [ "$missing" -ne 0 ]; then
    echo "Some dev dependencies are missing. Installing dev dependencies..."
    pip install -e ".[dev]"
fi

# Run linters
echo "Running linters: flake8 and mypy"
flake8 mlvern/ tests/ --max-line-length=100 || { echo "Flake8 reported issues"; exit 1; }
python -m mypy mlvern/ --ignore-missing-imports || { echo "mypy reported issues"; exit 1; }

# Run the structured test suite
echo "Running tests (tests/data)..."
pytest tests/data -v || { echo "Tests failed"; exit 1; }

# Run coverage and produce HTML report
echo ""
echo "Running tests with coverage..."
coverage run -m pytest tests/data
coverage report -m
coverage html

echo ""
echo "All tests passed and coverage report generated in htmlcov/index.html"
