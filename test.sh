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

# Run tests
echo "Running unit tests..."
pytest tests/ -v

# Run with coverage
echo ""
echo "Running tests with coverage..."
pytest tests/ --cov=mlvern --cov-report=term-missing --cov-report=html

echo ""
echo "All tests passed!"
echo "Coverage report generated in htmlcov/index.html"
