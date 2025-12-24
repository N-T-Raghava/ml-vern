@echo off
REM Local test script for mlvern (Windows)

echo.
echo Running mlvern test suite...
echo.

REM Check if pytest is installed
python -m pip show pytest > nul 2>&1
if errorlevel 1 (
    echo pytest not found. Installing dev dependencies...
    pip install -e ".[dev]"
)

REM Run tests
echo Running unit tests...
pytest tests/ -v

REM Run with coverage
echo.
echo Running tests with coverage...
pytest tests/ --cov=mlvern --cov-report=term-missing --cov-report=html

echo.
echo All tests passed!
echo Coverage report generated in htmlcov\index.html
