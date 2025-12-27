@echo off
REM Local test script for mlvern (Windows)

echo.
echo Running mlvern test suite...

REM Ensure dev dependencies (pytest, flake8, mypy, coverage) are available
python -m pip show pytest > nul 2>&1 || set _MISSING_PKG=1
python -m pip show flake8 > nul 2>&1 || set _MISSING_PKG=1
python -m pip show mypy > nul 2>&1 || set _MISSING_PKG=1
python -m pip show coverage > nul 2>&1 || set _MISSING_PKG=1
if defined _MISSING_PKG (
    echo Some dev dependencies are missing. Installing dev dependencies...
    pip install -e ".[dev]"
    set _MISSING_PKG=
)

echo Running linters: flake8 and mypy
flake8 mlvern/ tests/ --max-line-length=100 || (
    echo Flake8 reported issues. Fix them before continuing.
    exit /b 1
)
python -m mypy mlvern/ --ignore-missing-imports || (
    echo mypy reported issues. Fix them before continuing.
    exit /b 1
)

REM Run only the new structured test suite
echo Running tests (tests/data)...
pytest tests/data -v || (
    echo Tests failed.
    exit /b 1
)

REM Run coverage and produce HTML report
echo Running tests with coverage...
coverage run -m pytest tests/data
coverage report -m
coverage html

echo All tests passed and coverage report generated in htmlcov\index.html
