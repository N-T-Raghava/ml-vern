"""Test explicit skip behavior for insufficient rows."""

import pandas as pd

from mlvern.data.inspect import DataInspector


def test_numeric_ranges_skipped_single_row():
    """Single row (< 2) -> numeric_ranges should report skipped."""
    df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    inspector = DataInspector(df)
    result = inspector._profile_numeric_ranges()
    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_rows"
    assert result["required_min_rows"] == 2
    assert result["actual_rows"] == 1


def test_numeric_ranges_available_two_rows():
    """Two rows (>= 2) -> numeric_ranges should compute."""
    df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    inspector = DataInspector(df)
    result = inspector._profile_numeric_ranges()
    assert "status" not in result
    assert "x" in result
    assert "std" in result["x"]


def test_outliers_skipped_four_rows():
    """Four rows (< 5) -> outliers should report skipped."""
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    inspector = DataInspector(df)
    result = inspector._profile_outliers()
    assert result["status"] == "skipped"
    assert result["reason"] == "insufficient_rows"
    assert result["required_min_rows"] == 5
    assert result["actual_rows"] == 4


def test_outliers_available_five_rows():
    """Five rows (>= 5) -> outliers should compute."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
    inspector = DataInspector(df)
    result = inspector._profile_outliers()
    assert "status" not in result  # Should be data dict or empty dict, not skip status


def test_full_inspection_explicit_skip_status():
    """Full inspection on single-row df should include skip status in report."""
    df = pd.DataFrame({"x": [1.0], "target": [0]})
    inspector = DataInspector(df, target="target")
    report = inspector.inspect()

    # numeric_ranges should be skipped
    assert report["part_1_profiling"]["numeric_ranges"]["status"] == "skipped"
    # outliers should be skipped
    assert report["part_1_profiling"]["outliers"]["status"] == "skipped"
    # leakage check should be skipped
    assert report["part_2_validation"]["leakage_checks"]["status"] == "skipped"
