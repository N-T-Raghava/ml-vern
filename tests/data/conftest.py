from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})


@pytest.fixture
def df_with_missing():
    return pd.DataFrame({"x": [1, None, 3, None], "y": [0, 1, None, 0]})


@pytest.fixture
def df_with_duplicates():
    return pd.DataFrame({"x": [1, 2, 1, 3], "y": [0, 1, 0, 1]})


@pytest.fixture
def numeric_df():
    return pd.DataFrame({"n": [1, 2, 1000, -999], "m": [1.5, 2.5, 3.5, 4.5]})


@pytest.fixture
def target_df():
    return pd.DataFrame(
        {"feat": list(range(10)), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}
    )


@pytest.fixture
def tmp_reports_dir(tmp_path: Path):
    return tmp_path / "reports"


@pytest.fixture
def risk_df():
    # DataFrame with a binary target and a sensitive column
    return pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6],
            "feat2": [10, 11, 12, 13, 14, 15],
            "target": [0, 1, 0, 1, 0, 1],
            "sex": ["M", "F", "M", "F", "M", "F"],
        }
    )


@pytest.fixture
def baseline_current_df():
    baseline = pd.DataFrame({"x": list(range(100)), "cat": ["a"] * 50 + ["b"] * 50})
    current = pd.DataFrame({"x": list(range(50, 150)), "cat": ["a"] * 30 + ["b"] * 70})
    return baseline, current


@pytest.fixture
def train_test_df():
    train = pd.DataFrame({"x": list(range(100)), "y": [0] * 50 + [1] * 50})
    test = pd.DataFrame({"x": list(range(50, 150)), "y": [0] * 30 + [1] * 70})
    return train, test


@pytest.fixture
def stats_df():
    # DataFrame with multiple numeric columns for statistics tests
    import numpy as _np

    rng = _np.random.RandomState(0)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, size=200),
            "b": rng.normal(5, 2, size=200),
            "c": rng.uniform(-1, 1, size=200),
            "target": [0, 1] * 100,
        }
    )


@pytest.fixture
def risk_module():
    # lazy import to avoid import costs during unrelated tests
    from mlvern.data import risk_check

    return risk_check


@pytest.fixture
def stats_module():
    from mlvern.data import statistics

    return statistics
