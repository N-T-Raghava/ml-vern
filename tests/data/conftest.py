import pandas as pd
import pytest

from pathlib import Path


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
    return pd.DataFrame({"feat": list(range(10)), "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})


@pytest.fixture
def tmp_reports_dir(tmp_path: Path):
    return tmp_path / "reports"
