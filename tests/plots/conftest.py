import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def iris_df():
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    return data.frame


@pytest.fixture
def numeric_df():
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})


@pytest.fixture
def df_with_target(iris_df):
    return iris_df.copy()


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def single_row():
    return pd.DataFrame({"x": [1], "y": [2]})


@pytest.fixture
def all_null_numeric():
    return pd.DataFrame({"x": [None, None], "y": [None, None]})


@pytest.fixture
def no_numeric_columns():
    return pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})


@pytest.fixture
def mlvern_dir(tmp_path: Path):
    # create a .mlvern directory inside the tmp workspace
    # (simulates examples/.mlvern)
    root = tmp_path
    mlvern = root / ".mlvern"
    mlvern.mkdir()
    # ensure reports and plots exist
    (mlvern / "reports").mkdir()
    (mlvern / "plots").mkdir()
    return str(mlvern)


@pytest.fixture
def run_eda(mlvern_dir):
    from mlvern.visual.eda import basic_eda

    def _run(df, target=None):
        return basic_eda(
            df,
            output_dir="ignored",
            mlvern_dir=mlvern_dir,
            target=target,
        )

    return _run


@pytest.fixture
def sample_df():
    """Sample dataframe with numeric columns for EDA tests."""
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "feat1": np.random.randn(100),
            "feat2": np.random.randn(100) * 2 + 1,
            "target": np.random.choice([0, 1], size=100),
        }
    )
    return df


@pytest.fixture
def sample_df_missing(sample_df):
    """Sample dataframe with missing values."""
    df = sample_df.copy()
    df.loc[0:4, "feat1"] = np.nan
    return df


@pytest.fixture
def tmp_mlvern_dir(tmp_path: Path):
    """Create a temporary .mlvern directory structure."""
    d = tmp_path / ".mlvern"
    plots = d / "plots"
    reports = d / "reports"
    plots.mkdir(parents=True)
    reports.mkdir(parents=True)
    return str(d)


@pytest.fixture
def tmp_examples_mlvern(tmp_path, monkeypatch):
    """Create examples/.mlvern and change to tmp_path."""
    examples = tmp_path / "examples"
    m = examples / ".mlvern"
    (m / "plots").mkdir(parents=True)
    (m / "reports").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    yield str(m)
    # cleanup
    try:
        shutil.rmtree(str(tmp_path))
    except Exception:
        pass
