import pandas as pd
import pytest

from mlvern.data.inspect import inspect_data


def test_empty_dataframe_behaviour(stats_module):
    df = pd.DataFrame()
    out = stats_module.compute_statistics(df)
    assert isinstance(out, dict)


def test_empty_dataframe_raises(tmp_path):
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        inspect_data(df, "target", str(tmp_path))
