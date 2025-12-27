def test_empty_dataframe_behaviour(stats_module):
    import pandas as pd

    df = pd.DataFrame()
    out = stats_module.compute_statistics(df)
    assert isinstance(out, dict)
import pytest
from mlvern.data.inspect import inspect_data
import pandas as pd


def test_empty_dataframe_raises(tmp_path):
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        inspect_data(df, "target", str(tmp_path))
