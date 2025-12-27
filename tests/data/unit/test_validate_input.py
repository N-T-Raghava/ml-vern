import pandas as pd
import pytest

from mlvern.data.inspect import DataInspector


def test_empty_dataframe_raises():
    df = pd.DataFrame()
    inspector = DataInspector(df)
    with pytest.raises(ValueError):
        inspector.inspect()
