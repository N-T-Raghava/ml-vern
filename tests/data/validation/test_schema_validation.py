import pandas as pd

from mlvern.data.inspect import DataInspector


def test_duplicate_column_names():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    inspector = DataInspector(df)
    res = inspector._validate_schema()
    assert not res["is_valid"]
