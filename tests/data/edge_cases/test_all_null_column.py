import pandas as pd

from mlvern.data.inspect import DataInspector


def test_all_null_column():
    df = pd.DataFrame({"a": [None, None, None], "target": [0, 1, 0]})
    inspector = DataInspector(df)
    res = inspector._validate_null_thresholds()
    assert not res["is_valid"]
    assert "a" in res["violations"]
