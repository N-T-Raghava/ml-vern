import pandas as pd

from mlvern.data.inspect import DataInspector


def test_temporal_parsing():
    df = pd.DataFrame({"created_at": ["2020-01-01", "not a date", "2020-03-01"]})
    inspector = DataInspector(df)
    res = inspector._validate_temporal()
    assert res["has_temporal_columns"]
    assert "created_at" in res["columns"]
