import pandas as pd

from mlvern.data.inspect import DataInspector


def test_mixed_types_object_col():
    df = pd.DataFrame({"o": [1, "two", None]})
    inspector = DataInspector(df)
    res = inspector._validate_type_consistency()
    assert not res["is_valid"]
    assert "o" in res["issues"]
