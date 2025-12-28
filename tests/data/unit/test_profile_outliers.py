import pandas as pd

from mlvern.data.inspect import DataInspector


def test_profile_outliers():
    df = pd.DataFrame({"v": [1, 1, 2, 1000, 1, 2]})
    inspector = DataInspector(df)
    out = inspector._profile_outliers()
    assert "v" in out
    assert out["v"]["count"] >= 1
