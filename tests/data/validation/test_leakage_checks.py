import pandas as pd

from mlvern.data.inspect import DataInspector


def test_perfect_correlation_leakage():
    # feature equals target -> perfect correlation
    df = pd.DataFrame({"f": [0, 1, 0, 1], "target": [0, 1, 0, 1]})
    inspector = DataInspector(df, target="target")
    res = inspector._validate_leakage()
    assert res["has_leakage_risk"]
    assert any(d["feature"] == "f" for d in res["indicators"])
