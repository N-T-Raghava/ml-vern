import pandas as pd


def test_hypothesis_test_two_samples(stats_module):
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    y = pd.Series([2.0, 3.0, 4.0, 5.0])
    out = stats_module.hypothesis_test_two_samples(x, y)
    assert "pvalue" in out or out.get("status") == "skipped"
