def test_constant_columns(stats_module):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
    v = stats_module.vif(df)
    assert isinstance(v, dict)
