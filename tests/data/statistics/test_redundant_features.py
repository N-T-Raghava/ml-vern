def test_redundant_features_detects_perfect(stats_module):
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [3, 2, 1]})
    pairs = stats_module.redundant_features(df, threshold=0.99)
    assert any(p[0] == "a" and p[1] == "b" for p in pairs)
