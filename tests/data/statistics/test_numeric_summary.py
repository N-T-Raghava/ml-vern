def test_numeric_summary(stats_module, stats_df):
    out = stats_module.numeric_summary(stats_df, cols=["a", "b"])
    assert "a" in out and "b" in out
    assert out["a"]["count"] > 0
