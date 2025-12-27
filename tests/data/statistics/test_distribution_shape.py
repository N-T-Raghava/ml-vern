def test_distribution_shape_basic(stats_module, stats_df):
    out = stats_module.distribution_shape(stats_df, "a")
    assert "skewness" in out and "kurtosis" in out
