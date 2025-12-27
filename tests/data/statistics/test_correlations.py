def test_correlations_returns_matrix(stats_module, stats_df):
    mat = stats_module.correlations(stats_df, method="pearson")
    assert "a" in mat.columns and "b" in mat.index
