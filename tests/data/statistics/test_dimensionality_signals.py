def test_dimensionality_signals_basic(stats_module, stats_df):
    out = stats_module.dimensionality_signals(stats_df, n_components=3)
    assert "explained_variance_ratio" in out or out.get("status")
