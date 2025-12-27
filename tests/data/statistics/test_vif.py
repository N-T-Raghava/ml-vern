def test_vif_basic(stats_module, stats_df):
    v = stats_module.vif(stats_df[["a", "b", "c"]])
    assert isinstance(v, dict)
