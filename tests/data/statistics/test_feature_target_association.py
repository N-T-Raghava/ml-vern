def test_feature_target_association_basic(stats_module, stats_df):
    out = stats_module.feature_target_association(stats_df, "target")
    assert "a" in out and "b" in out
