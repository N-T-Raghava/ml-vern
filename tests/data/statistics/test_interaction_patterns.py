def test_interaction_patterns_basic(stats_module, stats_df):
    out = stats_module.interaction_patterns(stats_df, target="target", top_n=5)
    assert "interactions" in out
