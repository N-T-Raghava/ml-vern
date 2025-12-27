def test_compute_statistics_smoke(stats_module, stats_df):
    report = stats_module.compute_statistics(stats_df, target="target")
    assert "numeric_summary" in report
    assert "vif" in report
