def test_statistics_pipeline_smoke(stats_module, stats_df):
    report = stats_module.compute_statistics(stats_df, target="target")
    assert "numeric_summary" in report
