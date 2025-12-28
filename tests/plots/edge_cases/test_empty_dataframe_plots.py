from pathlib import Path


def test_empty_dataframe_does_not_crash(run_eda, mlvern_dir, empty_df):
    res = run_eda(empty_df)
    # should still write report and return a summary
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    assert rpt.exists()
    assert "n_rows" in res
