from pathlib import Path


def test_eda_report_written(run_eda, mlvern_dir, iris_df):
    run_eda(iris_df, target="target")
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    assert rpt.exists()
