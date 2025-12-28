from pathlib import Path


def test_no_numeric_columns_creates_missingness_and_report(
    run_eda, mlvern_dir, no_numeric_columns
):
    res = run_eda(no_numeric_columns)
    # correlation shouldn't be in plots
    assert not any("correlation" in p for p in res["plots"])
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    assert rpt.exists()
