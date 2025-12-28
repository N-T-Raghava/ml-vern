from pathlib import Path


def test_single_row_runs(run_eda, mlvern_dir, single_row):
    res = run_eda(single_row)
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    assert rpt.exists()
    assert res["n_rows"] == 1
