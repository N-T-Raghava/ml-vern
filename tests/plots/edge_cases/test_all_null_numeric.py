from pathlib import Path


def test_all_null_numeric_generates_report(run_eda, mlvern_dir, all_null_numeric):
    res = run_eda(all_null_numeric)
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    assert rpt.exists()
    # plots may be empty but function must return summary
    assert "n_rows" in res
