from pathlib import Path


def test_basic_eda_end_to_end(run_eda, mlvern_dir, iris_df):
    res = run_eda(iris_df, target="target")
    # smoke test: at least one distribution, correlation and report exist
    assert any("distributions" in p for p in res["plots"])
    assert any("correlation_heatmap.png" in p for p in res["plots"])
    assert (Path(mlvern_dir) / "reports" / "eda_report.json").exists()
