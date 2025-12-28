from pathlib import Path


def test_basic_eda_no_target(run_eda, mlvern_dir, iris_df):
    res = run_eda(iris_df, target=None)
    # target_vs_features should not be present
    assert not any("target_vs_features" in p for p in res["plots"])
    assert (Path(mlvern_dir) / "reports" / "eda_report.json").exists()
