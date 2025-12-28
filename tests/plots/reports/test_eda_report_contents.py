import json
from pathlib import Path


def test_eda_report_contents(run_eda, mlvern_dir, iris_df):
    run_eda(iris_df, target="target")
    rpt = Path(mlvern_dir) / "reports" / "eda_report.json"
    data = json.loads(rpt.read_text(encoding="utf-8"))
    assert "n_rows" in data and data["n_rows"] == 150
    assert "numeric_columns" in data
    assert "plots" in data and isinstance(data["plots"], list)
