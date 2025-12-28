import json

import pandas as pd

from mlvern.data.inspect import inspect_data


def test_report_saved_and_json(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    inspect_data(df, "target", str(tmp_path))
    p = tmp_path / "reports" / "data_inspection_report.json"
    assert p.exists()
    data = json.loads(p.read_text())
    assert "metadata" in data
    assert "part_1_profiling" in data
