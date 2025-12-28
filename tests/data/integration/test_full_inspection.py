import pandas as pd

from mlvern.data.inspect import inspect_data


def test_full_inspection(tmp_path):
    df = pd.DataFrame({"a": [1, 2, None], "target": [0, 1, 0]})
    report = inspect_data(df, "target", str(tmp_path))
    assert "part_1_profiling" in report
    assert "part_2_validation" in report
