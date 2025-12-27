def test_single_row_stats(stats_module):
    import pandas as pd

    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    out = stats_module.numeric_summary(df)
    assert out["a"]["count"] == 1
from mlvern.data.inspect import inspect_data
import pandas as pd


def test_single_row(tmp_path):
    df = pd.DataFrame({"x": [1], "target": [0]})
    report = inspect_data(df, "target", str(tmp_path))
    assert report["part_1_profiling"]["dataset_shape"]["rows"] == 1
