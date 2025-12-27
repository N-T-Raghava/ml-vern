from mlvern.data.inspect import inspect_data
import pandas as pd


def test_single_row(tmp_path):
    df = pd.DataFrame({"x": [1], "target": [0]})
    report = inspect_data(df, "target", str(tmp_path))
    assert report["part_1_profiling"]["dataset_shape"]["rows"] == 1
