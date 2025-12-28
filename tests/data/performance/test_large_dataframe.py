import pandas as pd

from mlvern.data.inspect import inspect_data


def test_large_dataframe_performance(stats_module):
    n = 20000
    df = pd.DataFrame({"a": range(n), "b": range(n)})
    # run a moderately expensive operation
    v = stats_module.vif(df)
    assert isinstance(v, dict)


def test_large_dataframe(tmp_path):
    # medium sized test to sanity check performance
    n = 20000
    df = pd.DataFrame(
        {"a": range(n), "b": range(n), "target": [i % 2 for i in range(n)]}
    )
    report = inspect_data(df, "target", str(tmp_path))
    assert report["part_1_profiling"]["dataset_shape"]["rows"] == n
