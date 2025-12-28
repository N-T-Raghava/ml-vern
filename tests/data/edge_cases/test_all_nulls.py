def test_all_nulls_detection(stats_module):
    import numpy as np
    import pandas as pd

    # use numeric NaNs so pandas treats columns as numeric
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    out = stats_module.numeric_summary(df, cols=["a", "b"])
    # numeric_summary returns "no_data" status for empty numeric series
    assert all(v.get("status") == "no_data" for v in out.values())
