import numpy as np
import pandas as pd

from mlvern.data.inspect import DataInspector


def test_infinite_values():
    df = pd.DataFrame({"x": [1, np.inf, 3]})
    inspector = DataInspector(df)
    res = inspector._validate_ranges()
    assert not res["is_valid"]
    assert "x" in res["violations"]
