import os
import tempfile

import pandas as pd

from mlvern.data.inspect import inspect_data


def test_data_inspection_creates_report():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})

    with tempfile.TemporaryDirectory() as tmp:
        report = inspect_data(df, "y", tmp)

        assert "missing_values" in report
        assert os.path.exists(f"{tmp}/reports/data_inspection.json")
