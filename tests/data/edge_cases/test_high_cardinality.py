import pandas as pd

from mlvern.data.inspect import DataInspector


def test_high_cardinality(tmp_path):
    df = pd.DataFrame({"cat": [str(i) for i in range(1000)]})
    inspector = DataInspector(df)
    card = inspector._profile_cardinality()
    assert card["cat"]["unique_values"] == 1000
