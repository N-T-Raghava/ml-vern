import pandas as pd

from mlvern.data.inspect import DataInspector


def test_profile_cardinality():
    df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
    inspector = DataInspector(df)
    card = inspector._profile_cardinality()
    assert "cat" in card
    assert card["cat"]["unique_values"] == 3
