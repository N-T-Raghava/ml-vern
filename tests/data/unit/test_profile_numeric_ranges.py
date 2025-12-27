from mlvern.data.inspect import DataInspector


def test_profile_numeric_ranges(numeric_df):
    inspector = DataInspector(numeric_df)
    ranges = inspector._profile_numeric_ranges()
    assert "n" in ranges
    assert ranges["n"]["min"] <= ranges["n"]["max"]
