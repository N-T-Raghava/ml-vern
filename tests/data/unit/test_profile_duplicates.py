from mlvern.data.inspect import DataInspector


def test_profile_duplicates(df_with_duplicates):
    inspector = DataInspector(df_with_duplicates)
    dups = inspector._profile_duplicates()
    assert dups["total"] >= 1
