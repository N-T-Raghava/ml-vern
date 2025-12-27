from mlvern.data.inspect import DataInspector


def test_profile_missing(df_with_missing):
    inspector = DataInspector(df_with_missing)
    missing = inspector._profile_missing()
    assert missing["total_missing"] > 0
    assert "x" in missing["details"]
