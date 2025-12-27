from mlvern.data.inspect import DataInspector


def test_profile_shape(sample_df):
    inspector = DataInspector(sample_df)
    shape = inspector._profile_shape()
    assert shape["rows"] == 3
    assert shape["columns"] == 3
    assert shape["memory_mb"] >= 0
