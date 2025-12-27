from mlvern.data.inspect import DataInspector


def test_import_and_init(sample_df):
    inspector = DataInspector(sample_df, target="target")
    assert inspector.df is not None
    assert inspector.target == "target"
