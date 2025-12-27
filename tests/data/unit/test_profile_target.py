from mlvern.data.inspect import DataInspector


def test_profile_target(target_df):
    inspector = DataInspector(target_df, target="target")
    t = inspector._profile_target()
    assert t["target"] == "target"
    assert t["n_classes"] == 2
