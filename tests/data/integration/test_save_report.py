import pandas as pd

from mlvern.data.inspect import DataInspector


def test_custom_save_report(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
    inspector = DataInspector(df, target="target", mlvern_dir=str(tmp_path))
    inspector.inspect()
    path = inspector.save_report(filename="custom.json")
    assert path.exists()
    assert path.name == "custom.json"
