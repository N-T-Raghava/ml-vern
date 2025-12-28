import os
from pathlib import Path

import pandas as pd

from mlvern.visual.eda import basic_eda


def test_missingness_map_created(run_eda, mlvern_dir):
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 2]})
    res = run_eda(df)
    miss = Path(mlvern_dir) / "plots" / "eda" / "missingness" / "missingness_map.png"
    assert str(miss) in res["plots"]
    assert miss.exists()


def test_missingness_plot(tmp_mlvern_dir, sample_df_missing):
    out = basic_eda(sample_df_missing, output_dir="unused", mlvern_dir=tmp_mlvern_dir)
    plots = out["plots"]
    assert any("missingness_map.png" in p for p in plots)
    mmap = [p for p in plots if p.endswith("missingness_map.png")][0]
    assert os.path.isfile(mmap)
