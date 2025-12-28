import os
from pathlib import Path

from mlvern.visual.eda import basic_eda


def test_correlation_heatmap_created(run_eda, mlvern_dir, numeric_df):
    res = run_eda(numeric_df)
    corr = (
        Path(mlvern_dir) / "plots" / "eda" / "correlation" / "correlation_heatmap.png"
    )
    assert str(corr) in res["plots"]
    assert corr.exists()


def test_correlation_plot(tmp_mlvern_dir, sample_df):
    out = basic_eda(sample_df, output_dir="unused", mlvern_dir=tmp_mlvern_dir)
    plots = out["plots"]
    assert any("correlation_heatmap.png" in p for p in plots)
    hmap = [p for p in plots if p.endswith("correlation_heatmap.png")][0]
    assert os.path.isfile(hmap)
