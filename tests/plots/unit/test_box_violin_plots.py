import os
from pathlib import Path

from mlvern.visual.eda import basic_eda


def test_box_and_violin_plots_created(run_eda, mlvern_dir, numeric_df):
    res = run_eda(numeric_df)
    plots = res["plots"]
    for col in ["a", "b"]:
        box = Path(mlvern_dir) / "plots" / "eda" / "box_violin" / f"{col}_box.png"
        violin = Path(mlvern_dir) / "plots" / "eda" / "box_violin" / f"{col}_violin.png"
        assert str(box) in plots
        assert box.exists()
        # violin may be missing in some envs but if reported it must exist
        if str(violin) in plots:
            assert violin.exists()


def test_box_violin_plots(tmp_mlvern_dir, sample_df):
    out = basic_eda(sample_df, output_dir="unused", mlvern_dir=tmp_mlvern_dir)
    plots = out["plots"]
    assert any("_box.png" in p for p in plots)
    assert any("_violin.png" in p for p in plots)
    for p in plots:
        assert os.path.isfile(p)
