import os
from pathlib import Path

from mlvern.visual.eda import basic_eda


def test_numeric_distribution_plots_created(run_eda, mlvern_dir, numeric_df):
    res = run_eda(numeric_df)
    plots = res["plots"]
    # expect histogram for each numeric column
    for col in ["a", "b"]:
        expected = (
            Path(mlvern_dir) / "plots" / "eda" / "distributions" / f"{col}_hist.png"
        )
        assert str(expected) in plots
        assert expected.exists()


def test_numeric_distributions(tmp_mlvern_dir, sample_df):
    out = basic_eda(sample_df, output_dir="unused", mlvern_dir=tmp_mlvern_dir)
    plots = out["plots"]
    # Expect hist for feat1 and feat2
    assert any("feat1_hist.png" in p for p in plots)
    assert any("feat2_hist.png" in p for p in plots)
    # Files should exist
    for p in plots:
        assert os.path.isfile(p)
