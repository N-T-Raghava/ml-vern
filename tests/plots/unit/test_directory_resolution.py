import os
from pathlib import Path

from mlvern.visual.eda import basic_eda


def test_directory_resolution_creates_expected_dirs(run_eda, mlvern_dir, iris_df):
    run_eda(iris_df)
    plots_dir = Path(mlvern_dir) / "plots" / "eda"
    assert plots_dir.exists()
    # check subfolders
    assert (plots_dir / "distributions").exists()
    assert (plots_dir / "box_violin").exists()
    assert (plots_dir / "correlation").exists()
    assert (plots_dir / "missingness").exists()


def test_directory_resolution(tmp_examples_mlvern, sample_df):
    # don't pass mlvern_dir -> basic_eda should detect examples/.mlvern
    # in cwd
    basic_eda(sample_df, output_dir="unused", mlvern_dir=None)
    # ensure plots were created under examples/.mlvern directory
    plots_dir = os.path.join(tmp_examples_mlvern, "plots", "eda")
    assert os.path.isdir(plots_dir)
    # report present
    report_path = os.path.join(tmp_examples_mlvern, "reports", "eda_report.json")
    assert os.path.isfile(report_path)
