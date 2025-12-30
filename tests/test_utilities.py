"""
Tests for environment capture and dataset utilities.
"""

import os
import tempfile

import pandas as pd
import pytest

# ============================================================================
# TESTS: Environment Utilities
# ============================================================================


class TestEnvironmentUtilities:
    """Tests for environment capture utilities."""

    def test_get_python_version(self):
        """Test getting Python version information."""
        from mlvern.utils.environment import get_python_version

        version_info = get_python_version()

        assert "version" in version_info
        assert "version_info" in version_info
        assert "implementation" in version_info
        assert "executable" in version_info

        assert version_info["version_info"]["major"] >= 3

    def test_get_installed_packages(self):
        """Test getting installed packages."""
        from mlvern.utils.environment import get_installed_packages

        packages = get_installed_packages()

        assert isinstance(packages, dict)

    def test_get_pip_freeze(self):
        """Test getting pip freeze output."""
        from mlvern.utils.environment import get_pip_freeze

        freeze = get_pip_freeze()

        assert isinstance(freeze, list)

    def test_save_environment(self):
        """Test saving environment information."""
        from mlvern.utils.environment import save_environment

        with tempfile.TemporaryDirectory() as tmp_path:
            run_path = os.path.join(tmp_path, "test_run")

            env_info = save_environment(run_path)

            assert env_info is not None
            assert "python" in env_info
            assert "packages" in env_info
            assert "pip_freeze" in env_info
            assert "captured_at" in env_info

            env_path = os.path.join(run_path, "environment.json")
            assert os.path.exists(env_path)

    def test_load_environment(self):
        """Test loading saved environment information."""
        from mlvern.utils.environment import load_environment, save_environment

        with tempfile.TemporaryDirectory() as tmp_path:
            run_path = os.path.join(tmp_path, "test_run")

            save_environment(run_path)
            loaded = load_environment(run_path)

            assert loaded is not None
            assert "python" in loaded
            assert "packages" in loaded

    def test_load_environment_not_found(self):
        """Test loading environment from non-existent run."""
        from mlvern.utils.environment import load_environment

        result = load_environment("/nonexistent/path")

        assert result is None


# ============================================================================
# TESTS: Dataset Utilities
# ============================================================================


@pytest.fixture
def sample_df():
    """Create a sample dataframe."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "target": [0, 1, 0],
        }
    )


class TestDatasetUtilities:
    """Tests for dataset save/load utilities."""

    def test_save_dataset_to_path(self, sample_df):
        """Test saving dataset to path."""
        from mlvern.utils.dataset_utils import save_dataset_to_path

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            save_dataset_to_path(sample_df, dataset_path)

            data_dir = os.path.join(dataset_path, "data")
            assert os.path.exists(data_dir)

            pickle_path = os.path.join(data_dir, "dataset.pkl")
            assert os.path.exists(pickle_path)

    def test_load_dataset_from_path(self, sample_df):
        """Test loading dataset from path."""
        from mlvern.utils.dataset_utils import (
            load_dataset_from_path,
            save_dataset_to_path,
        )

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            save_dataset_to_path(sample_df, dataset_path)
            loaded_df = load_dataset_from_path(dataset_path)

            assert loaded_df.shape == sample_df.shape
            assert list(loaded_df.columns) == list(sample_df.columns)

    def test_load_dataset_from_path_not_found(self):
        """Test loading from non-existent path raises error."""
        from mlvern.utils.dataset_utils import load_dataset_from_path

        with pytest.raises(FileNotFoundError):
            load_dataset_from_path("/nonexistent/path")

    def test_save_load_dataset_metadata(self):
        """Test saving and loading dataset metadata."""
        from mlvern.utils.dataset_utils import (
            load_dataset_metadata,
            save_dataset_metadata,
        )

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            metadata = {
                "name": "test_data",
                "version": "1.0",
                "tags": {"exp": "exp1"},
            }

            save_dataset_metadata(dataset_path, metadata)
            loaded = load_dataset_metadata(dataset_path)

            assert loaded["name"] == "test_data"
            assert loaded["version"] == "1.0"
            assert loaded["tags"]["exp"] == "exp1"

    def test_load_dataset_metadata_not_found(self):
        """Test loading metadata from non-existent file."""
        from mlvern.utils.dataset_utils import load_dataset_metadata

        result = load_dataset_metadata("/nonexistent/path")

        assert result == {}

    def test_get_dataset_report(self, sample_df):
        """Test getting aggregated dataset report."""
        from mlvern.utils.dataset_utils import get_dataset_report, save_dataset_to_path

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            save_dataset_to_path(sample_df, dataset_path)

            report = get_dataset_report(dataset_path)

            assert "dataset_path" in report
            assert "inspection" in report
            assert "statistics" in report
            assert "risk" in report
            assert "eda" in report

    def test_dataset_roundtrip_preserves_data(self):
        """Test that save/load roundtrip preserves data."""
        from mlvern.utils.dataset_utils import (
            load_dataset_from_path,
            save_dataset_to_path,
        )

        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [1.1, 2.2, 3.3, 4.4],
                "c": ["x", "y", "z", "w"],
                "d": [True, False, True, False],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            save_dataset_to_path(df, dataset_path)
            loaded_df = load_dataset_from_path(dataset_path)

            pd.testing.assert_frame_equal(df, loaded_df)

    def test_dataset_roundtrip_preserves_dtypes(self):
        """Test that save/load preserves data types."""
        from mlvern.utils.dataset_utils import (
            load_dataset_from_path,
            save_dataset_to_path,
        )

        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_path:
            dataset_path = os.path.join(tmp_path, "test_dataset")

            save_dataset_to_path(df, dataset_path)
            loaded_df = load_dataset_from_path(dataset_path)

            assert loaded_df["int_col"].dtype == df["int_col"].dtype
            assert loaded_df["float_col"].dtype == df["float_col"].dtype
            assert loaded_df["str_col"].dtype == df["str_col"].dtype
