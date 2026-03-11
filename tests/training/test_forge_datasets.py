"""Tests for Forge dataset management and accessors."""

import os

import pytest

from mlvern.core.forge import Forge


class TestForgeDatasetOperations:
    """Tests for Forge dataset management."""

    def test_forge_register_dataset(self, tmp_mlvern_dir, sample_df):
        """Test registering a dataset."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, is_new = forge.register_dataset(sample_df, "target")

        assert is_new
        assert "dataset_hash" in fp
        assert fp["rows"] == 50
        assert fp["columns"] == 4
        assert fp["schema"]["target"] == "target"

    def test_forge_register_dataset_duplicate(self, tmp_mlvern_dir, sample_df):
        """Test registering the same dataset twice returns cached."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp1, is_new1 = forge.register_dataset(sample_df, "target")
        fp2, is_new2 = forge.register_dataset(sample_df, "target")

        assert is_new1  # First registration
        assert not is_new2  # Second is from cache
        assert fp1["dataset_hash"] == fp2["dataset_hash"]

    def test_forge_list_datasets(self, tmp_mlvern_dir, sample_df):
        """Test listing registered datasets."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        datasets = forge.list_datasets()
        assert len(datasets) == 0

        fp, _ = forge.register_dataset(sample_df, "target")
        datasets = forge.list_datasets()

        assert len(datasets) == 1
        assert fp["dataset_hash"] in datasets

    def test_forge_list_datasets_multiple(self, tmp_mlvern_dir):
        """Test listing multiple datasets."""
        import pandas as pd

        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        df1 = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9], "target": [1, 0, 1]})

        fp1, _ = forge.register_dataset(df1, "target")
        fp2, _ = forge.register_dataset(df2, "target")

        datasets = forge.list_datasets()
        assert len(datasets) == 2
        assert fp1["dataset_hash"] in datasets
        assert fp2["dataset_hash"] in datasets


class TestForgeDatasetAccessors:
    """Tests for Forge dataset accessor methods."""

    def test_get_dataset_path_success(self, tmp_mlvern_dir, sample_df):
        """Test retrieving dataset path by hash."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        path = forge.get_dataset_path(dataset_hash)

        assert path is not None
        assert os.path.exists(path)
        assert dataset_hash in path

    def test_get_dataset_path_not_found(self, tmp_mlvern_dir):
        """Test get_dataset_path raises error for non-existent hash."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        with pytest.raises(ValueError, match="not found"):
            forge.get_dataset_path("nonexistent_hash")

    def test_load_dataset_metadata(self, tmp_mlvern_dir, sample_df):
        """Test loading dataset metadata and paths."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        dataset_info = forge.load_dataset(dataset_hash)

        assert dataset_info["dataset_hash"] == dataset_hash
        assert "path" in dataset_info
        assert "schema" in dataset_info
        assert "metadata" in dataset_info
        assert "report_paths" in dataset_info
        assert "plot_paths" in dataset_info
        assert dataset_info["schema"]["target"] == "target"

    def test_load_dataset_not_found(self, tmp_mlvern_dir):
        """Test load_dataset raises error for non-existent hash."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        with pytest.raises(ValueError):
            forge.load_dataset("nonexistent_hash")

    def test_load_dataset_has_reports(self, tmp_mlvern_dir, sample_df):
        """Test that loaded dataset includes report paths when available."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        dataset_info = forge.load_dataset(dataset_hash)

        assert "report_paths" in dataset_info
        assert isinstance(dataset_info["report_paths"], dict)
