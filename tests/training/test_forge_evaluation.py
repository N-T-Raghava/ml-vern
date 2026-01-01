"""Tests for Forge evaluation and dataset save/load methods."""

import json
import os

import pandas as pd
import pytest

from mlvern.core.forge import Forge


class TestForgeEvaluation:
    """Tests for Forge evaluation and prediction methods."""

    def test_predict_with_run_id(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test prediction using run_id."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        predictions = forge.predict(run_id, X_val)

        assert predictions is not None
        assert len(predictions) == len(X_val)

    def test_predict_with_model_object(
        self, sample_train_data, sample_val_data, logistic_model
    ):
        """Test prediction using model object directly."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        logistic_model.fit(X_train, y_train)
        forge = Forge("project")

        predictions = forge.predict(logistic_model, X_val)

        assert predictions is not None
        assert len(predictions) == len(X_val)

    def test_evaluate_with_run_id(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test evaluation using run_id."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        result = forge.evaluate(run_id, X_val, y_val)

        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert "precision" in result["metrics"]
        assert "recall" in result["metrics"]
        assert "f1" in result["metrics"]

    def test_evaluate_with_model_object(
        self, sample_train_data, sample_val_data, logistic_model, tmp_path
    ):
        """Test evaluation using model object directly."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        logistic_model.fit(X_train, y_train)
        forge = Forge("project")

        output_dir = os.path.join(str(tmp_path), "evaluation")
        result = forge.evaluate(logistic_model, X_val, y_val, output_dir=output_dir)

        assert "metrics" in result
        assert "accuracy" in result["metrics"]
        assert isinstance(result["metrics"]["accuracy"], float)

    def test_evaluate_saves_report(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that evaluation saves a report file."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        forge.evaluate(run_id, X_val, y_val)

        eval_report_path = os.path.join(
            forge.mlvern_dir, "runs", run_id, "evaluation", "evaluation_report.json"
        )
        assert os.path.exists(eval_report_path)

        with open(eval_report_path) as f:
            report = json.load(f)

        assert "metrics" in report


class TestForgeDatasetSaveLoad:
    """Tests for Forge dataset save/load methods."""

    def test_save_dataset(self, tmp_mlvern_dir, sample_df):
        """Test saving a dataset."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        result = forge.save_dataset(sample_df, dataset_hash, name="test_dataset")

        assert result["dataset_hash"] == dataset_hash
        assert result["saved"] is True
        assert result["metadata"]["name"] == "test_dataset"

    def test_save_dataset_with_tags(self, tmp_mlvern_dir, sample_df):
        """Test saving dataset with tags."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        tags = {"experiment": "exp1", "version": "v1"}
        result = forge.save_dataset(sample_df, dataset_hash, tags=tags)

        assert result["metadata"]["tags"] == tags

    def test_load_dataset_by_hash(self, tmp_mlvern_dir, sample_df):
        """Test loading dataset by hash."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        forge.save_dataset(sample_df, dataset_hash)

        loaded_df = forge.load_dataset_by_hash(dataset_hash)

        assert loaded_df.shape == sample_df.shape
        assert list(loaded_df.columns) == list(sample_df.columns)

    def test_load_dataset_by_hash_not_found(self, tmp_mlvern_dir):
        """Test loading non-existent dataset raises error."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        with pytest.raises(FileNotFoundError):
            forge.load_dataset_by_hash("nonexistent_hash")

    def test_get_dataset_report(self, tmp_mlvern_dir, sample_df):
        """Test getting aggregated dataset report."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        report = forge.get_dataset_report(dataset_hash)

        assert "dataset_hash" in report
        assert report["dataset_hash"] == dataset_hash
        assert "metadata" in report
        assert "inspection" in report
        assert "statistics" in report
        assert "risk" in report

    def test_load_dataset_preserves_dtypes(self, tmp_mlvern_dir):
        """Test that loading preserves data types."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        fp, _ = forge.register_dataset(df, "target")
        dataset_hash = fp["dataset_hash"]

        forge.save_dataset(df, dataset_hash)
        loaded_df = forge.load_dataset_by_hash(dataset_hash)

        assert loaded_df["int_col"].dtype == df["int_col"].dtype
        assert loaded_df["float_col"].dtype == df["float_col"].dtype
        assert loaded_df["str_col"].dtype == df["str_col"].dtype
