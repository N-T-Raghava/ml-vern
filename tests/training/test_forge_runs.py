"""Tests for Forge training runs and accessors."""

import json
import os

import joblib
import pytest

from mlvern.core.forge import Forge


class TestForgeTrainingRun:
    """Tests for Forge.run() training workflow."""

    def test_forge_run_basic(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test basic training run workflow."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        # Register dataset
        fp, _ = forge.register_dataset(sample_df, "target")

        # Prepare data
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data
        config = {"model": "logistic", "learning_rate": 0.01}

        # Run training
        run_id, metrics = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, config, fp
        )

        assert run_id is not None
        assert run_id.startswith("run_")
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_forge_run_creates_run_directory(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run creates proper directory structure."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data
        config = {"model": "lr"}

        run_id, _ = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, config, fp
        )

        run_path = os.path.join(forge.mlvern_dir, "runs", run_id)
        assert os.path.exists(run_path)
        assert os.path.exists(os.path.join(run_path, "meta.json"))
        assert os.path.exists(os.path.join(run_path, "config.json"))
        assert os.path.exists(os.path.join(run_path, "metrics.json"))

    def test_forge_run_saves_model(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run saves model artifact."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)
        model_path = os.path.join(
            forge.mlvern_dir, "runs", run_id, "artifacts", "model.pkl"
        )
        assert os.path.exists(model_path)

        # Load and verify model
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    def test_forge_run_saves_metadata(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run saves metadata correctly."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        meta_path = os.path.join(forge.mlvern_dir, "runs", run_id, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["run_id"] == run_id
        assert meta["dataset_hash"] == fp["dataset_hash"]
        assert "timestamp" in meta

    def test_forge_run_saves_config(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run saves config correctly."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data
        config = {"model": "logistic", "epochs": 50, "batch_size": 32}

        run_id, _ = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, config, fp
        )

        config_path = os.path.join(forge.mlvern_dir, "runs", run_id, "config.json")
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config == config

    def test_forge_run_saves_metrics(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run saves metrics correctly."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, metrics = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, {}, fp
        )

        metrics_path = os.path.join(forge.mlvern_dir, "runs", run_id, "metrics.json")
        with open(metrics_path) as f:
            saved_metrics = json.load(f)

        assert "accuracy" in saved_metrics
        assert saved_metrics["accuracy"] == metrics["accuracy"]

    def test_forge_list_runs(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test listing all runs."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        runs = forge.list_runs()
        assert len(runs) == 0

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        runs = forge.list_runs()
        assert len(runs) == 1
        assert run_id in runs

    def test_forge_multiple_runs(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
        forest_model,
    ):
        """Test tracking multiple runs."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id1, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)
        run_id2, _ = forge.run(forest_model, X_train, y_train, X_val, y_val, {}, fp)

        runs = forge.list_runs()
        assert len(runs) == 2
        assert run_id1 in runs
        assert run_id2 in runs

    def test_forge_run_registry_update(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run updates registry correctly."""
        from mlvern.utils.registry import load_registry

        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, metrics = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, {}, fp
        )

        registry = load_registry(forge.mlvern_dir)
        assert run_id in registry["runs"]
        assert registry["runs"][run_id]["model"] == "LogisticRegression"
        assert registry["runs"][run_id]["metrics"] == metrics


class TestForgeRunAccessors:
    """Tests for Forge run and model accessor methods."""

    def test_get_run_success(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test retrieving run metadata and information."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        run_info = forge.get_run(run_id)

        assert run_info["run_id"] == run_id
        assert "metadata" in run_info
        assert "metrics" in run_info
        assert "config" in run_info
        assert "tags" in run_info
        assert "registry_info" in run_info

    def test_get_run_not_found(self, tmp_mlvern_dir):
        """Test get_run raises error for non-existent run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        with pytest.raises(ValueError, match="not found"):
            forge.get_run("nonexistent_run")

    def test_get_run_metrics(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test retrieving metrics for a specific run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, metrics = forge.run(
            logistic_model, X_train, y_train, X_val, y_val, {}, fp
        )

        retrieved_metrics = forge.get_run_metrics(run_id)

        assert retrieved_metrics == metrics
        assert "accuracy" in retrieved_metrics

    def test_get_run_artifacts(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test retrieving artifact paths for a run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        artifacts = forge.get_run_artifacts(run_id)

        assert "run_dir" in artifacts
        assert "meta.json" in artifacts
        assert "config.json" in artifacts
        assert "metrics.json" in artifacts
        assert "artifact_model.pkl" in artifacts
        assert all(os.path.exists(path) for path in artifacts.values())

    def test_load_model_success(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test loading a trained model from a run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        loaded_model = forge.load_model(run_id)

        assert loaded_model is not None
        assert hasattr(loaded_model, "predict")
        assert hasattr(loaded_model, "coef_")

    def test_load_model_not_found(self, tmp_mlvern_dir):
        """Test load_model raises error for non-existent run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        with pytest.raises(ValueError):
            forge.load_model("nonexistent_run")

    def test_load_model_makes_predictions(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that loaded model can make predictions."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        loaded_model = forge.load_model(run_id)
        predictions = loaded_model.predict(X_val)

        assert predictions is not None
        assert len(predictions) == len(X_val)
