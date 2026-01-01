"""Tests for Forge model registry and management."""

import json
import os

from mlvern.core.forge import Forge


class TestForgeModelRegistry:
    """Tests for Forge model registry methods."""

    def test_register_model_success(self, tmp_mlvern_dir, logistic_model):
        """Test registering a model."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        metadata = {
            "model_name": "logistic_regression_v1",
            "description": "Test model",
            "hyperparameters": {"C": 1.0},
        }

        model_id = forge.register_model(logistic_model, metadata)

        assert model_id is not None
        assert "model_" in model_id

        # Verify model was saved
        model_path = os.path.join(forge.mlvern_dir, "models", f"{model_id}.pkl")
        assert os.path.exists(model_path)

    def test_register_model_custom_id(self, tmp_mlvern_dir, logistic_model):
        """Test registering a model with custom ID."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        metadata = {"model_name": "test_model"}
        custom_id = "custom_model_v1"

        model_id = forge.register_model(logistic_model, metadata, model_id=custom_id)

        assert model_id == custom_id

    def test_list_models(self, tmp_mlvern_dir, logistic_model, forest_model):
        """Test listing registered models."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        models = forge.list_models()
        assert len(models) == 0

        id1 = forge.register_model(
            logistic_model, {"name": "lr"}, model_id="model_lr_v1"
        )
        id2 = forge.register_model(forest_model, {"name": "rf"}, model_id="model_rf_v1")

        models = forge.list_models()
        assert len(models) == 2
        assert id1 in models
        assert id2 in models

    def test_register_model_metadata_saved(self, tmp_mlvern_dir, logistic_model):
        """Test that model metadata is saved alongside the model."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        metadata = {
            "model_name": "test_model",
            "version": "1.0",
            "source_run_id": "run_123",
        }

        model_id = forge.register_model(logistic_model, metadata)

        model_path = os.path.join(forge.mlvern_dir, "models", f"{model_id}.pkl")
        metadata_path = model_path.replace(".pkl", "_metadata.json")

        assert os.path.exists(metadata_path)

        with open(metadata_path) as f:
            saved_metadata = json.load(f)

        assert saved_metadata["model_name"] == "test_model"
        assert saved_metadata["version"] == "1.0"

    def test_tag_run_success(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test tagging a run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        tags = {"experiment": "baseline", "status": "approved"}
        forge.tag_run(run_id, tags)

        retrieved_tags = forge.get_run_tags(run_id)

        assert retrieved_tags["experiment"] == "baseline"
        assert retrieved_tags["status"] == "approved"

    def test_tag_run_merge_tags(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that tagging merges with existing tags."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        forge.tag_run(run_id, {"tag1": "value1"})
        forge.tag_run(run_id, {"tag2": "value2"})

        tags = forge.get_run_tags(run_id)

        assert tags["tag1"] == "value1"
        assert tags["tag2"] == "value2"

    def test_get_run_tags_nonexistent_run(self, tmp_mlvern_dir):
        """Test getting tags for non-existent run returns empty dict."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        tags = forge.get_run_tags("nonexistent_run")

        assert tags == {}
