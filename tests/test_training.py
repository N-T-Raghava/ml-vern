"""
Comprehensive tests for training module and core Forge functionality.

Covers:
- trainer.py: train_model() function
- forge.py: Forge class with training workflows
"""

import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlvern.core.forge import Forge
from mlvern.train.trainer import train_model
from mlvern.utils.registry import load_registry, save_registry

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmp_mlvern_dir():
    """Create a temporary directory for mlvern projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_train_data():
    """Create sample training data with features and target."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_val_data():
    """Create sample validation data."""
    np.random.seed(43)
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, 30)
    return X, y


@pytest.fixture
def logistic_model():
    """Create a logistic regression model."""
    return LogisticRegression(random_state=42)


@pytest.fixture
def forest_model():
    """Create a random forest classifier."""
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def sample_df():
    """Create sample dataframe with features and target."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "feat3": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        }
    )


# ============================================================================
# TESTS: train_model()
# ============================================================================


class TestTrainModel:
    """Tests for the train_model() function."""

    def test_train_model_basic(self, sample_train_data, logistic_model):
        """Test basic training without validation data."""
        X, y = sample_train_data
        model, metrics = train_model(logistic_model, X, y)

        assert model is not None
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # No validation data provided

    def test_train_model_with_validation(
        self, sample_train_data, sample_val_data, logistic_model
    ):
        """Test training with validation data."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        model, metrics = train_model(logistic_model, X_train, y_train, X_val, y_val)

        assert model is not None
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_model_accuracy_score(
        self, sample_train_data, sample_val_data, forest_model
    ):
        """Test that metrics contain valid accuracy scores."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        model, metrics = train_model(forest_model, X_train, y_train, X_val, y_val)

        # Manually calculate expected accuracy
        preds = model.predict(X_val)
        expected_accuracy = accuracy_score(y_val, preds)

        assert metrics["accuracy"] == expected_accuracy

    def test_train_model_returns_fitted_model(self, sample_train_data, logistic_model):
        """Test that returned model is fitted."""
        X, y = sample_train_data
        model, _ = train_model(logistic_model, X, y)

        # Fitted model should have coef_ attribute
        assert hasattr(model, "coef_")
        assert model.coef_ is not None

    def test_train_model_different_algorithms(self, sample_train_data, sample_val_data):
        """Test training with different model algorithms."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        lr_model, lr_metrics = train_model(
            LogisticRegression(random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
        )
        rf_model, rf_metrics = train_model(
            RandomForestClassifier(n_estimators=5, random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
        )

        assert "accuracy" in lr_metrics
        assert "accuracy" in rf_metrics
        assert lr_model is not None
        assert rf_model is not None


# ============================================================================
# TESTS: Forge Class
# ============================================================================


class TestForgeInitialization:
    """Tests for Forge class initialization."""

    def test_forge_init_creates_project(self, tmp_mlvern_dir):
        """Test Forge initialization creates project structure."""
        forge = Forge("test_project", tmp_mlvern_dir)
        assert forge.project == "test_project"
        assert forge.base_dir == tmp_mlvern_dir
        assert forge.mlvern_dir == os.path.join(tmp_mlvern_dir, ".mlvern_test_project")

    def test_forge_init_method(self, tmp_mlvern_dir):
        """Test Forge.init() creates required directories."""
        forge = Forge("myproject", tmp_mlvern_dir)
        forge.init()

        # Check directories are created
        assert os.path.exists(os.path.join(forge.mlvern_dir, "datasets"))
        assert os.path.exists(os.path.join(forge.mlvern_dir, "runs"))
        assert os.path.exists(os.path.join(forge.mlvern_dir, "models"))

    def test_forge_init_creates_registry(self, tmp_mlvern_dir):
        """Test Forge.init() creates registry.json."""
        forge = Forge("myproject", tmp_mlvern_dir)
        forge.init()

        registry_path = os.path.join(forge.mlvern_dir, "registry.json")
        assert os.path.exists(registry_path)

        registry = load_registry(forge.mlvern_dir)
        assert registry["project"] == "myproject"
        assert "created_at" in registry
        assert "datasets" in registry
        assert "runs" in registry

    def test_forge_multiple_init_idempotent(self, tmp_mlvern_dir):
        """Test that multiple init() calls don't cause errors."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()
        forge.init()  # Should not raise

        assert os.path.exists(os.path.join(forge.mlvern_dir, "datasets"))


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
        """Test registering the same dataset twice returns cached result."""
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


class TestForgeIntegration:
    """Integration tests for complete Forge workflows."""

    def test_forge_complete_workflow(
        self, tmp_mlvern_dir, sample_df, sample_train_data, sample_val_data
    ):
        """Test complete workflow: init -> register -> train -> list."""
        forge = Forge("ml_project", tmp_mlvern_dir)
        forge.init()

        # Register dataset
        fp, is_new = forge.register_dataset(sample_df, "target")
        assert is_new

        # List datasets
        datasets = forge.list_datasets()
        assert len(datasets) == 1

        # Train models
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id1, _ = forge.run(
            LogisticRegression(random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
            {"model": "logistic"},
            fp,
        )
        run_id2, _ = forge.run(
            RandomForestClassifier(n_estimators=5, random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
            {"model": "rf"},
            fp,
        )

        # List runs
        runs = forge.list_runs()
        assert len(runs) == 2
        assert run_id1 in runs
        assert run_id2 in runs

    def test_forge_different_projects(self, tmp_mlvern_dir, sample_df):
        """Test managing multiple projects in same directory."""
        forge1 = Forge("project_a", tmp_mlvern_dir)
        forge2 = Forge("project_b", tmp_mlvern_dir)

        forge1.init()
        forge2.init()

        fp1, _ = forge1.register_dataset(sample_df, "target")
        fp2, _ = forge2.register_dataset(sample_df, "target")

        # Verify they're in different registries
        registry1 = load_registry(forge1.mlvern_dir)
        registry2 = load_registry(forge2.mlvern_dir)

        assert registry1["project"] == "project_a"
        assert registry2["project"] == "project_b"
        assert fp1["dataset_hash"] in registry1["datasets"]
        assert fp2["dataset_hash"] in registry2["datasets"]


class TestRegistrySaveOperations:
    """Tests for saving registry state during Forge operations."""

    def test_forge_run_persists_registry(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test that run updates are persisted in registry file."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        # Reload registry from disk to verify persistence
        registry = load_registry(forge.mlvern_dir)

        assert run_id in registry["runs"]
        assert registry["runs"][run_id]["dataset_hash"] == fp["dataset_hash"]
        assert registry["runs"][run_id]["model"] == "LogisticRegression"

    def test_manual_registry_save_load_roundtrip(self, tmp_mlvern_dir):
        """Test manual save/load of registry with custom data."""
        custom_registry = {
            "project": "test",
            "datasets": {
                "hash123": {
                    "rows": 100,
                    "columns": 5,
                    "target": "label",
                }
            },
            "runs": {
                "run_001": {
                    "model": "RandomForest",
                    "accuracy": 0.95,
                }
            },
            "metadata": {
                "version": "1.0",
                "author": "test_user",
            },
        }

        # Save registry
        save_registry(tmp_mlvern_dir, custom_registry)

        # Load and verify
        loaded = load_registry(tmp_mlvern_dir)

        assert loaded == custom_registry
        assert loaded["metadata"]["version"] == "1.0"
        assert loaded["datasets"]["hash123"]["rows"] == 100
        assert loaded["runs"]["run_001"]["accuracy"] == 0.95
