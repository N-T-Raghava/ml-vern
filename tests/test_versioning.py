"""
Comprehensive tests for versioning and run management.

Covers:
- run_manager.py: create_run() function and run management
- registry.py: registry load/save/init operations
- fingerprint.py: dataset fingerprinting
- register.py: dataset registration
"""

import json
import os
import tempfile
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mlvern.data.fingerprint import fingerprint_dataset
from mlvern.data.register import register_dataset
from mlvern.utils.hashing import hash_object
from mlvern.utils.registry import (
    init_registry,
    list_models_registry,
    load_registry,
    register_model_metadata,
    save_registry,
)
from mlvern.version.run_manager import create_run

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def tmp_mlvern_dir():
    """Create a temporary mlvern directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_df():
    """Create a sample dataframe."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feat1": np.random.randn(30),
            "feat2": np.random.randn(30),
            "feat3": np.random.randn(30),
            "target": np.random.randint(0, 2, 30),
        }
    )


@pytest.fixture
def sample_model():
    """Create a simple trained model."""
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_metrics():
    """Create sample metrics dict."""
    return {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.97,
        "f1": 0.95,
    }


@pytest.fixture
def sample_config():
    """Create sample training config."""
    return {
        "model": "logistic_regression",
        "learning_rate": 0.01,
        "epochs": 100,
        "batch_size": 32,
    }


@pytest.fixture
def initialized_mlvern_dir(tmp_mlvern_dir):
    """Create an initialized mlvern directory."""
    init_registry(tmp_mlvern_dir, "test_project")
    for d in ["datasets", "runs"]:
        os.makedirs(os.path.join(tmp_mlvern_dir, d), exist_ok=True)
    return tmp_mlvern_dir


# ============================================================================
# TESTS: hash_object()
# ============================================================================


class TestHashObject:
    """Tests for hash_object utility function."""

    def test_hash_object_deterministic(self):
        """Test that same object produces same hash."""
        obj = {"a": 1, "b": 2, "c": [3, 4, 5]}
        hash1 = hash_object(obj)
        hash2 = hash_object(obj)

        assert hash1 == hash2

    def test_hash_object_different_objects(self):
        """Test that different objects produce different hashes."""
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1, "b": 3}

        hash1 = hash_object(obj1)
        hash2 = hash_object(obj2)

        assert hash1 != hash2

    def test_hash_object_order_independent(self):
        """Test that dict order doesn't affect hash."""
        obj1 = {"a": 1, "b": 2, "c": 3}
        obj2 = {"c": 3, "a": 1, "b": 2}

        hash1 = hash_object(obj1)
        hash2 = hash_object(obj2)

        assert hash1 == hash2

    def test_hash_object_complex_types(self):
        """Test hashing complex nested structures."""
        obj = {
            "nested": {"deep": {"value": [1, 2, 3]}},
            "list": [1, 2, {"key": "val"}],
            "number": 42,
            "string": "test",
        }
        hash_val = hash_object(obj)

        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex length


# ============================================================================
# TESTS: fingerprint_dataset()
# ============================================================================


class TestFingerprintDataset:
    """Tests for dataset fingerprinting."""

    def test_fingerprint_basic(self, sample_df):
        """Test basic fingerprinting."""
        fp = fingerprint_dataset(sample_df, "target")

        assert "dataset_hash" in fp
        assert "rows" in fp
        assert "columns" in fp
        assert "schema" in fp

    def test_fingerprint_shape(self, sample_df):
        """Test fingerprint captures correct shape."""
        fp = fingerprint_dataset(sample_df, "target")

        assert fp["rows"] == 30
        assert fp["columns"] == 4

    def test_fingerprint_schema(self, sample_df):
        """Test fingerprint schema structure."""
        fp = fingerprint_dataset(sample_df, "target")
        schema = fp["schema"]

        assert "columns" in schema
        assert "dtypes" in schema
        assert "target" in schema
        assert schema["target"] == "target"
        assert set(schema["columns"]) == {"feat1", "feat2", "feat3", "target"}

    def test_fingerprint_deterministic(self, sample_df):
        """Test that same dataframe produces same fingerprint."""
        fp1 = fingerprint_dataset(sample_df, "target")
        fp2 = fingerprint_dataset(sample_df, "target")

        assert fp1["dataset_hash"] == fp2["dataset_hash"]

    def test_fingerprint_different_dataframes(self):
        """Test different dataframes produce different fingerprints."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"a": [1, 2, 4], "target": [0, 1, 0]})

        fp1 = fingerprint_dataset(df1, "target")
        fp2 = fingerprint_dataset(df2, "target")

        assert fp1["dataset_hash"] != fp2["dataset_hash"]

    def test_fingerprint_hash_short(self, sample_df):
        """Test that dataset hash is shortened."""
        fp = fingerprint_dataset(sample_df, "target")

        assert len(fp["dataset_hash"]) == 12  # First 12 chars of SHA256

    def test_fingerprint_different_targets(self, sample_df):
        """Test fingerprint with different target columns."""
        fp1 = fingerprint_dataset(sample_df, "target")
        fp2 = fingerprint_dataset(sample_df, "feat1")

        assert fp1["dataset_hash"] != fp2["dataset_hash"]

    def test_fingerprint_dtype_preservation(self):
        """Test that dtypes are correctly captured."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "target": [0, 1, 0],
            }
        )

        fp = fingerprint_dataset(df, "target")
        dtypes = fp["schema"]["dtypes"]

        assert "int64" in str(dtypes["int_col"]) or "int32" in str(dtypes["int_col"])
        assert "float" in str(dtypes["float_col"])


# ============================================================================
# TESTS: Registry Operations
# ============================================================================


class TestRegistryInit:
    """Tests for registry initialization."""

    def test_init_registry_creates_file(self, tmp_mlvern_dir):
        """Test init_registry creates registry.json."""
        init_registry(tmp_mlvern_dir, "test_project")

        registry_path = os.path.join(tmp_mlvern_dir, "registry.json")
        assert os.path.exists(registry_path)

    def test_init_registry_structure(self, tmp_mlvern_dir):
        """Test registry has correct initial structure."""
        init_registry(tmp_mlvern_dir, "my_project")

        registry = load_registry(tmp_mlvern_dir)

        assert registry["project"] == "my_project"
        assert "created_at" in registry
        assert "datasets" in registry
        assert "runs" in registry
        assert registry["datasets"] == {}
        assert registry["runs"] == {}

    def test_init_registry_timestamp(self, tmp_mlvern_dir):
        """Test registry contains valid timestamp."""
        init_registry(tmp_mlvern_dir, "project")

        registry = load_registry(tmp_mlvern_dir)
        timestamp = registry["created_at"]

        # Should be parseable as ISO format
        datetime.fromisoformat(timestamp)


class TestRegistryLoadSave:
    """Tests for loading and saving registry."""

    def test_load_nonexistent_registry(self, tmp_mlvern_dir):
        """Test loading registry when file doesn't exist."""
        registry = load_registry(tmp_mlvern_dir)

        assert registry == {}

    def test_save_and_load_registry(self, tmp_mlvern_dir):
        """Test save and load roundtrip."""
        test_registry = {
            "project": "test",
            "datasets": {"hash1": {"rows": 100}},
            "runs": {"run1": {"metrics": {}}},
        }

        save_registry(tmp_mlvern_dir, test_registry)
        loaded = load_registry(tmp_mlvern_dir)

        assert loaded == test_registry

    def test_registry_json_format(self, tmp_mlvern_dir):
        """Test registry is saved as valid JSON."""
        registry = {
            "project": "test",
            "data": {"nested": [1, 2, 3]},
        }

        save_registry(tmp_mlvern_dir, registry)

        registry_path = os.path.join(tmp_mlvern_dir, "registry.json")
        with open(registry_path) as f:
            loaded = json.load(f)

        assert loaded == registry

    def test_save_registry_overwrites(self, tmp_mlvern_dir):
        """Test that saving overwrites previous registry."""
        registry1 = {"project": "old"}
        registry2 = {"project": "new", "extra": "data"}

        save_registry(tmp_mlvern_dir, registry1)
        save_registry(tmp_mlvern_dir, registry2)

        loaded = load_registry(tmp_mlvern_dir)
        assert loaded["project"] == "new"
        assert loaded["extra"] == "data"


# ============================================================================
# TESTS: register_dataset()
# ============================================================================


class TestRegisterDataset:
    """Tests for dataset registration."""

    def test_register_dataset_creates_directory(
        self, initialized_mlvern_dir, sample_df
    ):
        """Test registering dataset creates directory structure."""
        fp, is_new = register_dataset(sample_df, "target", initialized_mlvern_dir)

        assert is_new
        dataset_path = os.path.join(
            initialized_mlvern_dir, "datasets", fp["dataset_hash"]
        )
        assert os.path.exists(dataset_path)

    def test_register_dataset_saves_schema(self, initialized_mlvern_dir, sample_df):
        """Test dataset registration saves schema."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        schema_path = os.path.join(
            initialized_mlvern_dir,
            "datasets",
            fp["dataset_hash"],
            "schema.json",
        )
        assert os.path.exists(schema_path)

        with open(schema_path) as f:
            schema = json.load(f)

        assert "columns" in schema
        assert "dtypes" in schema
        assert schema["target"] == "target"

    def test_register_dataset_updates_registry(self, initialized_mlvern_dir, sample_df):
        """Test registering dataset updates registry."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        registry = load_registry(initialized_mlvern_dir)
        assert fp["dataset_hash"] in registry["datasets"]

        dataset_entry = registry["datasets"][fp["dataset_hash"]]
        assert dataset_entry["rows"] == 30
        assert dataset_entry["columns"] == 4
        assert dataset_entry["target"] == "target"

    def test_register_dataset_duplicate_returns_cached(
        self, initialized_mlvern_dir, sample_df
    ):
        """Test registering same dataset twice returns cached result."""
        fp1, is_new1 = register_dataset(sample_df, "target", initialized_mlvern_dir)
        fp2, is_new2 = register_dataset(sample_df, "target", initialized_mlvern_dir)

        assert is_new1
        assert not is_new2
        assert fp1["dataset_hash"] == fp2["dataset_hash"]

    def test_register_dataset_returns_fingerprint(
        self, initialized_mlvern_dir, sample_df
    ):
        """Test registration returns complete fingerprint."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        assert "dataset_hash" in fp
        assert "rows" in fp
        assert "columns" in fp
        assert "schema" in fp
        assert fp["rows"] == 30
        assert fp["columns"] == 4

    def test_register_different_datasets(self, initialized_mlvern_dir):
        """Test registering multiple different datasets."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9], "target": [1, 0, 1]})

        fp1, _ = register_dataset(df1, "target", initialized_mlvern_dir)
        fp2, _ = register_dataset(df2, "target", initialized_mlvern_dir)

        registry = load_registry(initialized_mlvern_dir)
        assert len(registry["datasets"]) == 2
        assert fp1["dataset_hash"] in registry["datasets"]
        assert fp2["dataset_hash"] in registry["datasets"]


# ============================================================================
# TESTS: create_run()
# ============================================================================


class TestCreateRun:
    """Tests for creating training runs."""

    def test_create_run_basic(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test basic run creation."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        assert run_id is not None
        assert run_id.startswith("run_")

    def test_create_run_directory_structure(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run creates proper directory structure."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        run_path = os.path.join(initialized_mlvern_dir, "runs", run_id)
        assert os.path.exists(run_path)
        assert os.path.exists(os.path.join(run_path, "meta.json"))
        assert os.path.exists(os.path.join(run_path, "config.json"))
        assert os.path.exists(os.path.join(run_path, "metrics.json"))
        assert os.path.exists(os.path.join(run_path, "artifacts"))

    def test_create_run_saves_metadata(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run metadata is saved correctly."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        meta_path = os.path.join(initialized_mlvern_dir, "runs", run_id, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["run_id"] == run_id
        assert meta["dataset_hash"] == fp["dataset_hash"]
        assert "timestamp" in meta

    def test_create_run_saves_config(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run config is saved correctly."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        config_path = os.path.join(
            initialized_mlvern_dir, "runs", run_id, "config.json"
        )
        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config == sample_config

    def test_create_run_saves_metrics(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run metrics are saved correctly."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        metrics_path = os.path.join(
            initialized_mlvern_dir, "runs", run_id, "metrics.json"
        )
        with open(metrics_path) as f:
            saved_metrics = json.load(f)

        assert saved_metrics == sample_metrics

    def test_create_run_saves_model(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run saves model artifact."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        model_path = os.path.join(
            initialized_mlvern_dir, "runs", run_id, "artifacts", "model.pkl"
        )
        assert os.path.exists(model_path)

        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

    def test_create_run_updates_registry(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run creation updates registry."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        registry = load_registry(initialized_mlvern_dir)
        assert run_id in registry["runs"]

        run_entry = registry["runs"][run_id]
        assert run_entry["dataset_hash"] == fp["dataset_hash"]
        assert run_entry["model"] == "LogisticRegression"
        assert run_entry["metrics"] == sample_metrics
        assert "created_at" in run_entry
        assert "path" in run_entry

    def test_create_run_unique_ids(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test that multiple runs get unique IDs."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id1 = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )
        run_id2 = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        assert run_id1 != run_id2

    def test_create_run_timestamp_format(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test run timestamp is in ISO format."""
        fp, _ = register_dataset(sample_df, "target", initialized_mlvern_dir)

        run_id = create_run(
            initialized_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        meta_path = os.path.join(initialized_mlvern_dir, "runs", run_id, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)

        # Should be parseable as ISO format
        datetime.fromisoformat(meta["timestamp"])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestVersioningIntegration:
    """Integration tests for complete versioning workflows."""

    def test_complete_registration_run_workflow(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_model,
        sample_metrics,
        sample_config,
    ):
        """Test complete workflow: init -> register -> run."""
        # Initialize
        init_registry(tmp_mlvern_dir, "project")
        for d in ["datasets", "runs"]:
            os.makedirs(os.path.join(tmp_mlvern_dir, d), exist_ok=True)

        # Register dataset
        fp, is_new = register_dataset(sample_df, "target", tmp_mlvern_dir)
        assert is_new

        # Create run
        run_id = create_run(
            tmp_mlvern_dir,
            fp,
            sample_model,
            sample_metrics,
            sample_config,
        )

        # Verify final state
        registry = load_registry(tmp_mlvern_dir)
        assert fp["dataset_hash"] in registry["datasets"]
        assert run_id in registry["runs"]

    def test_multiple_datasets_multiple_runs(
        self, tmp_mlvern_dir, sample_model, sample_metrics, sample_config
    ):
        """Test managing multiple datasets and runs."""
        init_registry(tmp_mlvern_dir, "project")
        for d in ["datasets", "runs"]:
            os.makedirs(os.path.join(tmp_mlvern_dir, d), exist_ok=True)

        # Create multiple datasets
        df1 = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        df2 = pd.DataFrame({"b": [4, 5, 6], "c": [7, 8, 9], "target": [1, 0, 1]})

        fp1, _ = register_dataset(df1, "target", tmp_mlvern_dir)
        fp2, _ = register_dataset(df2, "target", tmp_mlvern_dir)

        # Create runs for each dataset
        run_id1 = create_run(
            tmp_mlvern_dir, fp1, sample_model, sample_metrics, sample_config
        )
        run_id2 = create_run(
            tmp_mlvern_dir, fp2, sample_model, sample_metrics, sample_config
        )

        registry = load_registry(tmp_mlvern_dir)
        assert len(registry["datasets"]) == 2
        assert len(registry["runs"]) == 2
        assert registry["runs"][run_id1]["dataset_hash"] == fp1["dataset_hash"]
        assert registry["runs"][run_id2]["dataset_hash"] == fp2["dataset_hash"]

    def test_run_lineage_traceability(
        self, tmp_mlvern_dir, sample_df, sample_model, sample_metrics
    ):
        """Test that runs are traceable to their datasets."""
        init_registry(tmp_mlvern_dir, "project")
        for d in ["datasets", "runs"]:
            os.makedirs(os.path.join(tmp_mlvern_dir, d), exist_ok=True)

        fp, _ = register_dataset(sample_df, "target", tmp_mlvern_dir)

        config1 = {"model": "logistic"}
        config2 = {"model": "rf"}

        run_id1 = create_run(tmp_mlvern_dir, fp, sample_model, sample_metrics, config1)
        run_id2 = create_run(tmp_mlvern_dir, fp, sample_model, sample_metrics, config2)

        registry = load_registry(tmp_mlvern_dir)

        # Both runs should reference the same dataset
        assert (
            registry["runs"][run_id1]["dataset_hash"]
            == registry["runs"][run_id2]["dataset_hash"]
            == fp["dataset_hash"]
        )

        # But have different configs
        config_path1 = os.path.join(tmp_mlvern_dir, "runs", run_id1, "config.json")
        config_path2 = os.path.join(tmp_mlvern_dir, "runs", run_id2, "config.json")

        with open(config_path1) as f:
            saved_config1 = json.load(f)
        with open(config_path2) as f:
            saved_config2 = json.load(f)

        assert saved_config1["model"] == "logistic"
        assert saved_config2["model"] == "rf"


# ============================================================================
# TESTS: Registry - Model Management
# ============================================================================


class TestRegistryModelManagement:
    """Tests for model registry functions."""

    def test_register_model_metadata(self, tmp_mlvern_dir):
        """Test registering model metadata in registry."""

        init_registry(tmp_mlvern_dir, "test_project")

        metadata = {
            "model_name": "lr_v1",
            "hyperparameters": {"C": 1.0},
            "source_run_id": "run_123",
        }

        register_model_metadata(tmp_mlvern_dir, "model_001", metadata)

        models = list_models_registry(tmp_mlvern_dir)

        assert "model_001" in models
        assert models["model_001"]["model_name"] == "lr_v1"
        assert models["model_001"]["source_run_id"] == "run_123"
        assert "registered_at" in models["model_001"]

    def test_list_models_empty(self, tmp_mlvern_dir):
        """Test list_models_registry returns empty dict initially."""
        from mlvern.utils.registry import list_models_registry

        init_registry(tmp_mlvern_dir, "test_project")

        models = list_models_registry(tmp_mlvern_dir)

        assert models == {}

    def test_list_models_multiple(self, tmp_mlvern_dir):
        """Test listing multiple registered models."""

        init_registry(tmp_mlvern_dir, "test_project")

        register_model_metadata(tmp_mlvern_dir, "model_001", {"name": "lr"})
        register_model_metadata(tmp_mlvern_dir, "model_002", {"name": "rf"})

        models = list_models_registry(tmp_mlvern_dir)

        assert len(models) == 2
        assert "model_001" in models
        assert "model_002" in models


# ============================================================================
# TESTS: Registry - Run Tagging
# ============================================================================


class TestRegistryRunTagging:
    """Tests for run tagging and search functions."""

    def test_tag_run_success(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_config,
    ):
        """Test adding tags to a run."""
        from mlvern.data.fingerprint import fingerprint_dataset
        from mlvern.utils.registry import get_run_tags, tag_run

        fp = fingerprint_dataset(sample_df, "target")
        run_id = create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.95}, sample_config
        )

        tags = {"experiment": "baseline", "status": "approved"}
        tag_run(initialized_mlvern_dir, run_id, tags)

        retrieved_tags = get_run_tags(initialized_mlvern_dir, run_id)

        assert retrieved_tags["experiment"] == "baseline"
        assert retrieved_tags["status"] == "approved"

    def test_tag_run_not_found(self, tmp_mlvern_dir):
        """Test tagging non-existent run raises error."""
        from mlvern.utils.registry import tag_run

        init_registry(tmp_mlvern_dir, "test_project")

        with pytest.raises(KeyError):
            tag_run(tmp_mlvern_dir, "nonexistent_run", {"tag": "value"})

    def test_get_run_tags_empty(
        self, initialized_mlvern_dir, sample_df, sample_model, sample_config
    ):
        """Test getting tags from run with no tags."""
        from mlvern.data.fingerprint import fingerprint_dataset
        from mlvern.utils.registry import get_run_tags

        fp = fingerprint_dataset(sample_df, "target")
        run_id = create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.95}, sample_config
        )

        tags = get_run_tags(initialized_mlvern_dir, run_id)

        assert tags == {}

    def test_tag_run_merge_tags(
        self, initialized_mlvern_dir, sample_df, sample_model, sample_config
    ):
        """Test that tagging merges with existing tags."""
        from mlvern.data.fingerprint import fingerprint_dataset
        from mlvern.utils.registry import get_run_tags, tag_run

        fp = fingerprint_dataset(sample_df, "target")
        run_id = create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.95}, sample_config
        )

        tag_run(initialized_mlvern_dir, run_id, {"tag1": "value1"})
        tag_run(initialized_mlvern_dir, run_id, {"tag2": "value2"})

        tags = get_run_tags(initialized_mlvern_dir, run_id)

        assert tags["tag1"] == "value1"
        assert tags["tag2"] == "value2"

    def test_search_runs_by_tag(
        self, initialized_mlvern_dir, sample_df, sample_model, sample_config
    ):
        """Test searching runs by tag key-value pair."""
        from mlvern.data.fingerprint import fingerprint_dataset
        from mlvern.utils.registry import search_runs_by_tag, tag_run

        fp = fingerprint_dataset(sample_df, "target")

        run_id1 = create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.95}, sample_config
        )
        run_id2 = create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.93}, sample_config
        )

        tag_run(initialized_mlvern_dir, run_id1, {"status": "approved"})
        tag_run(initialized_mlvern_dir, run_id2, {"status": "review"})
        tag_run(initialized_mlvern_dir, run_id1, {"experiment": "exp1"})

        approved_runs = search_runs_by_tag(initialized_mlvern_dir, "status", "approved")

        assert run_id1 in approved_runs
        assert run_id2 not in approved_runs

    def test_search_runs_by_tag_no_matches(
        self,
        initialized_mlvern_dir,
        sample_df,
        sample_model,
        sample_config,
    ):
        """Test search with no matching tags."""
        from mlvern.data.fingerprint import fingerprint_dataset
        from mlvern.utils.registry import search_runs_by_tag

        fp = fingerprint_dataset(sample_df, "target")
        create_run(
            initialized_mlvern_dir, fp, sample_model, {"accuracy": 0.95}, sample_config
        )

        results = search_runs_by_tag(initialized_mlvern_dir, "nonexistent", "value")

        assert results == []


# ============================================================================
# TESTS: Artifact Utilities
# ============================================================================


class TestArtifactUtilities:
    """Tests for artifact save/load utilities."""

    def test_save_model_safe(self, tmp_path, sample_model):
        """Test saving a model with save_model_safe."""
        from mlvern.utils.artifact import load_model_safe, save_model_safe

        model_path = str(tmp_path / "model.pkl")
        save_model_safe(sample_model, model_path)

        assert os.path.exists(model_path)

        loaded = load_model_safe(model_path, safe=False)
        assert loaded is not None

    def test_save_model_with_metadata(self, tmp_path, sample_model):
        """Test saving model with metadata."""
        from mlvern.utils.artifact import get_model_metadata, save_model_safe

        model_path = str(tmp_path / "model.pkl")
        metadata = {"version": "1.0", "accuracy": 0.95}

        save_model_safe(sample_model, model_path, metadata=metadata)

        retrieved_metadata = get_model_metadata(model_path)

        assert retrieved_metadata["version"] == "1.0"
        assert retrieved_metadata["accuracy"] == 0.95

    def test_get_model_metadata_no_file(self, tmp_path):
        """Test getting metadata when metadata file doesn't exist."""
        from mlvern.utils.artifact import get_model_metadata

        model_path = str(tmp_path / "nonexistent.pkl")
        metadata = get_model_metadata(model_path)

        assert metadata == {}

    def test_load_model_safe_not_found(self, tmp_path):
        """Test load_model_safe raises error for missing file."""
        from mlvern.utils.artifact import load_model_safe

        model_path = str(tmp_path / "nonexistent.pkl")

        with pytest.raises(FileNotFoundError):
            load_model_safe(model_path)

    def test_remove_directory_safe_requires_confirm(self, tmp_path):
        """Test remove_directory_safe requires confirm=True."""
        from mlvern.utils.artifact import remove_directory_safe

        test_dir = tmp_path / "to_remove"
        test_dir.mkdir()

        result = remove_directory_safe(str(test_dir), confirm=False)

        assert result is False
        assert test_dir.exists()

    def test_remove_directory_safe_success(self, tmp_path):
        """Test removing directory with confirmation."""
        from mlvern.utils.artifact import remove_directory_safe

        test_dir = tmp_path / "to_remove"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test")

        result = remove_directory_safe(str(test_dir), confirm=True)

        assert result is True
        assert not test_dir.exists()

    def test_get_directory_size_mb(self, tmp_path):
        """Test calculating directory size."""
        from mlvern.utils.artifact import get_directory_size_mb

        test_dir = tmp_path / "test_size"
        test_dir.mkdir()

        file1 = test_dir / "file1.txt"
        file1.write_text("a" * 1000)

        file2 = test_dir / "file2.txt"
        file2.write_text("b" * 2000)

        size_mb = get_directory_size_mb(str(test_dir))

        assert size_mb > 0
        assert size_mb < 1

    def test_get_directory_created_time(self, tmp_path):
        """Test getting directory creation time."""
        from mlvern.utils.artifact import get_directory_created_time

        test_dir = tmp_path / "test_time"
        test_dir.mkdir()

        created_time = get_directory_created_time(str(test_dir))

        assert created_time is not None
        from datetime import datetime

        assert isinstance(created_time, datetime)

    def test_get_directory_created_time_nonexistent(self, tmp_path):
        """Test getting time for non-existent directory."""
        from mlvern.utils.artifact import get_directory_created_time

        nonexistent = tmp_path / "nonexistent"

        result = get_directory_created_time(str(nonexistent))

        assert result is None
