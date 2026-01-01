"""Integration tests and cleanup operations for Forge."""

import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mlvern.core.forge import Forge
from mlvern.utils.registry import load_registry, save_registry


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


class TestForgeCleanup:
    """Tests for Forge cleanup and deletion methods."""

    def test_remove_run_requires_confirmation(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test remove_run requires confirm=True."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        result = forge.remove_run(run_id, confirm=False)

        assert result is False

        # Verify run still exists
        assert run_id in forge.list_runs()

    def test_remove_run_success(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test removing a run with confirmation."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        run_id, _ = forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        result = forge.remove_run(run_id, confirm=True)

        assert result is True
        assert run_id not in forge.list_runs()

        run_path = os.path.join(forge.mlvern_dir, "runs", run_id)
        assert not os.path.exists(run_path)

    def test_remove_run_not_found(self, tmp_mlvern_dir):
        """Test remove_run handles non-existent run."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        result = forge.remove_run("nonexistent_run", confirm=True)

        assert result is False

    def test_prune_datasets_requires_confirmation(self, tmp_mlvern_dir, sample_df):
        """Test prune_datasets requires confirm=True."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        result = forge.prune_datasets(older_than_days=0, confirm=False)

        assert result == []
        assert dataset_hash in forge.list_datasets()

    def test_prune_datasets_no_old_datasets(self, tmp_mlvern_dir, sample_df):
        """Test prune_datasets when no datasets are old enough."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        dataset_hash = fp["dataset_hash"]

        result = forge.prune_datasets(older_than_days=365, confirm=True)

        assert result == []
        assert dataset_hash in forge.list_datasets()

    def test_get_project_stats(
        self,
        tmp_mlvern_dir,
        sample_df,
        sample_train_data,
        sample_val_data,
        logistic_model,
    ):
        """Test getting project statistics."""
        forge = Forge("project", tmp_mlvern_dir)
        forge.init()

        fp, _ = forge.register_dataset(sample_df, "target")
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        forge.run(logistic_model, X_train, y_train, X_val, y_val, {}, fp)

        forge.register_model(logistic_model, {"name": "test"})

        stats = forge.get_project_stats()

        assert stats["project"] == "project"
        assert stats["datasets_count"] == 1
        assert stats["runs_count"] == 1
        assert stats["models_count"] == 1
        assert "total_size_mb" in stats
