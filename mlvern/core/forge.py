import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from mlvern.data.register import register_dataset
from mlvern.train.trainer import train_model
from mlvern.utils.artifact import (
    get_directory_created_time,
    get_directory_size_mb,
    load_model_safe,
    remove_directory_safe,
)
from mlvern.utils.registry import (
    get_run_tags,
    init_registry,
    load_registry,
    register_model_metadata,
    save_registry,
    tag_run,
)
from mlvern.version.run_manager import create_run


class Forge:
    def __init__(self, project: str, base_dir: str = "."):
        self.project = project
        self.base_dir = os.path.abspath(base_dir)
        self.mlvern_dir = os.path.join(
            self.base_dir, f".mlvern_{project}"
        )

    def init(self):
        """Initialize the mlvern project directory structure."""
        os.makedirs(self.mlvern_dir, exist_ok=True)
        for d in ["datasets", "runs", "models"]:
            os.makedirs(os.path.join(self.mlvern_dir, d), exist_ok=True)

        registry_path = os.path.join(self.mlvern_dir, "registry.json")
        if not os.path.exists(registry_path):
            init_registry(self.mlvern_dir, self.project)

    # -------- DATASET ACCESSORS --------
    def register_dataset(self, df, target: str):
        return register_dataset(df, target, self.mlvern_dir)

    def list_datasets(self):
        """List all registered datasets in the project."""
        return load_registry(self.mlvern_dir).get("datasets", {})

    def get_dataset_path(self, dataset_hash: str) -> str:
        """Get the filesystem path for a dataset by hash.

        Args:
            dataset_hash: The dataset hash identifier

        Returns:
            Absolute path to the dataset directory

        Raises:
            ValueError: If dataset hash not found
        """
        datasets_dir = os.path.join(self.mlvern_dir, "datasets")
        dataset_path = os.path.join(datasets_dir, dataset_hash)

        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset '{dataset_hash}' not found")

        return dataset_path

    def load_dataset(self, dataset_hash: str) -> Optional[Dict[str, Any]]:
        """Load dataset metadata and paths by hash.

        Args:
            dataset_hash: The dataset hash identifier

        Returns:
            Dictionary containing dataset info and paths to reports/plots

        Raises:
            ValueError: If dataset not found
        """
        dataset_path = self.get_dataset_path(dataset_hash)

        # Load schema if available
        schema_path = os.path.join(dataset_path, "schema.json")
        schema = {}
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema = json.load(f)

        # Collect report paths
        report_paths = {}
        reports_dir = os.path.join(dataset_path, "reports")
        if os.path.exists(reports_dir):
            for fname in os.listdir(reports_dir):
                if fname.endswith(".json"):
                    report_paths[fname] = os.path.join(reports_dir, fname)

        # Collect plot paths
        plot_paths = []
        plots_dir = os.path.join(dataset_path, "plots")
        if os.path.exists(plots_dir):
            for root, dirs, files in os.walk(plots_dir):
                for fname in files:
                    if fname.endswith((".png", ".jpg", ".jpeg")):
                        plot_paths.append(os.path.join(root, fname))

        registry = load_registry(self.mlvern_dir)
        dataset_meta = registry.get("datasets", {}).get(dataset_hash, {})

        return {
            "dataset_hash": dataset_hash,
            "path": dataset_path,
            "schema": schema,
            "metadata": dataset_meta,
            "report_paths": report_paths,
            "plot_paths": plot_paths,
        }

    # -------- RUN/MODEL ACCESSORS --------
    def list_runs(self) -> Dict[str, Any]:
        """List all runs in the project."""
        return load_registry(self.mlvern_dir).get("runs", {})

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run metadata and information by run_id.

        Args:
            run_id: The run identifier

        Returns:
            Dictionary with run metadata, metrics, and paths

        Raises:
            ValueError: If run not found
        """
        registry = load_registry(self.mlvern_dir)
        runs = registry.get("runs", {})

        if run_id not in runs:
            raise ValueError(f"Run '{run_id}' not found")

        run_path = os.path.join(self.mlvern_dir, "runs", run_id)

        if not os.path.exists(run_path):
            raise ValueError(f"Run path does not exist: {run_path}")

        # Load metadata
        meta = {}
        meta_path = os.path.join(run_path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)

        # Load metrics
        metrics = {}
        metrics_path = os.path.join(run_path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

        # Load config
        config = {}
        config_path = os.path.join(run_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

        return {
            "run_id": run_id,
            "path": run_path,
            "metadata": meta,
            "metrics": metrics,
            "config": config,
            "tags": get_run_tags(self.mlvern_dir, run_id),
            "registry_info": runs[run_id],
        }

    def get_run_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a specific run.

        Args:
            run_id: The run identifier

        Returns:
            Metrics dictionary
        """
        run_info = self.get_run(run_id)
        return run_info.get("metrics", {})

    def get_run_artifacts(self, run_id: str) -> Dict[str, str]:
        """Get paths to all artifacts for a run (model, config, metrics, etc).

        Args:
            run_id: The run identifier

        Returns:
            Dictionary mapping artifact names to their filesystem paths
        """
        run_path = os.path.join(self.mlvern_dir, "runs", run_id)

        if not os.path.exists(run_path):
            raise ValueError(f"Run '{run_id}' not found")

        artifacts = {}
        artifacts["run_dir"] = run_path

        # Standard files
        for fname in ["meta.json", "config.json", "metrics.json"]:
            fpath = os.path.join(run_path, fname)
            if os.path.exists(fpath):
                artifacts[fname] = fpath

        # Model artifacts
        artifacts_dir = os.path.join(run_path, "artifacts")
        if os.path.exists(artifacts_dir):
            for fname in os.listdir(artifacts_dir):
                fpath = os.path.join(artifacts_dir, fname)
                artifacts[f"artifact_{fname}"] = fpath

        return artifacts

    def load_model(self, run_id: str, safe: bool = True) -> Any:
        """Load a trained model from a run.

        Args:
            run_id: The run identifier
            safe: If True, warn about pickle security risks

        Returns:
            The loaded model object

        Raises:
            ValueError: If run not found or model not found
        """
        artifacts = self.get_run_artifacts(run_id)
        model_path = artifacts.get("artifact_model.pkl")

        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"Model artifact not found for run '{run_id}'")

        return load_model_safe(model_path, safe=safe)

    # -------- MODEL REGISTRY HELPERS --------
    def register_model(
        self,
        model: Any,
        metadata: Dict[str, Any],
        model_id: Optional[str] = None,
    ) -> str:
        """Register a model in the model registry.

        Args:
            model: The model object (will be saved)
            metadata: Metadata dict (should include: model_name, source_run_id,
            description, hyperparameters, etc.)
            model_id: Optional custom model ID; auto-generated if not provided

        Returns:
            The model ID
        """
        from mlvern.utils.artifact import save_model_safe

        if model_id is None:
            # Auto-generate model ID
            model_id = f"model_{datetime.now(timezone.utc).strftime(
                '%Y-%m-%d_%H-%M-%S')}"

        # Create model directory
        models_dir = os.path.join(self.mlvern_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        model_path = os.path.join(models_dir, f"{model_id}.pkl")

        # Save model with metadata
        save_model_safe(model, model_path, metadata=metadata)

        # Register in registry
        register_model_metadata(self.mlvern_dir, model_id, metadata)

        return model_id

    def list_models(self) -> Dict[str, Any]:
        """List all registered models."""
        return load_registry(self.mlvern_dir).get("models", {})

    def tag_run(self, run_id: str, tags: Dict[str, Any]) -> None:
        """Add or update tags on a run for searchability.

        Args:
            run_id: The run identifier
            tags: Dictionary of tags to add/update
        """
        tag_run(self.mlvern_dir, run_id, tags)

    def get_run_tags(self, run_id: str) -> Dict[str, Any]:
        """Get tags for a specific run.

        Args:
            run_id: The run identifier

        Returns:
            Dictionary of tags
        """
        return get_run_tags(self.mlvern_dir, run_id)

    # -------- DELETION & CLEANUP --------
    def remove_run(self, run_id: str, confirm: bool = False) -> bool:
        """Remove a run and its artifacts.

        Args:
            run_id: The run identifier
            confirm: Must be True to perform deletion (safety check)

        Returns:
            True if removal succeeded, False otherwise
        """
        if not confirm:
            print(f"⚠️  Run deletion skipped (confirm=False): {run_id}")
            return False

        run_path = os.path.join(self.mlvern_dir, "runs", run_id)

        if not os.path.exists(run_path):
            print(f"❌ Run '{run_id}' not found")
            return False

        # Remove from registry
        registry = load_registry(self.mlvern_dir)
        if "runs" in registry and run_id in registry["runs"]:
            del registry["runs"][run_id]
            save_registry(self.mlvern_dir, registry)

        # Remove directory
        return remove_directory_safe(run_path, confirm=True)

    def prune_datasets(self, older_than_days: int = 30,
                       confirm: bool = False) -> List[str]:
        """Remove datasets older than specified number of days.

        Args:
            older_than_days: Remove datasets older than this many days
            confirm: Must be True to perform deletion (safety check)

        Returns:
            List of removed dataset hashes
        """
        if not confirm:
            msg = "⚠️  Dataset pruning skipped (confirm=False)"
            print(msg)
            return []

        datasets_dir = os.path.join(self.mlvern_dir, "datasets")
        if not os.path.exists(datasets_dir):
            return []

        cutoff_time = datetime.now(
            timezone.utc) - timedelta(days=older_than_days)
        removed = []

        for dataset_hash in os.listdir(datasets_dir):
            dataset_path = os.path.join(datasets_dir, dataset_hash)
            if not os.path.isdir(dataset_path):
                continue

            created_time = get_directory_created_time(dataset_path)
            if created_time and created_time < cutoff_time:
                if remove_directory_safe(dataset_path, confirm=True):
                    # Remove from registry
                    registry = load_registry(self.mlvern_dir)
                    if "datasets" in registry and dataset_hash in registry["datasets"]:
                        del registry["datasets"][dataset_hash]
                        save_registry(self.mlvern_dir, registry)

                    removed.append(dataset_hash)
                    print(f"✓ Removed dataset: {dataset_hash}")

        return removed

    def get_project_stats(self) -> Dict[str, Any]:
        """Get overall project statistics.

        Returns:
            Dictionary with dataset count, run count, total size, etc.
        """
        datasets = self.list_datasets()
        runs = self.list_runs()
        models = self.list_models()

        datasets_dir = os.path.join(self.mlvern_dir, "datasets")
        runs_dir = os.path.join(self.mlvern_dir, "runs")
        models_dir = os.path.join(self.mlvern_dir, "models")

        if os.path.exists(datasets_dir):
            datasets_size = get_directory_size_mb(datasets_dir)
        else:
            datasets_size = 0
        if os.path.exists(runs_dir):
            runs_size = get_directory_size_mb(runs_dir)
        else:
            runs_size = 0
        if os.path.exists(models_dir):
            models_size = get_directory_size_mb(models_dir)
        else:
            models_size = 0

        return {
            "project": self.project,
            "datasets_count": len(datasets),
            "runs_count": len(runs),
            "models_count": len(models),
            "datasets_size_mb": round(datasets_size, 2),
            "runs_size_mb": round(runs_size, 2),
            "models_size_mb": round(models_size, 2),
            "total_size_mb": round(datasets_size + runs_size + models_size, 2),
        }

    # -------- EVALUATION & PREDICTION --------
    def predict(
        self,
        run_id_or_model: Any,
        X_test,
    ) -> Any:
        """Make predictions using a model from a run or passed model object.

        Args:
            run_id_or_model: Either a run_id (str) or a model object
            X_test: Test data for predictions

        Returns:
            Predictions array
        """
        if isinstance(run_id_or_model, str):
            model = self.load_model(run_id_or_model, safe=False)
        else:
            model = run_id_or_model

        return model.predict(X_test)

    def evaluate(
        self,
        run_id_or_model: Any,
        X_test,
        y_test,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a model and generate evaluation metrics and plots.

        Args:
            run_id_or_model: Either a run_id (str) or a model object
            X_test: Test features
            y_test: Test labels
            output_dir: Directory to save evaluation plots

        Returns:
            Dict with metrics and paths to generated plots
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        from mlvern.visual.auto_plot import auto_plot

        if isinstance(run_id_or_model, str):
            model = self.load_model(run_id_or_model, safe=False)
            run_id = run_id_or_model
        else:
            model = run_id_or_model
            run_id = None

        # Determine output directory
        if output_dir is None:
            if run_id:
                output_dir = os.path.join(
                    self.mlvern_dir, "runs", run_id, "evaluation"
                )
            else:
                output_dir = "./evaluation"

        os.makedirs(output_dir, exist_ok=True)

        # Make predictions
        y_pred = model.predict(X_test)

        # Try to get probability predictions
        y_prob = None
        try:
            if hasattr(model, "predict_proba"):
                y_proba_matrix = model.predict_proba(X_test)
                # For binary classification, use probability of positive class
                if y_proba_matrix.shape[1] == 2:
                    y_prob = y_proba_matrix[:, 1]
                else:
                    y_prob = y_proba_matrix.max(axis=1)
        except Exception:
            pass

        # Compute metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),
        }

        # Try to compute ROC-AUC
        try:
            if y_prob is not None:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_test, y_prob, multi_class="ovr", zero_division=0)
                )
        except Exception:
            pass

        # Generate plots
        plot_paths = {}
        try:
            auto_plot(
                "classification",
                y_test,
                y_pred,
                y_prob,
                output_dir,
            )

            # Collect generated plots
            for fname in os.listdir(output_dir):
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    plot_paths[fname] = os.path.join(
                        output_dir, fname
                    )
        except Exception as e:
            msg = f"Warning: Could not generate plots: {e}"
            print(msg)

        # Save evaluation report
        report = {
            "metrics": metrics,
            "plot_paths": plot_paths,
        }

        eval_report_path = os.path.join(
            output_dir, "evaluation_report.json"
        )
        with open(eval_report_path, "w") as f:
            json.dump(report, f, indent=4)

        report["plot_paths"] = plot_paths
        return report

    # -------- DATASET SAVING & LOADING --------
    def save_dataset(
        self,
        df: pd.DataFrame,
        dataset_hash: str,
        name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Save a dataframe to an existing dataset directory.

        Args:
            df: DataFrame to save
            dataset_hash: Hash of the dataset
            name: Optional friendly name for the dataset
            tags: Optional tags dict

        Returns:
            Dict with save info
        """
        from mlvern.utils.dataset_utils import (
            save_dataset_metadata,
            save_dataset_to_path,
        )

        dataset_path = self.get_dataset_path(dataset_hash)

        # Save actual data
        save_dataset_to_path(df, dataset_path)

        # Save metadata
        metadata = {
            "name": name or dataset_hash,
            "tags": tags or {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        }
        save_dataset_metadata(dataset_path, metadata)

        return {
            "dataset_hash": dataset_hash,
            "path": dataset_path,
            "saved": True,
            "metadata": metadata,
        }

    def load_dataset_by_hash(self, dataset_hash: str) -> pd.DataFrame:
        """Load a dataset from storage by hash.

        Args:
            dataset_hash: Hash of the dataset

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If dataset not found
        """
        from mlvern.utils.dataset_utils import load_dataset_from_path

        try:
            dataset_path = self.get_dataset_path(dataset_hash)
        except ValueError:
            raise FileNotFoundError(f"Dataset '{dataset_hash}' not found")

        return load_dataset_from_path(dataset_path)

    def get_dataset_report(self, dataset_hash: str) -> Dict[str, Any]:
        """Get aggregated report for a dataset.

        Includes inspection, statistics, risk, and EDA reports.

        Args:
            dataset_hash: Hash of the dataset

        Returns:
            Aggregated report dict
        """
        from mlvern.utils.dataset_utils import get_dataset_report

        dataset_path = self.get_dataset_path(dataset_hash)
        report = get_dataset_report(dataset_path)

        # Also include metadata if available
        from mlvern.utils.dataset_utils import load_dataset_metadata
        metadata = load_dataset_metadata(dataset_path)

        return {
            "dataset_hash": dataset_hash,
            "metadata": metadata,
            **report,
        }

    # -------- TRAIN + RUN --------
    def run(
        self,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        config: dict,
        dataset_fp,
    ):
        """Train a model and create a run record."""
        from mlvern.utils.environment import save_environment

        model, metrics = train_model(
            model, X_train, y_train, X_val, y_val
        )

        run_id = create_run(
            self.mlvern_dir,
            dataset_fp,
            model,
            metrics,
            config,
        )

        # Capture environment information
        run_path = os.path.join(self.mlvern_dir, "runs", run_id)
        save_environment(run_path)

        return run_id, metrics
