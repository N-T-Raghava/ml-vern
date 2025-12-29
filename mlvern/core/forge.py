import os

from mlvern.data.register import register_dataset
from mlvern.train.trainer import train_model
from mlvern.utils.registry import init_registry
from mlvern.version.run_manager import create_run


class Forge:
    def __init__(self, project: str, base_dir: str = "."):
        self.project = project
        self.base_dir = os.path.abspath(base_dir)
        self.mlvern_dir = os.path.join(
            self.base_dir, f".mlvern_{project}"
        )

    def init(self):
        os.makedirs(self.mlvern_dir, exist_ok=True)
        for d in ["datasets", "runs", "models"]:
            os.makedirs(os.path.join(self.mlvern_dir, d), exist_ok=True)

        registry_path = os.path.join(self.mlvern_dir, "registry.json")
        if not os.path.exists(registry_path):
            init_registry(self.mlvern_dir, self.project)

    # -------- DATASET --------
    def register_dataset(self, df, target: str):
        return register_dataset(df, target, self.mlvern_dir)

    def list_datasets(self):
        from mlvern.utils.registry import load_registry
        return load_registry(self.mlvern_dir).get("datasets", {})

    def list_runs(self):
        from mlvern.utils.registry import load_registry
        return load_registry(self.mlvern_dir).get("runs", {})

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

        return run_id, metrics
