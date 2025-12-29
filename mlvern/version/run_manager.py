import json
import os
from datetime import datetime, timezone

import joblib

from mlvern.utils.registry import load_registry, save_registry


def create_run(
    mlvern_dir,
    dataset_fp,
    model,
    metrics,
    config,
    artifacts=None,
):
    runs_dir = os.path.join(mlvern_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S-%f')}"
    run_path = os.path.join(runs_dir, run_id)
    os.makedirs(run_path, exist_ok=True)

    # ---- Metadata ----
    with open(os.path.join(run_path, "meta.json"), "w") as f:
        json.dump(
            {
                "run_id": run_id,
                "dataset_hash": dataset_fp["dataset_hash"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=4,
        )

    # ---- Config ----
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ---- Metrics ----
    with open(os.path.join(run_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ---- Model ----
    artifacts_dir = os.path.join(run_path, "artifacts")
    os.makedirs(artifacts_dir)
    joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))

    # ---- Registry update ----
    registry = load_registry(mlvern_dir)
    registry.setdefault("runs", {})

    registry.setdefault("runs", {})[run_id] = {
        "dataset_hash": dataset_fp["dataset_hash"],
        "model": model.__class__.__name__,
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path": f"runs/{run_id}",
    }

    save_registry(mlvern_dir, registry)

    return run_id
