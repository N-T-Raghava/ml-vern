import json
import os
from datetime import datetime, timezone

from mlvern.data.fingerprint import fingerprint_dataset
from mlvern.data.inspect import inspect_data
from mlvern.data.risk_check import run_risk_checks
from mlvern.data.statistics import compute_statistics
from mlvern.utils.registry import load_registry, save_registry


def register_dataset(df, target, mlvern_dir):
    datasets_dir = os.path.join(mlvern_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    fp = fingerprint_dataset(df, target)
    ds_hash = fp["dataset_hash"]

    dataset_path = os.path.join(datasets_dir, ds_hash)

    if os.path.exists(dataset_path):
        return fp, False  # already analyzed

    os.makedirs(dataset_path)

    # ---- Heavy checks (ONCE) ----
    inspect_data(df, target, dataset_path)
    compute_statistics(df, target, dataset_path)
    run_risk_checks(df, target=target, mlvern_dir=dataset_path)

    # ---- Save schema ----
    with open(os.path.join(dataset_path, "schema.json"), "w") as f:
        json.dump(fp["schema"], f, indent=4)

    # ---- Registry update ----
    registry = load_registry(mlvern_dir)
    registry.setdefault("datasets", {})

    registry.setdefault("datasets", {})[ds_hash] = {
        "rows": fp["rows"],
        "columns": fp["columns"],
        "target": target,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    save_registry(mlvern_dir, registry)

    return fp, True
