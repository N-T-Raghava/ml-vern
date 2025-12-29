import json
import os
from datetime import datetime, timezone


def _registry_path(mlvern_dir):
    return os.path.join(mlvern_dir, "registry.json")


def load_registry(mlvern_dir):
    path = _registry_path(mlvern_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_registry(mlvern_dir, registry):
    path = _registry_path(mlvern_dir)
    with open(path, "w") as f:
        json.dump(registry, f, indent=4)


def init_registry(mlvern_dir, project_name):
    registry = {
        "project": project_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {},
        "runs": {},
    }
    save_registry(mlvern_dir, registry)
