import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


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
        "models": {},
    }
    save_registry(mlvern_dir, registry)


# -------- Model registry operations --------
def register_model_metadata(
    mlvern_dir: str,
    model_id: str,
    metadata: Dict[str, Any],
) -> None:
    """Register model metadata in the registry.

    Args:
        mlvern_dir: Path to mlvern directory
        model_id: Unique model identifier
        metadata: Model metadata dict (class, hyperparameters,
                  source run, etc.)
    """
    registry = load_registry(mlvern_dir)
    registry.setdefault("models", {})[model_id] = {
        **metadata,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    save_registry(mlvern_dir, registry)


def list_models_registry(mlvern_dir: str) -> Dict[str, Any]:
    """List all registered models.

    Args:
        mlvern_dir: Path to mlvern directory

    Returns:
        Dict of model_id -> metadata
    """
    registry = load_registry(mlvern_dir)
    return registry.get("models", {})


# -------- Run tagging operations --------
def tag_run(
    mlvern_dir: str,
    run_id: str,
    tags: Dict[str, Any],
) -> None:
    """Add or update tags on a run.

    Args:
        mlvern_dir: Path to mlvern directory
        run_id: Run identifier
        tags: Tags dict to merge with existing tags
    """
    registry = load_registry(mlvern_dir)
    if "runs" not in registry or run_id not in registry["runs"]:
        raise KeyError(f"Run '{run_id}' not found in registry")

    registry["runs"][run_id].setdefault("tags", {}).update(tags)
    save_registry(mlvern_dir, registry)


def get_run_tags(mlvern_dir: str, run_id: str) -> Dict[str, Any]:
    """Get tags for a specific run.

    Args:
        mlvern_dir: Path to mlvern directory
        run_id: Run identifier

    Returns:
        Dict of tags, or empty dict if none exist
    """
    registry = load_registry(mlvern_dir)
    if "runs" not in registry or run_id not in registry["runs"]:
        return {}
    return registry["runs"][run_id].get("tags", {})


def search_runs_by_tag(
    mlvern_dir: str,
    tag_key: str,
    tag_value: Any,
) -> list:
    """Search runs by tag key-value pair.

    Args:
        mlvern_dir: Path to mlvern directory
        tag_key: Tag key to search for
        tag_value: Tag value to match

    Returns:
        List of run IDs matching the tag
    """
    registry = load_registry(mlvern_dir)
    matching_runs = []
    for run_id, run_data in registry.get("runs", {}).items():
        tags = run_data.get("tags", {})
        if tags.get(tag_key) == tag_value:
            matching_runs.append(run_id)
    return matching_runs
