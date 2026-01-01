"""
Artifact management utilities for safe model loading/saving and run management.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib


def save_model_safe(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model with metadata to disk (joblib format).

    Args:
        model: The model object to save
        filepath: Path where the model will be saved
        metadata: Optional metadata dict to save alongside the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

    # Save metadata if provided
    if metadata:
        metadata_path = filepath.replace(".pkl", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)


def load_model_safe(filepath: str, safe: bool = True) -> Any:
    """Load model from disk with optional safety warnings.

    Args:
        filepath: Path to the model file
        safe: If True, warns about security risks of unpickling
              untrusted data

    Returns:
        The loaded model object

    Raises:
        FileNotFoundError: If model file does not exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    if safe:
        msg = f"⚠️  Warning: Loading pickled model from {filepath}"
        print(msg)
        msg2 = (
            "   Only load models from trusted sources. "
            "Pickles can execute arbitrary code."
        )
        print(msg2)

    return joblib.load(filepath)


def get_model_metadata(filepath: str) -> Dict[str, Any]:
    """Load model metadata if it exists.

    Args:
        filepath: Path to the model file

    Returns:
        Metadata dict, or empty dict if no metadata file exists
    """
    metadata_path = filepath.replace(".pkl", "_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}


def remove_directory_safe(path: str, confirm: bool = False) -> bool:
    """Safely remove a directory tree.

    Args:
        path: Directory path to remove
        confirm: If False, returns False without removing (for safety)

    Returns:
        True if removal succeeded, False otherwise
    """
    if not os.path.isdir(path):
        return False

    if not confirm:
        msg = f"⚠️  Directory deletion skipped (confirm=False): {path}"
        print(msg)
        return False

    try:
        shutil.rmtree(path)
        return True
    except Exception as e:
        msg = f"❌ Error removing directory {path}: {e}"
        print(msg)
        return False


def get_directory_size_mb(path: str) -> float:
    """Calculate total size of a directory in MB.

    Args:
        path: Directory path

    Returns:
        Size in MB
    """
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(path)
        for filename in filenames
    )
    return total_size / (1024 * 1024)


def get_directory_created_time(path: str) -> Optional[datetime]:
    """Get the creation/modification time of a directory.

    Args:
        path: Directory path

    Returns:
        datetime object if path exists, None otherwise
    """
    if not os.path.exists(path):
        return None

    stat = os.stat(path)
    # Use modification time as proxy for creation time
    return datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
