"""
Dataset save/load utilities for efficient storage and retrieval.
"""

import json
import os
import pickle
from typing import Any, Dict

import pandas as pd


def save_dataset_to_path(df: pd.DataFrame, dataset_path: str) -> None:
    """Save a dataframe to the data subdirectory of a dataset path.

    Args:
        df: DataFrame to save
        dataset_path: Root path of the dataset directory
    """
    data_dir = os.path.join(dataset_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save as pickle to preserve dtypes without external dependencies
    pickle_path = os.path.join(data_dir, "dataset.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)


def load_dataset_from_path(dataset_path: str) -> pd.DataFrame:
    """Load a dataframe from the data subdirectory of a dataset path.

    Args:
        dataset_path: Root path of the dataset directory

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If dataset file not found
    """
    pickle_path = os.path.join(dataset_path, "data", "dataset.pkl")

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Dataset not found at {pickle_path}")

    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def save_dataset_metadata(
    dataset_path: str,
    metadata: Dict[str, Any],
) -> None:
    """Save dataset metadata (name, tags, etc.) to data subdirectory.

    Args:
        dataset_path: Root path of the dataset directory
        metadata: Metadata dict
    """
    data_dir = os.path.join(dataset_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def load_dataset_metadata(dataset_path: str) -> Dict[str, Any]:
    """Load dataset metadata from data subdirectory.

    Args:
        dataset_path: Root path of the dataset directory

    Returns:
        Metadata dict, or empty dict if file doesn't exist
    """
    metadata_path = os.path.join(dataset_path, "data", "metadata.json")

    if not os.path.exists(metadata_path):
        return {}

    with open(metadata_path, "r") as f:
        return json.load(f)


def get_dataset_report(dataset_path: str) -> Dict[str, Any]:
    """Aggregate all reports for a dataset.

    Collects inspection, statistics, and risk reports from the
    dataset directory.

    Args:
        dataset_path: Root path of the dataset directory

    Returns:
        Aggregated report dict
    """
    reports = {
        "dataset_path": dataset_path,
        "inspection": None,
        "statistics": None,
        "risk": None,
        "eda": None,
    }

    reports_dir = os.path.join(dataset_path, "reports")

    if not os.path.exists(reports_dir):
        return reports

    # Load inspection report
    inspection_path = os.path.join(reports_dir, "data_inspection_report.json")
    if os.path.exists(inspection_path):
        with open(inspection_path, "r") as f:
            reports["inspection"] = json.load(f)

    # Load statistics report
    statistics_path = os.path.join(reports_dir, "statistics_report.json")
    if os.path.exists(statistics_path):
        with open(statistics_path, "r") as f:
            reports["statistics"] = json.load(f)

    # Load risk report
    risk_path = os.path.join(reports_dir, "risk_report.json")
    if os.path.exists(risk_path):
        with open(risk_path, "r") as f:
            reports["risk"] = json.load(f)

    # Load EDA report
    eda_path = os.path.join(reports_dir, "eda_report.json")
    if os.path.exists(eda_path):
        with open(eda_path, "r") as f:
            reports["eda"] = json.load(f)

    return reports
    return reports
