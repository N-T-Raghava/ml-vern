import pandas as pd

from mlvern.utils.hashing import hash_object


def fingerprint_dataset(df: pd.DataFrame, target: str):
    """
    FAST fingerprinting – runs every time.
    """
    schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target": target,
    }

    content_hash = hash_object({
        "data": df.to_dict(orient="list"),
        "schema": schema,
    })

    return {
        "dataset_hash": content_hash[:12],  # short hash for paths
        "rows": df.shape[0],
        "columns": df.shape[1],
        "schema": schema,
    }
