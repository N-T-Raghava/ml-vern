import json
import os

import joblib


def checkout_commit(mlvern_dir, commit_id):
    commit_path = os.path.join(mlvern_dir, "commits", commit_id)

    if not os.path.exists(commit_path):
        raise ValueError(f"Commit {commit_id} does not exist")

    model = joblib.load(os.path.join(commit_path, "model.joblib"))

    with open(os.path.join(commit_path, "metrics.json")) as f:
        metrics = json.load(f)

    with open(os.path.join(commit_path, "params.json")) as f:
        params = json.load(f)

    return model, metrics, params
