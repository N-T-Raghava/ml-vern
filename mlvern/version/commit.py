import json
import os
from datetime import datetime

import joblib

from mlvern.utils.hashing import hash_object


def _next_commit_id(commits_dir):
    os.makedirs(commits_dir, exist_ok=True)
    existing = sorted(os.listdir(commits_dir))
    return f"{len(existing)+1:04d}"


def commit_run(mlvern_dir, message, model, metrics, params):
    commits_dir = os.path.join(mlvern_dir, "commits")
    os.makedirs(commits_dir, exist_ok=True)
    cid = _next_commit_id(commits_dir)

    commit_path = os.path.join(commits_dir, cid)
    os.makedirs(commit_path)

    joblib.dump(model, f"{commit_path}/model.joblib")

    with open(f"{commit_path}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(f"{commit_path}/params.json", "w") as f:
        json.dump(params, f, indent=4)

    meta = {
        "id": cid,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        "params_hash": hash_object(params),
        "metrics_hash": hash_object(metrics),
    }

    with open(f"{commit_path}/meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    return cid


def log_commits(mlvern_dir):
    commits_dir = os.path.join(mlvern_dir, "commits")
    logs = []

    for cid in sorted(os.listdir(commits_dir)):
        with open(f"{commits_dir}/{cid}/meta.json") as f:
            logs.append(json.load(f))

    return logs
