"""Comprehensive script that calls every `Forge` API method in-order.

This script is intended as a clear, linear demonstration for documentation
purposes. It calls each method on `mlvern.core.forge.Forge` in the order
they are defined in the class and prints short, human-readable outputs.

Note: the script is defensive (uses try/except) so it can be run repeatedly
without causing destructive changes when methods require confirmation.
"""

from pathlib import Path
import pprint
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlvern.core.forge import Forge


pp = pprint.PrettyPrinter(indent=2)


def safe_call(fn, *args, desc=None, **kwargs):
    try:
        result = fn(*args, **kwargs)
        print(f"   OK: {desc or fn.__name__}")
        return result
    except Exception as e:
        print(f"   ERROR calling {desc or fn.__name__}: {e}")
        return None


def main():
    print("Calling Forge APIs in the exact order they appear in `forge.py`")

    base_dir = str(Path(__file__).parent)

    # 1) __init__ (constructor)
    print("\n1) __init__")
    forge = Forge(project="iris_docs_example", base_dir=base_dir)
    print(f"   Created Forge for project: {forge.project}")

    # 2) init()
    print("\n2) init() -> initializing project structure")
    safe_call(forge.init, desc="forge.init()")

    # Prepare data used by many examples
    iris = load_iris(as_frame=True)
    df = iris.frame
    target = "target"
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3) register_dataset()
    print("\n3) register_dataset(df, target)")
    dataset_fp = safe_call(forge.register_dataset, df, target, desc="register_dataset")
    pp.pprint({"dataset_fp": dataset_fp})

    # 4) list_datasets()
    print("\n4) list_datasets()")
    datasets = safe_call(forge.list_datasets, desc="list_datasets")
    pp.pprint({"datasets_count": len(datasets) if datasets is not None else None})

    # If dataset_fp is a mapping, try to locate dataset_hash
    dataset_hash = None
    if isinstance(dataset_fp, dict):
        dataset_hash = dataset_fp.get("dataset_hash") or dataset_fp.get("hash")

    # 5) get_dataset_path()
    print("\n5) get_dataset_path(dataset_hash)")
    if dataset_hash:
        ds_path = safe_call(forge.get_dataset_path, dataset_hash, desc="get_dataset_path")
        print(f"   dataset_path: {ds_path}")
    else:
        print("   Skipping get_dataset_path: no dataset_hash available")

    # 6) load_dataset()
    print("\n6) load_dataset(dataset_hash)")
    if dataset_hash:
        ds_info = safe_call(forge.load_dataset, dataset_hash, desc="load_dataset")
        pp.pprint({"loaded_dataset_keys": list(ds_info.keys()) if isinstance(ds_info, dict) else None})
    else:
        print("   Skipping load_dataset: no dataset_hash available")

    # Create an initial run so later run-related accessors have something to work with
    print("\n-- Creating an initial quick run for subsequent examples --")
    quick_model = LogisticRegression(max_iter=200)
    quick_config = {"model_type": "LogisticRegression", "max_iter": 200}
    run_setup = safe_call(forge.run, quick_model, X_train, y_train, X_val, y_val, quick_config, dataset_fp, desc="initial run()")
    run_id = None
    if isinstance(run_setup, tuple) and len(run_setup) >= 1:
        run_id = run_setup[0]

    # 7) list_runs()
    print("\n7) list_runs()")
    runs = safe_call(forge.list_runs, desc="list_runs")
    pp.pprint({"runs_count": len(runs) if runs is not None else None})

    # 8) get_run(run_id)
    print("\n8) get_run(run_id)")
    if run_id:
        run_info = safe_call(forge.get_run, run_id, desc="get_run")
        pp.pprint({"run_keys": list(run_info.keys()) if isinstance(run_info, dict) else None})
    else:
        print("   Skipping get_run: no run_id available")

    # 9) get_run_metrics(run_id)
    print("\n9) get_run_metrics(run_id)")
    if run_id:
        metrics = safe_call(forge.get_run_metrics, run_id, desc="get_run_metrics")
        pp.pprint({"metrics": metrics})
    else:
        print("   Skipping get_run_metrics: no run_id available")

    # 10) get_run_artifacts(run_id)
    print("\n10) get_run_artifacts(run_id)")
    if run_id:
        artifacts = safe_call(forge.get_run_artifacts, run_id, desc="get_run_artifacts")
        pp.pprint({"artifacts": artifacts})
    else:
        print("   Skipping get_run_artifacts: no run_id available")

    # 11) load_model(run_id)
    print("\n11) load_model(run_id)")
    model_obj = None
    if run_id:
        model_obj = safe_call(forge.load_model, run_id, safe=False, desc="load_model")
        print(f"   Loaded model type: {type(model_obj).__name__ if model_obj is not None else None}")
    else:
        print("   Skipping load_model: no run_id available")

    # 12) register_model(model, metadata)
    print("\n12) register_model(model, metadata)")
    model_id = None
    if model_obj is not None:
        metadata = {"model_name": "initial_lr", "source_run_id": run_id}
        model_id = safe_call(forge.register_model, model_obj, metadata, None, desc="register_model")
        print(f"   Registered model_id: {model_id}")
    else:
        print("   Skipping register_model: no model object available")

    # 13) list_models()
    print("\n13) list_models()")
    models = safe_call(forge.list_models, desc="list_models")
    pp.pprint({"models": models})

    # 14) tag_run(run_id, tags)
    print("\n14) tag_run(run_id, tags)")
    if run_id:
        safe_call(forge.tag_run, run_id, {"stage": "test", "reviewed": False}, desc="tag_run")
        print("   Tagged run (added 'stage':'test')")
    else:
        print("   Skipping tag_run: no run_id available")

    # 15) get_run_tags(run_id)
    print("\n15) get_run_tags(run_id)")
    if run_id:
        tags = safe_call(forge.get_run_tags, run_id, desc="get_run_tags")
        pp.pprint({"tags": tags})
    else:
        print("   Skipping get_run_tags: no run_id available")

    # 16) remove_run(run_id, confirm)
    print("\n16) remove_run(run_id, confirm=False) -> safe check (won't delete)")
    if run_id:
        removed = safe_call(forge.remove_run, run_id, False, desc="remove_run (confirm=False)")
        print(f"   remove_run returned: {removed}")
    else:
        print("   Skipping remove_run: no run_id available")

    # 17) prune_datasets(older_than_days, confirm)
    print("\n17) prune_datasets(older_than_days=30, confirm=False) -> safe check")
    pruned = safe_call(forge.prune_datasets, 30, False, desc="prune_datasets (confirm=False)")
    print(f"   pruned list: {pruned}")

    # 18) get_project_stats()
    print("\n18) get_project_stats()")
    stats = safe_call(forge.get_project_stats, desc="get_project_stats")
    pp.pprint({"project_stats": stats})

    # 19) predict(run_id_or_model, X_test)
    print("\n19) predict(run_id_or_model, X_test)")
    if model_obj is not None:
        preds = safe_call(forge.predict, model_obj, X_val, desc="predict (model object)")
        print(f"   Predictions length: {len(preds) if preds is not None else None}")
    elif run_id:
        preds = safe_call(forge.predict, run_id, X_val, desc="predict (run_id)")
        print(f"   Predictions length: {len(preds) if preds is not None else None}")
    else:
        print("   Skipping predict: no model or run available")

    # 20) evaluate(run_id_or_model, X_test, y_test, output_dir=None)
    print("\n20) evaluate(run_id_or_model, X_test, y_test)")
    if model_obj is not None:
        eval_report = safe_call(forge.evaluate, model_obj, X_val, y_val, None, desc="evaluate (model)")
        pp.pprint({"evaluation_metrics": eval_report.get("metrics") if isinstance(eval_report, dict) else None})
    elif run_id:
        eval_report = safe_call(forge.evaluate, run_id, X_val, y_val, None, desc="evaluate (run_id)")
        pp.pprint({"evaluation_metrics": eval_report.get("metrics") if isinstance(eval_report, dict) else None})
    else:
        print("   Skipping evaluate: no model or run available")

    # 21) save_dataset(df, dataset_hash, name=None, tags=None)
    print("\n21) save_dataset(df, dataset_hash, name='iris', tags={})")
    if dataset_hash:
        saved = safe_call(forge.save_dataset, df, dataset_hash, "iris", {"source": "sklearn"}, desc="save_dataset")
        pp.pprint({"save_result": saved})
    else:
        print("   Skipping save_dataset: no dataset_hash available")

    # 22) load_dataset_by_hash(dataset_hash)
    print("\n22) load_dataset_by_hash(dataset_hash)")
    if dataset_hash:
        loaded_df = safe_call(forge.load_dataset_by_hash, dataset_hash, desc="load_dataset_by_hash")
        print(f"   Loaded DataFrame shape: {getattr(loaded_df, 'shape', None)}")
    else:
        print("   Skipping load_dataset_by_hash: no dataset_hash available")

    # 23) get_dataset_report(dataset_hash)
    print("\n23) get_dataset_report(dataset_hash)")
    if dataset_hash:
        ds_report = safe_call(forge.get_dataset_report, dataset_hash, desc="get_dataset_report")
        pp.pprint({"dataset_report_keys": list(ds_report.keys()) if isinstance(ds_report, dict) else None})
    else:
        print("   Skipping get_dataset_report: no dataset_hash available")

    # 24) run(model, X_train, y_train, X_val, y_val, config, dataset_fp)
    print("\n24) run(...) -> train a second model to demonstrate run creation")
    rf_model = LogisticRegression(max_iter=300)
    rf_config = {"model_type": "LogisticRegression", "max_iter": 300}
    run_second = safe_call(forge.run, rf_model, X_train, y_train, X_val, y_val, rf_config, dataset_fp, desc="run (second)")
    pp.pprint({"second_run": run_second})

    print("\nDone. All Forge methods were called in order (errors printed if any).")


if __name__ == "__main__":
    main()
