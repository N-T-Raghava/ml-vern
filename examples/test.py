"""
Comprehensive example demonstrating mlvern functionality.

Covers:
- Forge project initialization
- Dataset fingerprinting and registration
- Model training and versioning
- Run tracking and comparison
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlvern.core.forge import Forge


def main():
    print("=" * 70)
    print("MLVern Example: Complete ML Project Workflow")
    print("=" * 70)

    # Get examples directory as base
    base_dir = str(Path(__file__).parent)

    # ===== 1. INITIALIZE PROJECT =====
    print("\n1. Initializing MLVern Project...")
    forge = Forge(project="iris_classification", base_dir=base_dir)
    forge.init()
    print(f"   ✓ Project initialized at: {forge.mlvern_dir}")

    # ===== 2. LOAD AND PREPARE DATA =====
    print("\n2. Loading and Preparing Dataset...")
    data = load_iris(as_frame=True)
    df = data.frame
    target = "target"

    print(f"   Dataset shape: {df.shape}")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Target: {target}")

    # ===== 3. REGISTER DATASET =====
    print("\n3. Registering Dataset...")
    dataset_fp, is_new = forge.register_dataset(df, target)
    print(f"   Dataset hash: {dataset_fp['dataset_hash']}")
    print(f"   Rows: {dataset_fp['rows']}, Columns: {dataset_fp['columns']}")
    print(f"   New registration: {is_new}")

    # List registered datasets
    datasets = forge.list_datasets()
    print(f"   Total registered datasets: {len(datasets)}")

    # ===== 4. PREPARE TRAIN/VAL SPLIT =====
    print("\n4. Preparing Train/Validation Split...")
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")

    # ===== 5. TRAIN FIRST MODEL (Logistic Regression) =====
    print("\n5. Training Model 1: Logistic Regression...")
    lr_model = LogisticRegression(max_iter=200, random_state=42)

    config_lr = {
        "model_type": "LogisticRegression",
        "max_iter": 200,
        "solver": "lbfgs",
        "random_state": 42,
    }

    run_id_1, metrics_1 = forge.run(
        lr_model, X_train, y_train, X_val, y_val, config_lr, dataset_fp
    )

    print(f"   Run ID: {run_id_1}")
    print(f"   Accuracy: {metrics_1['accuracy']:.4f}")

    # ===== 6. TRAIN SECOND MODEL (Random Forest) =====
    print("\n6. Training Model 2: Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

    config_rf = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": 42,
    }

    run_id_2, metrics_2 = forge.run(
        rf_model, X_train, y_train, X_val, y_val, config_rf, dataset_fp
    )

    print(f"   Run ID: {run_id_2}")
    print(f"   Accuracy: {metrics_2['accuracy']:.4f}")

    # ===== 7. LIST AND COMPARE RUNS =====
    print("\n7. Run Tracking and Comparison...")
    runs = forge.list_runs()
    print(f"   Total runs: {len(runs)}")

    for run_id, run_info in runs.items():
        print(f"\n   Run: {run_id}")
        print(f"     Model: {run_info['model']}")
        print(f"     Accuracy: {run_info['metrics']['accuracy']:.4f}")
        print(f"     Created: {run_info['created_at'][:19]}")

    # ===== 8. DEMONSTRATE DATASET CACHING =====
    print("\n8. Testing Dataset Caching...")
    dataset_fp_2, is_new_2 = forge.register_dataset(df, target)
    print(f"   Same dataset registered again: {dataset_fp['dataset_hash']}")
    print(f"   Hashes match: {dataset_fp['dataset_hash'] == dataset_fp_2['dataset_hash']}")
    print(f"   New registration required: {is_new_2}")

    # ===== 9. SYNTHETIC DATA - DIFFERENT DATASET =====
    print("\n9. Registering Different Dataset...")
    np.random.seed(42)
    df_synthetic = pd.DataFrame({
        "feature1": np.random.randn(100),
        "feature2": np.random.randn(100),
        "feature3": np.random.randn(100),
        "target": np.random.randint(0, 3, 100),
    })

    dataset_fp_3, is_new_3 = forge.register_dataset(
        df_synthetic, "target"
    )
    print(f"   Synthetic dataset hash: {dataset_fp_3['dataset_hash']}")
    print(f"   Different from iris: {dataset_fp['dataset_hash'] != dataset_fp_3['dataset_hash']}")

    datasets_final = forge.list_datasets()
    print(f"   Total registered datasets: {len(datasets_final)}")

    # ===== 10. SUMMARY =====
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Project: iris_classification")
    print(f"Location: {forge.mlvern_dir}")
    print(f"Registered datasets: {len(datasets_final)}")
    print(f"Training runs: {len(runs)}")
    print(f"\nBest performer:")
    best_run = max(runs.items(), key=lambda x: x[1]["metrics"]["accuracy"])
    print(f"  Model: {best_run[1]['model']}")
    print(f"  Accuracy: {best_run[1]['metrics']['accuracy']:.4f}")
    print(f"  Run ID: {best_run[0]}")
    print("\n✓ All mlvern features demonstrated successfully!")


if __name__ == "__main__":
    main()
