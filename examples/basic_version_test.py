import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from mlvern import Forge


def main():
    iris = load_iris(as_frame=True)
    df = iris.frame

    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    forge = Forge(project="iris_example")

    print("🔍 Inspecting data...")
    inspection_report = forge.inspect(df, target="target")
    print(inspection_report)

    print("📊 Running EDA...")
    forge.eda(df)

    print("🤖 Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    trained_model, metrics = forge.train(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("Validation metrics:", metrics)

    y_pred = trained_model.predict(X_val)
    y_prob = trained_model.predict_proba(X_val)[:, 1] if hasattr(trained_model, "predict_proba") else None

    forge.plot(
        task="classification",
        y_true=y_val,
        y_pred=y_pred,
        y_prob=y_prob
    )

    print("📦 Committing experiment...")
    commit_id = forge.commit(
        message="random forest baseline on iris",
        model=trained_model,
        metrics=metrics,
        params={
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42
        }
    )

    print(f"✅ Commit created: {commit_id}")

    print("🔁 Checking out model...")
    restored_model, restored_metrics, restored_params = forge.checkout(commit_id)

    acc = accuracy_score(y_val, restored_model.predict(X_val))
    print("Restored model accuracy:", acc)
    print("Restored params:", restored_params)

if __name__ == "__main__":
    main()
