import json
import pandas as pd

def inspect_data(df: pd.DataFrame, target: str, mlvern_dir: str):
    report = {}
    report["shape"] = df.shape
    report["missing_values"] = df.isnull().sum().to_dict()
    report["duplicates"] = int(df.duplicated().sum())

    if target in df.columns:
        report["class_distribution"] = df[target].value_counts().to_dict()
    else:
        report["error"] = f"Target column '{target}' not found"

    path = f"{mlvern_dir}/reports/data_inspection.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return report
