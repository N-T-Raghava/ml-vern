import json
import os
from datetime import datetime, timezone

import pandas as pd


def inspect_data(df: pd.DataFrame, target: str, mlvern_dir: str):
    if df.empty:
        raise ValueError("Dataset is empty")

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "library": "mlvern",
            "version": "0.1.0",
        },
        "dataset_summary": {},
        "statistics": {},
        "target_analysis": {},
        "vulnerabilities": [],
        "recommendations": [],
    }

    report["dataset_summary"] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
    }

    missing = df.isnull().sum()
    duplicates = int(df.duplicated().sum())

    report["statistics"] = {
        "missing_values": missing[missing > 0].to_dict(),
        "duplicates": duplicates,
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

    if missing.any():
        report["vulnerabilities"].append({
            "severity": "WARNING",
            "type": "MISSING_VALUES",
            "message": f"{int(missing.sum())} missing values detected",
        })
        report["recommendations"].append(
            "Consider imputing missing values"
        )

    if duplicates > 0:
        report["vulnerabilities"].append({
            "severity": "WARNING",
            "type": "DUPLICATES",
            "message": f"{duplicates} duplicate rows detected",
        })
        report["recommendations"].append(
            "Consider removing duplicate rows"
        )

    if target not in df.columns:
        report["vulnerabilities"].append({
            "severity": "CRITICAL",
            "type": "TARGET_MISSING",
            "message": f"Target column '{target}' not found",
        })
    else:
        class_dist = df[target].value_counts().to_dict()
        max_class = max(class_dist.values())
        min_class = min(class_dist.values())
        imbalance_ratio = round(max_class / max(min_class, 1), 2)

        report["target_analysis"] = {
            "target": target,
            "class_distribution": class_dist,
            "imbalance_ratio": imbalance_ratio,
        }

        if imbalance_ratio > 3:
            report["vulnerabilities"].append({
                "severity": "WARNING",
                "type": "CLASS_IMBALANCE",
                "message": f"Imbalance ratio is {imbalance_ratio}",
            })
            report["recommendations"].append(
                "Consider class weighting or resampling techniques"
            )

    reports_dir = os.path.join(mlvern_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    path = os.path.join(reports_dir, "data_validation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)

    return report
