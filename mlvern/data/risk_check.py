import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.metrics import mutual_info_score
except Exception:
    mutual_info_score = None


def class_imbalance(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}
    counts = df[target].value_counts().to_dict()
    max_c = max(counts.values()) if counts else 0
    min_c = min(counts.values()) if counts else 0
    imbalance_ratio = float(max_c / max(min_c, 1))
    return {"counts": counts, "imbalance_ratio": imbalance_ratio}


def sensitive_attribute_imbalance(df: pd.DataFrame, sensitive_cols: List[str]) -> Dict[str, Any]:
    result = {}
    for col in sensitive_cols:
        if col not in df.columns:
            result[col] = {"error": "not_present"}
            continue
        result[col] = df[col].value_counts(normalize=True).to_dict()
    return result


def sampling_bias(baseline: pd.DataFrame, current: pd.DataFrame,
                  cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compare categorical distributions between baseline and current using chi-squared test."""
    if cols is None:
        cols = baseline.columns.intersection(current.columns).tolist()
    report = {}
    for col in cols:
        if pd.api.types.is_numeric_dtype(baseline[col]):
            # bin numeric into quartiles
            bins = np.quantile(np.concatenate([baseline[col].dropna(), current[col].dropna()]),
                               [0, 0.25, 0.5, 0.75, 1.0])
            b_cat = pd.cut(baseline[col], bins=bins, include_lowest=True)
            c_cat = pd.cut(current[col], bins=bins, include_lowest=True)
            tbl = pd.crosstab(b_cat, c_cat)
        else:
            tbl = pd.crosstab(baseline[col].fillna("__NA__"), current[col].fillna("__NA__"))
        try:
            chi2, p, _, _ = stats.chi2_contingency(tbl)
            report[col] = {"chi2": float(chi2), "pvalue": float(p)}
        except Exception:
            report[col] = cast(Any, {"error": "chi2_failed"})
    return report


def target_leakage_detection(df: pd.DataFrame, target: str,
                             threshold: float = 0.99) -> Dict[str, Any]:
    out = {}
    if target not in df.columns:
        return {"error": f"Target '{target}' not found"}
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        if col == target:
            continue
        corr = numeric[col].corr(numeric[target])
        mi = None
        if mutual_info_score is not None:
            try:
                mi = mutual_info_score(pd.cut(numeric[col], bins=10).astype(str),
                                       pd.cut(numeric[target], bins=10).astype(str))
            except Exception:
                mi = None
        if pd.isna(corr):
            corr = 0.0
        if abs(corr) >= threshold:
            out[col] = {"correlation": float(corr), "mutual_info": mi}
    return out


def data_drift(baseline: pd.DataFrame, current: pd.DataFrame,
               cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Check drift between baseline and current using KS for numeric, chi2 for categorical."""
    if cols is None:
        cols = baseline.columns.intersection(current.columns).tolist()
    report = {}
    for col in cols:
        try:
            if pd.api.types.is_numeric_dtype(baseline[col]):
                stat, p = stats.ks_2samp(baseline[col].dropna(), current[col].dropna())
                report[col] = {"ks_stat": float(stat), "pvalue": float(p)}
            else:
                # chi2 on contingency table of categories
                tbl = pd.crosstab(baseline[col].fillna("__NA__"), current[col].fillna("__NA__"))
                chi2, p, _, _ = stats.chi2_contingency(tbl)
                report[col] = {"chi2": float(chi2), "pvalue": float(p)}
        except Exception:
            report[col] = cast(Any, {"error": "test_failed"})
    return report


def train_test_mismatch(train: pd.DataFrame, test: pd.DataFrame,
                        cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """Wrapper around data_drift to check train vs test mismatch."""
    return data_drift(train, test, cols)


def run_risk_checks(
    df: pd.DataFrame,
    target: Optional[str] = None,
    sensitive: Optional[List[str]] = None,
    baseline: Optional[pd.DataFrame] = None,
    train: Optional[pd.DataFrame] = None,
    test: Optional[pd.DataFrame] = None,
    mlvern_dir: Optional[str] = None,
) -> Dict[str, Any]:

    report: Dict[str, Any] = {}
    if target is not None:
        report["class_imbalance"] = class_imbalance(df, target)
        report["target_leakage"] = target_leakage_detection(df, target)
    if sensitive:
        report["sensitive_imbalance"] = sensitive_attribute_imbalance(df, sensitive)
    if baseline is not None:
        report["sampling_bias"] = sampling_bias(baseline, df)
        report["data_drift"] = data_drift(baseline, df)
    if train is not None and test is not None:
        report["train_test_mismatch"] = train_test_mismatch(train, test)

    # Save report if mlvern_dir is provided
    if mlvern_dir is not None:
        reports_dir = Path(mlvern_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "risk_report.json"
        report_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    return report
