import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.metrics import mutual_info_score
except Exception:
    mutual_info_score = None


def _mutual_info_series(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """Estimate mutual information between two series.

    Uses sklearn if available, otherwise discretizes and computes empirical MI.
    """
    if mutual_info_score is not None:
        try:
            # sklearn expects arrays of labels for discrete mutual_info_score
            # If numeric, discretize
            if pd.api.types.is_numeric_dtype(x):
                x_disc = pd.cut(x, bins=bins, labels=False)
            else:
                x_disc = x.astype(str)
            if pd.api.types.is_numeric_dtype(y):
                y_disc = pd.cut(y, bins=bins, labels=False)
            else:
                y_disc = y.astype(str)
            return float(mutual_info_score(x_disc, y_disc))
        except Exception:
            pass

    # Fallback: discretize and compute empirical MI
    x_disc = pd.cut(x.fillna("__NA__"), bins=bins, labels=False) \
        if pd.api.types.is_numeric_dtype(x) else x.fillna("__NA__").astype(str)

    y_disc = pd.cut(y.fillna("__NA__"), bins=bins, labels=False) \
        if pd.api.types.is_numeric_dtype(y) else y.fillna("__NA__").astype(str)

    x_vals, x_counts = np.unique(x_disc, return_counts=True)
    y_vals, y_counts = np.unique(y_disc, return_counts=True)
    joint_vals, joint_counts = np.unique(list(zip(x_disc, y_disc)), return_counts=True, axis=0)

    n = len(x_disc)
    mi = 0.0
    px = {v: c / n for v, c in zip(x_vals, x_counts)}
    py = {v: c / n for v, c in zip(y_vals, y_counts)}
    pj = {tuple(v): c / n for v, c in zip(joint_vals, joint_counts)}

    for (xi, yi), pxy in pj.items():
        if pxy <= 0:
            continue
        pxi = px.get(xi, 1e-12)
        pyi = py.get(yi, 1e-12)
        mi += pxy * np.log(pxy / (pxi * pyi))
    return float(mi)


def numeric_summary(df: pd.DataFrame,
                    cols: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Compute mean, median, std, skewness for numeric columns."""
    if cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
    else:
        numeric_cols = cols

    out: Dict[str, Dict[str, Any]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            out[col] = {"status": "no_data"}
            continue
        out[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "count": int(series.count()),
        }
    return out


def distribution_shape(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Assess approximate distribution shape using skewness and kurtosis."""
    s = df[col].dropna()
    if s.empty:
        return {"status": "no_data"}
    skew = float(s.skew())
    kurt = float(s.kurtosis())
    shape = "normal"
    if abs(skew) > 1 or kurt > 3:
        shape = "heavy-tailed"
    return {"skewness": skew, "kurtosis": kurt, "shape": shape}


def correlations(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Compute correlation matrix (pearson or spearman)."""
    if method not in ("pearson", "spearman"):
        raise ValueError("method must be 'pearson' or 'spearman'")
    numeric = df.select_dtypes(include="number")
    if method == "pearson":
        return numeric.corr(method="pearson")
    return numeric.corr(method="spearman")


def feature_target_association(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    """Compute associations between features and target (correlation + mutual info)."""
    out: Dict[str, Any] = {}
    if target not in df.columns:
        return {"error": f"Target '{target}' not in dataframe"}
    for col in df.columns:
        if col == target:
            continue
        try:
            mi = _mutual_info_series(df[col], df[target])
        except Exception:
            mi = 0.0
        corr = None
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
            corr = float(df[col].corr(df[target]))
        out[col] = {"mutual_info": float(mi), "correlation": corr}
    return out


def hypothesis_test_two_samples(x: pd.Series, y: pd.Series) -> Dict[str, Any]:
    """Perform two-sample t-test (Welch's t-test)."""
    x = x.dropna()
    y = y.dropna()
    if len(x) < 2 or len(y) < 2:
        return {"status": "skipped", "reason": "insufficient_rows"}
    stat, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return {"statistic": float(stat), "pvalue": float(p)}


def vif(df: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, float]:
    """Compute Variance Inflation Factor for numeric features.

    Uses linear regression via numpy to compute R^2 for each feature against others.
    """
    if cols is None:
        numeric = df.select_dtypes(include="number").dropna()
        cols = numeric.columns.tolist()
    else:
        numeric = df[cols].dropna()

    X = numeric.values.astype(float)
    n, p = X.shape
    if p == 0:
        return {}

    vifs: Dict[str, float] = {}
    for i, col in enumerate(cols):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        if X_other.shape[1] == 0:
            vifs[col] = 0.0
            continue
        # Add intercept
        X_design = np.column_stack([np.ones(X_other.shape[0]), X_other])
        # solve least squares
        coef, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        y_pred = X_design.dot(coef)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot
        if r2 >= 1.0:
            vifs[col] = float('inf')
        else:
            vifs[col] = float(1.0 / (1.0 - r2))
    return vifs


def redundant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """Return pairs of features with abs(correlation) >= threshold."""
    corr = correlations(df, method="pearson")
    pairs: List[Tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            val = float(corr.at[a, b])
            if abs(val) >= threshold:
                pairs.append((a, b, val))
    return pairs


def interaction_patterns(df: pd.DataFrame,
                         target: Optional[str] = None, top_n: int = 10) -> Dict[str, Any]:
    """Detect simple interaction patterns by checking pairwise product correlation with target."""
    num = df.select_dtypes(include="number")
    cols = num.columns.tolist()
    interactions: List[Tuple[str, str, float]] = []
    if target is not None and target in df.columns:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                prod = num[cols[i]] * num[cols[j]]
                corr = prod.corr(df[target])
                if pd.isna(corr):
                    continue
                interactions.append((cols[i], cols[j], float(corr)))
        interactions.sort(key=lambda x: abs(x[2]), reverse=True)
        return {"interactions": interactions[:top_n]}
    return {"interactions": []}


def dimensionality_signals(df: pd.DataFrame, n_components: int = 5) -> Dict[str, Any]:
    num = df.select_dtypes(include="number").dropna()
    if num.shape[0] == 0:
        return {"status": "no_numeric_data"}
    X = num.values.astype(float)
    # center
    Xc = X - X.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        explained = (S ** 2) / (S ** 2).sum()
        return {"explained_variance_ratio": explained[:n_components].tolist()}
    except np.linalg.LinAlgError:
        return {"status": "svd_failed"}


def compute_statistics(df: pd.DataFrame, target: Optional[str] = None,
                       mlvern_dir: Optional[str] = None) -> Dict[str, Any]:
    """Top-level statistics collection combining many of the above functions."""

    stats_report: Dict[str, Any] = {}
    stats_report["numeric_summary"] = numeric_summary(df)
    stats_report["distribution_shapes"] = {
        c: distribution_shape(df, c) for c in df.select_dtypes(include="number").columns
    }
    stats_report["correlations_pearson"] = correlations(df, method="pearson").to_dict()
    stats_report["correlations_spearman"] = correlations(df, method="spearman").to_dict()
    if target is not None:
        stats_report["feature_target_association"] = feature_target_association(df, target)
    stats_report["vif"] = vif(df)
    stats_report["redundant_features"] = redundant_features(df)
    stats_report["interaction_patterns"] = interaction_patterns(df, target)
    stats_report["dimensionality_signals"] = dimensionality_signals(df)

    # Save report if mlvern_dir is provided
    if mlvern_dir is not None:
        reports_dir = Path(mlvern_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "statistics_report.json"
        report_path.write_text(json.dumps(stats_report, indent=4), encoding="utf-8")

    return stats_report
