import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class DataInspector:
    """Comprehensive data profiling and validation framework."""

    def __init__(
        self,
        df: pd.DataFrame,
        target: Optional[str] = None,
        mlvern_dir: str = ".",
    ):
        self.df = df
        self.target = target
        self.mlvern_dir = mlvern_dir
        self.report: dict[str, Any] = {}

    def validate_input(self) -> bool:
        """Validate input data."""
        if self.df.empty:
            raise ValueError("Dataset is empty")
        return True

    def safe_numeric_profile(self, min_rows: int = 2) -> dict[str, Any]:
        """Check if dataset is large enough for numeric profiling.

        Returns analysis or explicit skip status.
        """
        n_rows = len(self.df)
        if n_rows < min_rows:
            return {
                "status": "skipped",
                "reason": "insufficient_rows",
                "required_min_rows": min_rows,
                "actual_rows": n_rows,
            }
        return {"status": "available"}

    def profile_data(self) -> dict[str, Any]:
        """Part 1: Comprehensive data profiling."""
        profile = {
            "dataset_shape": self._profile_shape(),
            "schema": self._profile_schema(),
            "missing_values": self._profile_missing(),
            "duplicates": self._profile_duplicates(),
            "cardinality": self._profile_cardinality(),
            "numeric_ranges": self._profile_numeric_ranges(),
            "outliers": self._profile_outliers(),
            "target_distribution": self._profile_target() if self.target else {},
        }
        return profile

    def validate_data(self) -> dict[str, Any]:
        """Part 2: Comprehensive data validation."""
        validation = {
            "schema_validation": self._validate_schema(),
            "range_constraints": self._validate_ranges(),
            "null_thresholds": self._validate_null_thresholds(),
            "type_consistency": self._validate_type_consistency(),
            "leakage_checks": self._validate_leakage(),
            "temporal_validity": self._validate_temporal(),
        }
        return validation

    def _profile_shape(self) -> dict[str, Any]:
        """Profile dataset shape and size."""
        return {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
            "sparsity_percent": round(
                (self.df.size - self.df.count().sum()) / self.df.size * 100, 2
            ),
        }

    def _profile_schema(self) -> dict[str, Any]:
        """Profile data types and schema."""
        schema = {}
        for col in self.df.columns:
            schema[col] = {
                "dtype": str(self.df[col].dtype),
                "non_null_count": int(self.df[col].count()),
            }
        return schema

    def _profile_missing(self) -> dict[str, Any]:
        """Profile missing values with patterns."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)

        result = {}
        for col in missing[missing > 0].index:
            result[col] = {
                "count": int(missing[col]),
                "percentage": float(missing_pct[col]),
            }

        return {
            "total_missing": int(missing.sum()),
            "columns_affected": int((missing > 0).sum()),
            "details": result,
        }

    def _profile_duplicates(self) -> dict[str, Any]:
        """Profile duplicate rows."""
        total_dups = int(self.df.duplicated().sum())

        dup_info: dict[str, Any] = {"total": total_dups}
        if total_dups > 0:
            dup_info["percentage"] = round(total_dups / len(self.df) * 100, 2)
            # Check duplicates by subset
            dup_subset = int(
                self.df.duplicated(subset=self.df.columns[:-1], keep=False).sum()
            )
            dup_info["by_features"] = dup_subset

        return dup_info

    def _profile_cardinality(self) -> dict[str, Any]:
        """Profile cardinality of categorical features."""
        cardinality = {}
        categorical_cols = self.df.select_dtypes(exclude="number").columns

        for col in categorical_cols:
            unique_count = int(self.df[col].nunique())
            cardinality[col] = {
                "unique_values": unique_count,
                "cardinality_ratio": round(unique_count / len(self.df), 4),
                "top_values": self.df[col].value_counts().head(5).to_dict(),
            }

        return cardinality

    def _profile_numeric_ranges(self) -> dict[str, Any]:
        """Profile ranges and statistics of numeric features."""

        check = self.safe_numeric_profile(min_rows=2)
        if check["status"] == "skipped":
            return check

        numeric_cols = self.df.select_dtypes(include="number").columns
        ranges = {}

        for col in numeric_cols:
            # Use ddof=0 to avoid division-by-zero warnings
            with np.errstate(divide='ignore', invalid='ignore'):
                std_val = float(self.df[col].std(ddof=0))
                if pd.isna(std_val):
                    std_val = 0.0

            ranges[col] = {
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": std_val,
                "q25": float(self.df[col].quantile(0.25)),
                "q75": float(self.df[col].quantile(0.75)),
            }

        return ranges

    def _profile_outliers(self) -> dict[str, Any]:
        """Detect outliers using IQR method.

        Skips if <5 rows (insufficient for reliable outlier detection).
        """
        check = self.safe_numeric_profile(min_rows=5)
        if check["status"] == "skipped":
            return check

        numeric_cols = self.df.select_dtypes(include="number").columns
        outliers = {}

        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_count = int(
                ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            )

            if outlier_count > 0:
                outliers[col] = {
                    "count": outlier_count,
                    "percentage": round(outlier_count / len(self.df) * 100, 2),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }

        return outliers

    def _profile_target(self) -> dict[str, Any]:
        """Profile target variable distribution."""
        if self.target not in self.df.columns:
            return {"error": f"Target '{self.target}' not found"}

        target_col = self.df[self.target]
        class_dist = target_col.value_counts().to_dict()

        if len(class_dist) == 0:
            return {}

        max_class = max(class_dist.values())
        min_class = min(class_dist.values())
        imbalance_ratio = round(max_class / max(min_class, 1), 2)

        return {
            "target": self.target,
            "type": "categorical" if target_col.dtype == "object" else "numeric",
            "class_distribution": class_dist,
            "imbalance_ratio": imbalance_ratio,
            "n_classes": len(class_dist),
        }

    def _validate_schema(self) -> dict[str, Any]:
        """Validate schema consistency."""
        issues = []

        # Check for unnamed columns
        if self.df.columns.isnull().any():
            issues.append("Dataset has unnamed columns")

        # Check for duplicate column names
        if self.df.columns.duplicated().any():
            issues.append("Dataset has duplicate column names")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
        }

    def _validate_ranges(self) -> dict[str, Any]:
        """Validate value ranges."""
        violations = {}
        numeric_cols = self.df.select_dtypes(include="number").columns

        for col in numeric_cols:
            # Check for inf values
            inf_count = int(np.isinf(self.df[col]).sum())
            if inf_count > 0:
                violations[col] = {
                    "type": "infinite_values",
                    "count": inf_count,
                }

        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
        }

    def _validate_null_thresholds(self) -> dict[str, Any]:
        """Validate null value thresholds (default: 50%)."""
        threshold = 0.5
        violations = {}

        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        for col in missing_pct[missing_pct >= threshold * 100].index:
            violations[col] = {
                "missing_percentage": float(missing_pct[col]),
                "threshold": threshold * 100,
            }

        return {
            "is_valid": len(violations) == 0,
            "threshold_percent": threshold * 100,
            "violations": violations,
        }

    def _validate_type_consistency(self) -> dict[str, Any]:
        """Validate type consistency across columns."""
        issues = {}

        for col in self.df.columns:
            col_dtype = str(self.df[col].dtype)

            # Check for mixed types in object columns
            if col_dtype == "object":
                types_in_col = self.df[col].apply(lambda x: type(x).__name__).unique()
                if len(types_in_col) > 2:  # Allow None and one other type
                    issues[col] = {
                        "types_found": list(types_in_col),
                        "message": "Mixed types detected",
                    }

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
        }

    def _validate_leakage(self) -> dict[str, Any]:
        """Check for potential data leakage patterns."""
        leakage_indicators = []

        # Correlation requires >= 2 rows
        if len(self.df) < 2:
            return {
                "has_leakage_risk": False,
                "indicators": [],
                "status": "skipped",
                "reason": "insufficient_rows",
            }

        if self.target and self.target in self.df.columns:
            # Check for perfect correlations with target
            numeric_cols = self.df.select_dtypes(include="number").columns
            if self.target in numeric_cols:
                for col in numeric_cols:
                    if col != self.target:
                        corr = abs(self.df[col].corr(self.df[self.target]))
                        if corr > 0.99:
                            leakage_indicators.append({
                                "feature": col,
                                "correlation": float(corr),
                                "message": "Perfect correlation with target",
                            })

        return {
            "has_leakage_risk": len(leakage_indicators) > 0,
            "indicators": leakage_indicators,
        }

    def _validate_temporal(self) -> dict[str, Any]:
        """Validate temporal columns if present.

        Detects temporal columns by name heuristics or successful parsing.
        """
        temporal_cols = {}

        for col in self.df.columns:
            try:
                parsed = pd.to_datetime(self.df[col], errors="coerce")
                # count entries that parsed as datetime
                parsed_count = int(parsed.notnull().sum())
                original_nulls = int(self.df[col].isnull().sum())
                invalid_count = int(parsed.isnull().sum() - original_nulls)

                # heuristics: column name or any successful parse
                name_hint = any(
                    key in col.lower()
                    for key in ("date", "time", "timestamp", "created", "updated", "at")
                )

                if name_hint or parsed_count > 0:
                    temporal_cols[col] = {
                        "dtype": str(self.df[col].dtype),
                        "parsed_count": parsed_count,
                        "invalid_dates": invalid_count,
                        "date_range": {
                            "min": str(parsed.min()) if parsed_count > 0 else None,
                            "max": str(parsed.max()) if parsed_count > 0 else None,
                        },
                    }
            except Exception:
                temporal_cols[col] = {"error": "Failed to parse as datetime"}

        return {
            "has_temporal_columns": len(temporal_cols) > 0,
            "columns": temporal_cols,
        }

    def inspect(self) -> dict[str, Any]:
        """Run complete inspection."""
        self.validate_input()

        self.report = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "library": "mlvern",
                "version": "0.1.0",
            },
            "part_1_profiling": self.profile_data(),
            "part_2_validation": self.validate_data(),
            "vulnerabilities": [],
            "recommendations": [],
        }

        self._assess_vulnerabilities()
        return self.report

    def _assess_vulnerabilities(self):
        """Assess vulnerabilities and generate recommendations."""
        profile = self.report["part_1_profiling"]
        validation = self.report["part_2_validation"]

        # Check missing values
        if profile["missing_values"]["total_missing"] > 0:
            missing_count = profile["missing_values"]["total_missing"]
            self.report["vulnerabilities"].append({
                "severity": "WARNING",
                "type": "MISSING_VALUES",
                "message": f"{missing_count} missing values detected",
            })
            msg = (
                "Consider imputing missing values using mean, median, "
                "or KNN imputation"
            )
            self.report["recommendations"].append(msg)

        # Check duplicates
        if profile["duplicates"]["total"] > 0:
            self.report["vulnerabilities"].append({
                "severity": "WARNING",
                "type": "DUPLICATES",
                "message": f"{profile['duplicates']['total']} duplicate rows detected",
            })
            self.report["recommendations"].append(
                "Consider removing or investigating duplicate rows"
            )

        # Check target
        if self.target and self.target in self.df.columns:
            target_dist = profile["target_distribution"]
            if "imbalance_ratio" in target_dist and target_dist["imbalance_ratio"] > 3:
                self.report["vulnerabilities"].append({
                    "severity": "WARNING",
                    "type": "CLASS_IMBALANCE",
                    "message": f"Imbalance ratio is {target_dist['imbalance_ratio']}",
                })
                self.report["recommendations"].append(
                    "Use SMOTE, class weighting, or resampling for class imbalance"
                )

        # Check null thresholds
        if not validation["null_thresholds"]["is_valid"]:
            self.report["vulnerabilities"].append({
                "severity": "CRITICAL",
                "type": "HIGH_MISSING_THRESHOLD",
                "message": "Columns exceed 50% missing values",
            })
            self.report["recommendations"].append(
                "Consider dropping or heavily imputing columns with >50% missing values"
            )

        # Check leakage
        if validation["leakage_checks"]["has_leakage_risk"]:
            self.report["vulnerabilities"].append({
                "severity": "CRITICAL",
                "type": "DATA_LEAKAGE",
                "message": "Perfect correlation detected with target",
            })
            self.report["recommendations"].append(
                "Investigate and remove features with perfect target correlation"
            )

    def save_report(self, filename: str = "data_inspection_report.json") -> Path:
        """Save inspection report to JSON file and return Path."""
        reports_dir = Path(self.mlvern_dir) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        path = reports_dir / filename
        path.write_text(json.dumps(self.report, indent=4), encoding="utf-8")

        return path


def inspect_data(
    df: pd.DataFrame,
    target: Optional[str] = None,
    mlvern_dir: str = ".",
) -> dict[str, Any]:
    """Convenience function for data inspection."""
    inspector = DataInspector(df, target, mlvern_dir)
    report = inspector.inspect()
    inspector.save_report()
    return report
