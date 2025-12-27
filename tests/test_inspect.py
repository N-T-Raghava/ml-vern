import json
import os
import tempfile

import pandas as pd
import pytest

from mlvern.data.inspect import inspect_data


class TestInspectDataBasic:

    def test_creates_report_with_valid_data(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert "statistics" in report
            assert "missing_values" in report["statistics"]
            assert os.path.exists(f"{tmp}/reports/data_validation_report.json")

    def test_report_metadata_structure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "b", tmp)

            assert "metadata" in report
            assert report["metadata"]["library"] == "mlvern"
            assert report["metadata"]["version"] == "0.1.0"
            assert "generated_at" in report["metadata"]

    def test_report_sections_exist(self):
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert "dataset_summary" in report
            assert "statistics" in report
            assert "target_analysis" in report
            assert "vulnerabilities" in report
            assert "recommendations" in report

    def test_report_saved_as_json(self):
        df = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
        with tempfile.TemporaryDirectory() as tmp:
            inspect_data(df, "y", tmp)
            report_path = f"{tmp}/reports/data_validation_report.json"

            assert os.path.exists(report_path)
            with open(report_path) as f:
                saved_report = json.load(f)
            assert isinstance(saved_report, dict)


class TestDatasetSummary:

    def test_correct_row_count(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert report["dataset_summary"]["rows"] == 5

    def test_correct_column_count(self):
        df = pd.DataFrame({
            "col1": [1, 2],
            "col2": [3, 4],
            "col3": [5, 6],
            "target": [0, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert report["dataset_summary"]["columns"] == 4

    def test_memory_usage_calculation(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert "memory_mb" in report["dataset_summary"]
            assert report["dataset_summary"]["memory_mb"] > 0

    def test_large_dataset_summary(self):
        df = pd.DataFrame({
            "a": range(10000),
            "b": range(10000),
            "target": [i % 2 for i in range(10000)]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert report["dataset_summary"]["rows"] == 10000
            assert report["dataset_summary"]["columns"] == 3


class TestMissingValuesDetection:

    def test_no_missing_values(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert len(report["statistics"]["missing_values"]) == 0
            assert not any(v["type"] == "MISSING_VALUES"
                          for v in report["vulnerabilities"])

    def test_detects_missing_values(self):
        df = pd.DataFrame({
            "x": [1, 2, None, 4],
            "y": [0, None, 1, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert len(report["statistics"]["missing_values"]) > 0
            assert any(v["type"] == "MISSING_VALUES"
                      for v in report["vulnerabilities"])

    def test_missing_values_warning_message(self):
        df = pd.DataFrame({
            "x": [1, None, None, 4],
            "y": [0, 1, 1, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            missing_vulns = [v for v in report["vulnerabilities"]
                            if v["type"] == "MISSING_VALUES"]
            assert len(missing_vulns) == 1
            assert "2 missing values detected" in missing_vulns[0]["message"]

    def test_missing_values_recommendation(self):
        df = pd.DataFrame({"x": [1, None, 3], "y": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert any("imputing" in rec.lower()
                      for rec in report["recommendations"])


class TestDuplicatesDetection:

    def test_no_duplicates(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert report["statistics"]["duplicates"] == 0
            assert not any(v["type"] == "DUPLICATES"
                          for v in report["vulnerabilities"])

    def test_detects_duplicates(self):
        df = pd.DataFrame({
            "x": [1, 2, 1, 4],
            "y": [0, 1, 0, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert report["statistics"]["duplicates"] == 1
            assert any(v["type"] == "DUPLICATES"
                      for v in report["vulnerabilities"])

    def test_duplicate_warning_message(self):
        df = pd.DataFrame({
            "x": [1, 1, 1, 4],
            "y": [0, 0, 0, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            dup_vulns = [v for v in report["vulnerabilities"]
                        if v["type"] == "DUPLICATES"]
            assert len(dup_vulns) == 1
            assert "duplicate rows detected" in dup_vulns[0]["message"]

    def test_duplicate_removal_recommendation(self):
        df = pd.DataFrame({"x": [1, 1], "y": [0, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "y", tmp)

            assert any("removing duplicate" in rec.lower()
                      for rec in report["recommendations"])


class TestTargetAnalysis:

    def test_target_column_missing_raises_error(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "nonexistent", tmp)

            missing_target = [v for v in report["vulnerabilities"]
                             if v["type"] == "TARGET_MISSING"]
            assert len(missing_target) == 1
            assert "not found" in missing_target[0]["message"]

    def test_target_column_critical_severity(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "missing_target", tmp)

            missing_target = [v for v in report["vulnerabilities"]
                             if v["type"] == "TARGET_MISSING"]
            assert missing_target[0]["severity"] == "CRITICAL"

    def test_valid_target_analysis(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "target": [0, 1, 0, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert "target_analysis" in report
            assert report["target_analysis"]["target"] == "target"
            assert "class_distribution" in report["target_analysis"]

    def test_class_distribution_accuracy(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "target": [0, 0, 0, 1, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            dist = report["target_analysis"]["class_distribution"]
            assert dist[0] == 3
            assert dist[1] == 2

    def test_imbalance_ratio_calculation(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "target": [0, 0, 0, 0, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            imbalance = report["target_analysis"]["imbalance_ratio"]
            assert imbalance == 4.0


class TestClassImbalance:

    def test_no_class_imbalance(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4],
            "target": [0, 0, 1, 1]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert not any(v["type"] == "CLASS_IMBALANCE"
                          for v in report["vulnerabilities"])

    def test_detects_class_imbalance(self):
        df = pd.DataFrame({
            "x": list(range(100)),
            "target": [0] * 95 + [1] * 5
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            imbalance_vulns = [v for v in report["vulnerabilities"]
                              if v["type"] == "CLASS_IMBALANCE"]
            assert len(imbalance_vulns) == 1

    def test_imbalance_warning_severity(self):
        df = pd.DataFrame({
            "x": list(range(100)),
            "target": [0] * 90 + [1] * 10
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            imbalance_vulns = [v for v in report["vulnerabilities"]
                              if v["type"] == "CLASS_IMBALANCE"]
            assert imbalance_vulns[0]["severity"] == "WARNING"

    def test_imbalance_recommendation(self):
        df = pd.DataFrame({
            "x": list(range(50)),
            "target": [0] * 48 + [1] * 2
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert any("class weight" in rec.lower() or "resampl" in rec.lower()
                      for rec in report["recommendations"])


class TestEdgeCases:

    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="Dataset is empty"):
                inspect_data(df, "target", tmp)

    def test_single_row_dataset(self):
        df = pd.DataFrame({"x": [1], "target": [0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert report["dataset_summary"]["rows"] == 1

    def test_single_column_dataset(self):
        df = pd.DataFrame({"target": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert report["dataset_summary"]["columns"] == 1

    def test_all_missing_values_in_column(self):
        df = pd.DataFrame({
            "x": [None, None, None],
            "target": [0, 1, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            missing = report["statistics"]["missing_values"]
            assert missing["x"] == 3

    def test_multiclass_target(self):
        df = pd.DataFrame({
            "x": range(10),
            "target": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            dist = report["target_analysis"]["class_distribution"]
            assert len(dist) == 3

    def test_string_data_types(self):
        df = pd.DataFrame({
            "name": ["alice", "bob", "charlie"],
            "target": [0, 1, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            dtypes = report["statistics"]["dtypes"]
            assert "object" in dtypes["name"]

    def test_mixed_data_types(self):
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "target": [0, 1, 0]
        })
        with tempfile.TemporaryDirectory() as tmp:
            report = inspect_data(df, "target", tmp)

            assert len(report["statistics"]["dtypes"]) == 4
