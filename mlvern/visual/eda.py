import json
import os
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def basic_eda(df: pd.DataFrame, output_dir: str,
              mlvern_dir: Optional[str] = None, target: Optional[str] = None):

    resolved_mlvern_dir = mlvern_dir
    if not resolved_mlvern_dir or not os.path.isdir(resolved_mlvern_dir):
        candidate = os.path.join(os.getcwd(), "examples", ".mlvern")
        if os.path.isdir(candidate):
            resolved_mlvern_dir = candidate

    if resolved_mlvern_dir:
        output_dir = os.path.join(resolved_mlvern_dir, "plots", "eda")

    _ensure_dir(output_dir)

    plots_created = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Distribution plots
    dist_dir = os.path.join(output_dir, "distributions")
    _ensure_dir(dist_dir)
    for col in numeric_cols:
        plt.figure()
        df[col].dropna().hist(bins=20)
        plt.title(f"Distribution: {col}")
        p = os.path.join(dist_dir, f"{col}_hist.png")
        plt.savefig(p)
        plt.close()
        plots_created.append(p)

    # Boxplots and violin plots
    box_dir = os.path.join(output_dir, "box_violin")
    _ensure_dir(box_dir)
    for col in numeric_cols:
        # boxplot
        plt.figure()
        plt.boxplot(df[col].dropna().values)
        plt.title(f"Boxplot: {col}")
        p_box = os.path.join(box_dir, f"{col}_box.png")
        plt.savefig(p_box)
        plt.close()
        plots_created.append(p_box)

        # violin
        plt.figure()
        try:
            plt.violinplot(df[col].dropna().values)
            plt.title(f"Violin: {col}")
            p_violin = os.path.join(box_dir, f"{col}_violin.png")
            plt.savefig(p_violin)
            plt.close()
            plots_created.append(p_violin)
        except Exception:
            plt.close()

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        corr_dir = os.path.join(output_dir, "correlation")
        _ensure_dir(corr_dir)
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(max(6, len(numeric_cols) * 0.5), max(4, len(numeric_cols) * 0.4)))
        im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im)
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title("Correlation heatmap")
        p_corr = os.path.join(corr_dir, "correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(p_corr)
        plt.close()
        plots_created.append(p_corr)

    # Target vs feature plots
    if target is not None and target in df.columns:
        tvs_dir = os.path.join(output_dir, "target_vs_features")
        _ensure_dir(tvs_dir)
        # If target is numeric, scatter; if categorical, box by class
        is_target_numeric = pd.api.types.is_numeric_dtype(df[target])
        for col in numeric_cols:
            plt.figure()
            if is_target_numeric:
                plt.scatter(df[col], df[target], alpha=0.6)
                plt.xlabel(col)
                plt.ylabel(target)
                plt.title(f"{col} vs {target}")
            else:
                # boxplot grouped by target
                try:
                    groups = [group.dropna().values for _, group in df.groupby(target)[col]]
                    plt.boxplot(groups)
                    plt.xticks(range(1, len(groups) + 1), df[target].unique())
                    plt.title(f"{col} by {target}")
                except Exception:
                    pass
            p = os.path.join(tvs_dir, f"{col}_vs_{target}.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            plots_created.append(p)

    # Missingness map
    missing_dir = os.path.join(output_dir, "missingness")
    _ensure_dir(missing_dir)
    # Skip the full missingness heatmap for empty data to avoid matplotlib warnings
    if df.shape[0] == 0 or df.shape[1] == 0:
        # create a small placeholder image indicating no data
        plt.figure(figsize=(4, 2))
        plt.text(0.5, 0.5, "no data", ha="center", va="center")
        p_miss = os.path.join(missing_dir, "missingness_map.png")
        plt.axis("off")
        plt.savefig(p_miss)
        plt.close()
        plots_created.append(p_miss)
    else:
        plt.figure(figsize=(10, max(2, df.shape[0] * 0.01)))
        # visual binary map of missingness
        try:
            plt.imshow(df.isnull().T, aspect='auto', cmap='Greys', interpolation='nearest')
            plt.yticks(range(len(df.columns)), df.columns)
            plt.title('Missingness map (columns y-axis)')
            p_miss = os.path.join(missing_dir, "missingness_map.png")
            plt.tight_layout()
            plt.savefig(p_miss)
            plt.close()
            plots_created.append(p_miss)
        except Exception:
            plt.close()

    # Write a small JSON report into mlvern reports dir if resolved_mlvern_dir provided
    if resolved_mlvern_dir is not None:
        reports_dir = os.path.join(resolved_mlvern_dir, "reports")
        _ensure_dir(reports_dir)
        report_path = os.path.join(reports_dir, "eda_report.json")
        summary = {
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target": target,
            "plots": plots_created,
        }
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "plots": plots_created,
    }
