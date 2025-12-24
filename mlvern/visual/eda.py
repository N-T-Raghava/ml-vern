import os

import matplotlib.pyplot as plt
import pandas as pd


def basic_eda(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(exclude="number").columns

    for col in numeric_cols:
        df[col].hist(bins=20)
        plt.title(col)
        plt.savefig(os.path.join(output_dir, f"{col}_hist.png"))
        plt.close()

    for col in categorical_cols:
        df[col].value_counts().head(10).plot(kind="bar")
        plt.title(col)
        plt.savefig(os.path.join(output_dir, f"{col}_bar.png"))
        plt.close()
