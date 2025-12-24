import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    roc_curve,
)


def auto_plot(task, y_true=None, y_pred=None, y_prob=None, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)

    if task == "classification" and y_true is not None and y_pred is not None:

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
            plt.savefig(os.path.join(output_dir, "roc_curve.png"))
            plt.close()
