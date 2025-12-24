import os

from mlvern.data.inspect import inspect_data
from mlvern.train.trainer import train_model
from mlvern.version.checkout import checkout_commit
from mlvern.version.commit import commit_run, log_commits
from mlvern.visual.auto_plot import auto_plot
from mlvern.visual.eda import basic_eda


class Forge:
    def __init__(self, project: str, base_dir: str = "."):
        self.project = project
        self.base_dir = os.path.abspath(base_dir)
        self.mlvern_dir = os.path.join(self.base_dir, ".mlvern")
        self._init_workspace()

    def _init_workspace(self):
        os.makedirs(self.mlvern_dir, exist_ok=True)
        for d in ["commits", "plots", "reports"]:
            os.makedirs(os.path.join(self.mlvern_dir, d), exist_ok=True)

    def inspect(self, data, target: str):
        return inspect_data(data, target, self.mlvern_dir)

    def plot(self, task: str, y_true=None, y_pred=None, y_prob=None):
        auto_plot(
            task=task,
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            output_dir=os.path.join(self.mlvern_dir, "plots"),
        )

    def eda(self, data):
        basic_eda(data, os.path.join(self.mlvern_dir, "plots", "eda"))

    def train(self, model, X_train, y_train, X_val=None, y_val=None):
        return train_model(model, X_train, y_train, X_val, y_val)

    def commit(self, message: str, model, metrics: dict, params: dict):
        return commit_run(self.mlvern_dir, message, model, metrics, params)

    def checkout(self, commit_id: str):
        return checkout_commit(self.mlvern_dir, commit_id)

    def log(self):
        return log_commits(self.mlvern_dir)
