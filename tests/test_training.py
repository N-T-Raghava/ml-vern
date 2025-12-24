from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlvern.train.trainer import train_model


def test_training_returns_metrics():
    X, y = make_classification(n_samples=50)
    model = LogisticRegression()

    trained, metrics = train_model(model, X[:40], y[:40], X[40:], y[40:])

    assert "accuracy" in metrics
