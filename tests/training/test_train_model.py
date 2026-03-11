"""Tests for the train_model() function."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlvern.train.trainer import train_model


class TestTrainModel:
    """Tests for the train_model() function."""

    def test_train_model_basic(self, sample_train_data, logistic_model):
        """Test basic training without validation data."""
        X, y = sample_train_data
        model, metrics = train_model(logistic_model, X, y)

        assert model is not None
        assert isinstance(metrics, dict)
        assert len(metrics) == 0  # No validation data provided

    def test_train_model_with_validation(
        self, sample_train_data, sample_val_data, logistic_model
    ):
        """Test training with validation data."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        model, metrics = train_model(logistic_model, X_train, y_train, X_val, y_val)

        assert model is not None
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_train_model_accuracy_score(
        self, sample_train_data, sample_val_data, forest_model
    ):
        """Test that metrics contain valid accuracy scores."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        model, metrics = train_model(forest_model, X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)
        expected_accuracy = accuracy_score(y_val, preds)

        assert metrics["accuracy"] == expected_accuracy

    def test_train_model_returns_fitted_model(self, sample_train_data, logistic_model):
        """Test that returned model is fitted."""
        X, y = sample_train_data
        model, _ = train_model(logistic_model, X, y)

        # Fitted model should have coef_ attribute
        assert hasattr(model, "coef_")
        assert model.coef_ is not None

    def test_train_model_different_algorithms(self, sample_train_data, sample_val_data):
        """Test training with different model algorithms."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        lr_model, lr_metrics = train_model(
            LogisticRegression(random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
        )
        rf_model, rf_metrics = train_model(
            RandomForestClassifier(n_estimators=5, random_state=42),
            X_train,
            y_train,
            X_val,
            y_val,
        )

        assert "accuracy" in lr_metrics
        assert "accuracy" in rf_metrics
        assert lr_model is not None
        assert rf_model is not None
