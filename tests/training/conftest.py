"""
Pytest configuration and shared fixtures for training module tests.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def tmp_mlvern_dir():
    """Create a temporary directory for mlvern projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_train_data():
    """Create sample training data with features and target."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_val_data():
    """Create sample validation data."""
    np.random.seed(43)
    X = np.random.randn(30, 5)
    y = np.random.randint(0, 2, 30)
    return X, y


@pytest.fixture
def logistic_model():
    """Create a logistic regression model."""
    return LogisticRegression(random_state=42)


@pytest.fixture
def forest_model():
    """Create a random forest classifier."""
    return RandomForestClassifier(n_estimators=10, random_state=42)


@pytest.fixture
def sample_df():
    """Create sample dataframe with features and target."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "feat1": np.random.randn(50),
            "feat2": np.random.randn(50),
            "feat3": np.random.randn(50),
            "target": np.random.randint(0, 2, 50),
        }
    )
