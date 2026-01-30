import numpy as np
from sklearn.ensemble import RandomForestClassifier
from train import train_model


def test_train_model_returns_model():
    model = train_model()
    assert model is not None
    assert isinstance(model, RandomForestClassifier)


def test_train_model_accuracy():
    model = train_model()
    # Generate test data with same seed logic
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    accuracy = model.score(X, y)
    assert accuracy > 0.8, f"Expected accuracy > 0.8, got {accuracy}"


def test_numpy_import():
    assert np is not None
    arr = np.array([1, 2, 3])
    assert arr.shape == (3,)
