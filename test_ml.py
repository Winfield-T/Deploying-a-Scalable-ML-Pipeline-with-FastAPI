import numpy as np
import pytest

from ml.model import compute_model_metrics, inference, train_model


def test_train_model_returns_fitted_classifier():
    """
    Verify train_model returns a fitted estimator with a predict method.
    """
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)
    assert hasattr(model, "predict")


def test_inference_returns_correct_shape():
    """
    Verify inference returns a predictions array with the same length as input.
    """
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert preds.shape == y_train.shape


def test_compute_model_metrics_perfect_score():
    """
    Verify compute_model_metrics returns 1.0 for all metrics when predictions are perfect.
    """
    y = np.array([0, 1, 0, 1, 1])
    preds = np.array([0, 1, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0
