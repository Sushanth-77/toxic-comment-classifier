"""Tests for the Logistic Regression + TF-IDF baseline training module."""

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from toxic_clf.train_logreg import LABEL_COLUMNS, build_model, fit_and_evaluate


def _synthetic_data(n_samples=40, n_features=20, seed=0):
    rng = np.random.RandomState(seed)
    X = sparse.random(
        n_samples, n_features, density=0.3, random_state=rng, format="csr"
    )
    y = rng.randint(0, 2, size=(n_samples, len(LABEL_COLUMNS)))
    return X, y


def test_build_model_returns_one_vs_rest_classifier():
    params = {
        "C": 0.5,
        "max_iter": 200,
        "class_weight": "balanced",
        "solver": "liblinear",
    }
    model = build_model(params)
    assert isinstance(model, OneVsRestClassifier)
    assert isinstance(model.estimator, LogisticRegression)
    assert model.estimator.C == 0.5
    assert model.estimator.class_weight == "balanced"
    assert model.estimator.solver == "liblinear"


def test_fit_and_evaluate_returns_well_formed_metrics():
    X_train, y_train = _synthetic_data(seed=1)
    X_val, y_val = _synthetic_data(n_samples=15, seed=2)
    params = {
        "C": 1.0,
        "max_iter": 200,
        "class_weight": "balanced",
        "solver": "liblinear",
    }
    model = build_model(params)

    fitted_model, metrics = fit_and_evaluate(
        model, X_train, y_train, X_val, y_val, LABEL_COLUMNS
    )

    assert isinstance(fitted_model, OneVsRestClassifier)
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics


def test_predictions_are_valid_probabilities():
    X_train, y_train = _synthetic_data(n_samples=30, seed=3)
    X_val, y_val = _synthetic_data(n_samples=10, seed=4)
    params = {
        "C": 1.0,
        "max_iter": 200,
        "class_weight": "balanced",
        "solver": "liblinear",
    }
    model = build_model(params)

    fitted_model, _ = fit_and_evaluate(
        model, X_train, y_train, X_val, y_val, LABEL_COLUMNS
    )
    y_prob = fitted_model.predict_proba(X_val)

    assert y_prob.shape == (10, len(LABEL_COLUMNS))
    assert (y_prob >= 0).all() and (y_prob <= 1).all()
