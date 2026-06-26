"""Logistic Regression + TF-IDF baseline training (Phase 6).

Loads the TF-IDF features and labels produced by the Phase 4/5 DVC pipeline,
fits a OneVsRestClassifier wrapping LogisticRegression (one independent
binary classifier per label — required because LogisticRegression alone is
multiclass, not multilabel), evaluates with the shared compute_metrics(),
logs to MLflow, and persists the fitted model.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from toxic_clf.config import load_params
from toxic_clf.metrics import compute_metrics
from toxic_clf.mlflow_utils import (
    log_metrics_dict,
    log_params_dict,
    setup_experiment,
    start_run,
)

logger = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")


def load_features_and_labels():
    """Load TF-IDF feature matrices and label arrays for train/val splits."""
    X_train = sparse.load_npz(PROCESSED_DIR / "tfidf_train.npz")
    X_val = sparse.load_npz(PROCESSED_DIR / "tfidf_val.npz")

    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    y_train = train_df[LABEL_COLUMNS].values
    y_val = val_df[LABEL_COLUMNS].values

    return X_train, y_train, X_val, y_val


def build_model(logreg_params: dict) -> OneVsRestClassifier:
    """Construct a OneVsRestClassifier wrapping LogisticRegression."""
    base_estimator = LogisticRegression(
        C=logreg_params["C"],
        max_iter=logreg_params["max_iter"],
        class_weight=logreg_params["class_weight"],
        solver=logreg_params["solver"],
    )
    return OneVsRestClassifier(base_estimator)


def fit_and_evaluate(model, X_train, y_train, X_val, y_val, label_names):
    """Fit the model and evaluate on validation data. Returns (model, metrics)."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)
    metrics = compute_metrics(y_val, y_prob, label_names=label_names)
    return model, metrics


def train(run_name: str = "logreg-tfidf-baseline") -> dict:
    """Train, evaluate, and log the Logistic Regression + TF-IDF baseline."""
    params = load_params()
    logreg_params = params["logreg"]

    logger.info("Loading TF-IDF features and labels")
    X_train, y_train, X_val, y_val = load_features_and_labels()
    logger.info("Train shape: %s, Val shape: %s", X_train.shape, X_val.shape)

    model = build_model(logreg_params)

    setup_experiment(params["mlflow"]["experiment_name"])
    with start_run(run_name):
        log_params_dict({"logreg": logreg_params})

        logger.info("Fitting OneVsRestClassifier(LogisticRegression)")
        model, metrics = fit_and_evaluate(
            model, X_train, y_train, X_val, y_val, LABEL_COLUMNS
        )
        log_metrics_dict(metrics)
        logger.info("Validation macro F1: %.4f", metrics["macro_f1"])

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACTS_DIR / "logreg_model.joblib"
        joblib.dump(model, model_path)
        logger.info("Saved model to %s", model_path)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
