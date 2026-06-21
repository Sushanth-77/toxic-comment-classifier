"""Tests for src.toxic_clf.metrics."""

import numpy as np

from toxic_clf.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    result = compute_metrics(y_true, y_prob, label_names=["a", "b"])
    assert result["macro_f1"] == 1.0
    assert result["macro_precision"] == 1.0
    assert result["macro_recall"] == 1.0


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.6, 0.4], [0.3, 0.7]])
    result = compute_metrics(y_true, y_prob, label_names=["a", "b"])
    assert set(result.keys()) == {
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "macro_roc_auc",
        "per_label",
    }
    assert set(result["per_label"].keys()) == {"a", "b"}
    assert set(result["per_label"]["a"].keys()) == {
        "f1",
        "precision",
        "recall",
        "roc_auc",
    }


def test_compute_metrics_handles_label_with_no_positives():
    # label "b" has zero positives in y_true -> ROC-AUC undefined for it
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.1], [0.8, 0.2], [0.1, 0.1]])
    result = compute_metrics(y_true, y_prob, label_names=["a", "b"])
    assert (
        result["per_label"]["b"]["roc_auc"] != result["per_label"]["b"]["roc_auc"]
    )  # is NaN
    # macro_roc_auc should only average over label "a" (which has variance)
    assert result["macro_roc_auc"] == result["per_label"]["a"]["roc_auc"]


def test_compute_metrics_respects_threshold():
    y_true = np.array([[1], [0]])
    y_prob = np.array([[0.6], [0.6]])
    result_low_thresh = compute_metrics(
        y_true, y_prob, threshold=0.5, label_names=["a"]
    )
    result_high_thresh = compute_metrics(
        y_true, y_prob, threshold=0.7, label_names=["a"]
    )
    # at threshold=0.5, both predicted positive -> recall=1, precision=0.5
    # at threshold=0.7, both predicted negative -> recall=0
    assert result_low_thresh["per_label"]["a"]["recall"] == 1.0
    assert result_high_thresh["per_label"]["a"]["recall"] == 0.0
