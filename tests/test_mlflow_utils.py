"""Tests for src.toxic_clf.mlflow_utils."""

import mlflow
import pytest

from toxic_clf.mlflow_utils import (
    log_metrics_dict,
    log_params_dict,
    setup_experiment,
    start_run,
)


@pytest.fixture
def temp_mlflow_tracking(tmp_path, monkeypatch):
    tracking_uri = f"file:{tmp_path / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    yield tracking_uri


def test_setup_experiment_creates_experiment(temp_mlflow_tracking):
    setup_experiment("test-experiment")
    exp = mlflow.get_experiment_by_name("test-experiment")
    assert exp is not None


def test_start_run_yields_active_run(temp_mlflow_tracking):
    setup_experiment("test-experiment")
    with start_run("test-run-1") as run:
        assert mlflow.active_run() is not None
        assert run.info.run_name == "test-run-1"
    assert mlflow.active_run() is None  # run closed after context exits


def test_log_metrics_dict_skips_nan(temp_mlflow_tracking):
    setup_experiment("test-experiment")
    metrics = {
        "macro_f1": 0.75,
        "macro_roc_auc": float("nan"),
        "per_label": {
            "toxic": {
                "f1": 0.8,
                "precision": 0.7,
                "recall": 0.9,
                "roc_auc": float("nan"),
            },
        },
    }
    with start_run("test-run-2"):
        log_metrics_dict(metrics)  # should not raise despite NaN values


def test_log_params_dict_flattens_nested(temp_mlflow_tracking):
    setup_experiment("test-experiment")
    params = {"tfidf": {"max_features": 50000, "min_df": 2}, "top_level": "x"}
    with start_run("test-run-3"):
        log_params_dict(params)  # should not raise
