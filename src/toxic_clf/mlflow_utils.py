"""MLflow setup and logging conventions, shared across all training scripts."""

import logging
from contextlib import contextmanager

import mlflow

logger = logging.getLogger(__name__)


def setup_experiment(experiment_name: str) -> None:
    """Set the active MLflow experiment, creating it if it doesn't exist."""
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow experiment set to: %s", experiment_name)


@contextmanager
def start_run(run_name: str):
    """Context manager wrapping mlflow.start_run with a required run_name.

    Enforces naming runs explicitly -- unnamed runs are hard to find
    later when comparing models in the MLflow UI.
    """
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Started MLflow run: %s (id=%s)", run_name, run.info.run_id)
        yield run


def log_metrics_dict(metrics: dict, prefix: str = "") -> None:
    """Flatten and log a metrics dict (from toxic_clf.metrics.compute_metrics).

    Macro metrics are logged as top-level keys (e.g. "macro_f1"). Per-label
    metrics are logged with a "label_<name>_<metric>" naming convention
    (e.g. "label_toxic_f1") so they're filterable in the MLflow UI without
    colliding with the macro keys.

    NaN values (e.g. ROC-AUC for labels with no variance) are skipped --
    MLflow does not accept NaN metric values.
    """
    flat = {}
    for key, value in metrics.items():
        if key == "per_label":
            for label_name, label_metrics in value.items():
                for metric_name, metric_value in label_metrics.items():
                    flat[f"{prefix}label_{label_name}_{metric_name}"] = metric_value
        else:
            flat[f"{prefix}{key}"] = value

    skipped = [k for k, v in flat.items() if v != v]  # NaN check (NaN != NaN)
    if skipped:
        logger.warning("Skipping NaN metrics (not loggable to MLflow): %s", skipped)

    clean = {k: v for k, v in flat.items() if v == v}
    mlflow.log_metrics(clean)
    logger.info("Logged %d metrics to MLflow", len(clean))


def log_params_dict(params: dict, prefix: str = "") -> None:
    """Log a (possibly nested) params dict to MLflow, flattening one level."""
    flat = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{prefix}{key}_{sub_key}"] = sub_value
        else:
            flat[f"{prefix}{key}"] = value
    mlflow.log_params(flat)
    logger.info("Logged %d params to MLflow", len(flat))
