"""Shared multilabel evaluation metrics.

Used identically across Phases 6/7/8 (Logistic Regression, BiLSTM,
DistilBERT) so model comparison in Phase 9 is comparing the same
metric, computed the same way, every time.
"""

import logging

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

DEFAULT_THRESHOLD = 0.5


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    label_names: list | None = None,
) -> dict:
    """Compute per-label and macro-averaged multilabel metrics.

    Args:
        y_true: (n_samples, n_labels) binary ground truth array.
        y_prob: (n_samples, n_labels) predicted probabilities in [0, 1].
        threshold: probability cutoff for converting y_prob to binary
            predictions. Fixed at 0.5 project-wide for comparability
            across models.
        label_names: names for each label column, defaults to LABEL_COLS.

    Returns:
        dict with keys:
            - "macro_f1", "macro_precision", "macro_recall"
            - "macro_roc_auc" (mean of per-label ROC-AUC, skipping labels
              with no positive examples in y_true)
            - "per_label": dict of label_name -> {f1, precision, recall, roc_auc}
    """
    label_names = label_names or LABEL_COLS
    y_pred = (y_prob >= threshold).astype(int)

    per_label = {}
    roc_aucs = []
    for i, name in enumerate(label_names):
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)

        if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
            roc_auc = float("nan")
            logger.warning(
                "Label '%s' has no positive/negative variance in y_true; "
                "ROC-AUC undefined, excluded from macro_roc_auc",
                name,
            )
        else:
            roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
            roc_aucs.append(roc_auc)

        per_label[name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_roc_auc = float(np.mean(roc_aucs)) if roc_aucs else float("nan")

    return {
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_roc_auc": macro_roc_auc,
        "per_label": per_label,
    }
