"""DistilBERT fine-tuning with early stopping (Phase 8).

Designed to run unmodified in two environments: locally (CPU, smoke-test
only -- full training is impractical on CPU for this model size) and on
a GPU runtime such as Kaggle Notebooks (real training). Data location is
controlled by the TOXIC_CLF_DATA_DIR env var so the same script works in
both places without a code fork.

Mirrors train_logreg.py / train_bilstm.py's structure: load data -> build
model -> train -> evaluate -> log -> save. Reuses the same compute_metrics()
and 0.5 threshold for direct comparability with the other two models.
"""

import copy
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from toxic_clf.config import load_params
from toxic_clf.datasets_transformer import (
    LABEL_COLS,
    ToxicCommentTransformerDataset,
    build_tokenizer,
)
from toxic_clf.metrics import compute_metrics
from toxic_clf.mlflow_utils import (
    log_metrics_dict,
    log_params_dict,
    setup_experiment,
    start_run,
)

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.environ.get("TOXIC_CLF_DATA_DIR", "data/processed"))
ARTIFACTS_DIR = Path("models/artifacts/distilbert")
NUM_LABELS = len(LABEL_COLS)

SMOKE_TEST_TRAIN_SAMPLES = 200
SMOKE_TEST_VAL_SAMPLES = 50


def set_seed(seed: int) -> None:
    """Seed random, numpy, and torch for reproducible init/shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_and_tokenizer(max_length: int, model_name: str, smoke_test: bool = False):
    """Load processed train/val DataFrames and wrap as transformer datasets.

    smoke_test=True truncates to a tiny subset for a fast pipeline check --
    results are not meaningful in that mode, only the absence of crashes is.
    """
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")

    if smoke_test:
        train_df = train_df.head(SMOKE_TEST_TRAIN_SAMPLES)
        val_df = val_df.head(SMOKE_TEST_VAL_SAMPLES)
        logger.warning(
            "SMOKE TEST MODE: using %d train / %d val rows, 1 epoch only. "
            "Results are not meaningful -- this only verifies the pipeline runs.",
            len(train_df),
            len(val_df),
        )

    tokenizer = build_tokenizer(model_name)
    train_dataset = ToxicCommentTransformerDataset(
        train_df, tokenizer, max_length=max_length
    )
    val_dataset = ToxicCommentTransformerDataset(
        val_df, tokenizer, max_length=max_length
    )
    return train_dataset, val_dataset, tokenizer


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Per-label pos_weight = num_negatives / num_positives, for
    BCEWithLogitsLoss. Same formulation as train_bilstm.py -- duplicated
    rather than shared for now; small pure function, low risk, not worth
    touching already-verified Phase 7 code mid-Phase-8. Candidate for
    consolidation in Phase 15 cleanup."""
    num_pos = labels.sum(axis=0)
    num_neg = labels.shape[0] - num_pos
    pos_weight = np.where(num_pos > 0, num_neg / np.maximum(num_pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_model(
    model_name: str, num_labels: int = NUM_LABELS
) -> DistilBertForSequenceClassification:
    """Load pretrained DistilBERT with a fresh classification head.

    num_labels=6 raw logits are returned -- loss/sigmoid handled externally
    (BCEWithLogitsLoss with pos_weight during training, sigmoid at eval),
    matching the pattern from train_bilstm.py.
    """
    return DistilBertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a loader, returning (y_true, y_prob) numpy arrays."""
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train(run_name: str = "distilbert-baseline", smoke_test: bool = False) -> dict:
    """Fine-tune DistilBERT with early stopping, log to MLflow, save the
    best checkpoint (model + tokenizer) and export a metrics JSON for
    re-import into local MLflow when training ran on a remote GPU runtime.
    """
    params = load_params()
    t_params = params["transformer"]
    set_seed(t_params["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    logger.info("Loading data and tokenizer")
    train_dataset, val_dataset, tokenizer = load_data_and_tokenizer(
        t_params["max_length"], t_params["model_name"], smoke_test=smoke_test
    )
    logger.info("Train size: %d, Val size: %d", len(train_dataset), len(val_dataset))

    batch_size = t_params["batch_size"]
    max_epochs = 1 if smoke_test else t_params["max_epochs"]
    patience = t_params["patience"]

    num_workers = 2
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(t_params["model_name"]).to(device)
    pos_weight = compute_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_params["learning_rate"],
        weight_decay=t_params["weight_decay"],
    )

    num_training_steps = len(train_loader) * max_epochs
    num_warmup_steps = int(t_params["warmup_ratio"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_macro_f1 = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0

    setup_experiment(params["mlflow"]["experiment_name"])
    with start_run(run_name):
        log_params_dict({"transformer": t_params})

        for epoch in range(1, max_epochs + 1):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )
            y_true, y_prob = evaluate(model, val_loader, device)
            metrics = compute_metrics(y_true, y_prob, label_names=LABEL_COLS)
            elapsed = time.time() - start_time

            logger.info(
                "Epoch %d/%d - train_loss=%.4f val_macro_f1=%.4f (%.1fs)",
                epoch,
                max_epochs,
                train_loss,
                metrics["macro_f1"],
                elapsed,
            )

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        patience,
                    )
                    break

        log_metrics_dict(best_metrics)
        import mlflow

        mlflow.log_metric("best_epoch", best_epoch)
        logger.info(
            "Best validation macro F1: %.4f (epoch %d)", best_macro_f1, best_epoch
        )

        model.load_state_dict(best_state)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ARTIFACTS_DIR)
        tokenizer.save_pretrained(ARTIFACTS_DIR)
        logger.info("Saved best model + tokenizer to %s", ARTIFACTS_DIR)

        export_path = ARTIFACTS_DIR / "metrics_export.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "params": {"transformer": t_params},
                    "metrics": best_metrics,
                    "best_epoch": best_epoch,
                },
                f,
                indent=2,
            )
        logger.info("Exported metrics to %s", export_path)

    return best_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
