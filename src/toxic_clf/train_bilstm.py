"""BiLSTM training with early stopping (Phase 7).

Mirrors train_logreg.py's structure: load data -> build model -> train ->
evaluate -> log to MLflow -> save best checkpoint. Reuses the same shared
compute_metrics() and 0.5 threshold so results are directly comparable to
the Phase 6 Logistic Regression baseline.
"""

import copy
import logging
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from toxic_clf.bilstm_model import BiLSTMClassifier
from toxic_clf.config import load_params
from toxic_clf.datasets_bilstm import LABEL_COLS, ToxicCommentDataset
from toxic_clf.metrics import compute_metrics
from toxic_clf.mlflow_utils import (
    log_metrics_dict,
    log_params_dict,
    setup_experiment,
    start_run,
)
from toxic_clf.vocab import PAD_IDX, load_vocab

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("models/artifacts")
VOCAB_PATH = ARTIFACTS_DIR / "bilstm_vocab.json"


def set_seed(seed: int) -> None:
    """Seed random, numpy, and torch for reproducible weight init/shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data_and_vocab(max_len: int):
    """Load processed train/val DataFrames, the BiLSTM vocab, and wrap them
    as ToxicCommentDataset instances."""
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv")
    vocab = load_vocab(str(VOCAB_PATH))

    train_dataset = ToxicCommentDataset(train_df, vocab, max_len=max_len)
    val_dataset = ToxicCommentDataset(val_df, vocab, max_len=max_len)
    return train_dataset, val_dataset, vocab


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Per-label pos_weight = num_negatives / num_positives, for
    BCEWithLogitsLoss. Mirrors what class_weight='balanced' does for
    Logistic Regression, but in the neural-loss formulation."""
    num_pos = labels.sum(axis=0)
    num_neg = labels.shape[0] - num_pos
    pos_weight = np.where(num_pos > 0, num_neg / np.maximum(num_pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def build_model(vocab: dict, bilstm_params: dict) -> BiLSTMClassifier:
    return BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=bilstm_params["embedding_dim"],
        hidden_dim=bilstm_params["hidden_dim"],
        dropout=bilstm_params["dropout"],
        padding_idx=PAD_IDX,
    )


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Run the model over a loader, returning (y_true, y_prob) numpy arrays."""
    model.eval()
    all_labels, all_probs = [], []
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        logits = model(input_ids)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train(run_name: str = "bilstm-baseline") -> dict:
    """Train the BiLSTM with early stopping, log the best checkpoint to
    MLflow, and save it to disk. Returns the best validation metrics dict."""
    params = load_params()
    bilstm_params = params["bilstm"]
    set_seed(bilstm_params["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    logger.info("Loading data and vocab")
    train_dataset, val_dataset, vocab = load_data_and_vocab(bilstm_params["max_len"])
    logger.info(
        "Train size: %d, Val size: %d, Vocab size: %d",
        len(train_dataset),
        len(val_dataset),
        len(vocab),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=bilstm_params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bilstm_params["batch_size"], shuffle=False
    )

    model = build_model(vocab, bilstm_params).to(device)
    pos_weight = compute_pos_weight(train_dataset.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=bilstm_params["learning_rate"])

    best_macro_f1 = -1.0
    best_state = None
    best_metrics = None
    best_epoch = 0
    epochs_without_improvement = 0

    setup_experiment(params["mlflow"]["experiment_name"])
    with start_run(run_name):
        log_params_dict({"bilstm": bilstm_params})

        for epoch in range(1, bilstm_params["max_epochs"] + 1):
            start_time = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            y_true, y_prob = evaluate(model, val_loader, device)
            metrics = compute_metrics(y_true, y_prob, label_names=LABEL_COLS)
            elapsed = time.time() - start_time

            logger.info(
                "Epoch %d/%d - train_loss=%.4f val_macro_f1=%.4f (%.1fs)",
                epoch,
                bilstm_params["max_epochs"],
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
                if epochs_without_improvement >= bilstm_params["patience"]:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch,
                        bilstm_params["patience"],
                    )
                    break

        log_metrics_dict(best_metrics)
        mlflow.log_metric("best_epoch", best_epoch)
        logger.info(
            "Best validation macro F1: %.4f (epoch %d)", best_macro_f1, best_epoch
        )

        model.load_state_dict(best_state)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACTS_DIR / "bilstm_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info("Saved best model to %s", model_path)

    return best_metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
