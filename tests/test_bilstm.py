"""Tests for the BiLSTM model and training utilities (Phase 7).

Fast tests on tiny synthetic data -- not full training. Verifies shapes,
the pos_weight computation, and that one training step plus evaluation
run without crashing and produce well-formed outputs.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from toxic_clf.bilstm_model import BiLSTMClassifier
from toxic_clf.train_bilstm import compute_pos_weight, evaluate, train_one_epoch


def test_bilstm_forward_pass_output_shape():
    model = BiLSTMClassifier(vocab_size=100, embedding_dim=16, hidden_dim=8)
    input_ids = torch.randint(0, 100, (4, 20))  # batch=4, seq_len=20
    logits = model(input_ids)
    assert logits.shape == (4, 6)


def test_compute_pos_weight_shape_and_values():
    labels = np.array(
        [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype="float32",
    )
    pos_weight = compute_pos_weight(labels)
    assert pos_weight.shape == (6,)
    assert pos_weight[0].item() == 0.5  # 2 pos, 1 neg -> 1/2
    assert pos_weight[1].item() == 2.0  # 1 pos, 2 neg -> 2/1


def test_train_one_epoch_and_evaluate_run_without_crashing():
    model = BiLSTMClassifier(vocab_size=50, embedding_dim=8, hidden_dim=8)
    device = torch.device("cpu")

    input_ids = torch.randint(0, 50, (20, 10))
    labels = torch.randint(0, 2, (20, 6)).float()
    loader = DataLoader(TensorDataset(input_ids, labels), batch_size=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    assert isinstance(loss, float)

    y_true, y_prob = evaluate(model, loader, device)
    assert y_true.shape == (20, 6)
    assert y_prob.shape == (20, 6)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()
