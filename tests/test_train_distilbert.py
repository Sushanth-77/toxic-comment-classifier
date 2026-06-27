"""Tests for DistilBERT training utilities (Phase 8).

test_build_model and the training-step test load the real pretrained
DistilBERT classification head (downloads once, cached after -- same
network dependency test_datasets_transformer.py already established for
the tokenizer). These run slower than the other test files; that's expected.
"""

import numpy as np
import torch

from toxic_clf.train_distilbert import (
    build_model,
    compute_pos_weight,
    evaluate,
    train_one_epoch,
)

MODEL_NAME = "distilbert-base-uncased"


def test_compute_pos_weight_shape_and_values():
    labels = np.array(
        [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
        dtype="float32",
    )
    pos_weight = compute_pos_weight(labels)
    assert pos_weight.shape == (6,)
    assert pos_weight[0].item() == 0.5
    assert pos_weight[1].item() == 2.0


def test_build_model_has_correct_num_labels():
    model = build_model(MODEL_NAME, num_labels=6)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert logits.shape == (2, 6)


def test_train_one_epoch_and_evaluate_run_without_crashing():
    model = build_model(MODEL_NAME, num_labels=6)
    device = torch.device("cpu")

    input_ids = torch.randint(0, 1000, (8, 16))
    attention_mask = torch.ones(8, 16, dtype=torch.long)
    labels = torch.randint(0, 2, (8, 6)).float()

    class TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, idx):
            return {
                "input_ids": input_ids[idx],
                "attention_mask": attention_mask[idx],
                "labels": labels[idx],
            }

    loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=2
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    loss = train_one_epoch(model, loader, optimizer, scheduler, criterion, device)
    assert isinstance(loss, float)

    y_true, y_prob = evaluate(model, loader, device)
    assert y_true.shape == (8, 6)
    assert y_prob.shape == (8, 6)
    assert (y_prob >= 0).all() and (y_prob <= 1).all()
