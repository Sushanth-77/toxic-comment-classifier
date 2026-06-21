"""Tests for src.toxic_clf.datasets_bilstm."""

import pandas as pd
import torch

from toxic_clf.datasets_bilstm import LABEL_COLS, ToxicCommentDataset

VOCAB = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, "bad": 4}


def _make_df():
    return pd.DataFrame(
        {
            "clean_heavy": ["hello world", "bad bad bad"],
            "toxic": [0, 1],
            "severe_toxic": [0, 0],
            "obscene": [0, 0],
            "threat": [0, 0],
            "insult": [0, 1],
            "identity_hate": [0, 0],
        }
    )


def test_dataset_length():
    ds = ToxicCommentDataset(_make_df(), VOCAB, max_len=5)
    assert len(ds) == 2


def test_dataset_getitem_returns_tensors():
    ds = ToxicCommentDataset(_make_df(), VOCAB, max_len=5)
    input_ids, labels = ds[0]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.shape == (5,)
    assert labels.shape == (len(LABEL_COLS),)


def test_dataset_labels_match_dataframe():
    ds = ToxicCommentDataset(_make_df(), VOCAB, max_len=5)
    _, labels = ds[1]
    assert labels.tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_dataset_handles_nan_text():
    df = _make_df()
    df.loc[0, "clean_heavy"] = None
    ds = ToxicCommentDataset(df, VOCAB, max_len=5)
    input_ids, _ = ds[0]
    assert input_ids.tolist() == [0, 0, 0, 0, 0]  # all PAD
