"""Tests for src.toxic_clf.datasets_transformer.

Uses the real pretrained tokenizer (downloads/caches on first run) --
this is the only sensible way to test real tokenization behavior, and
matches what Phase 8 training will actually use.
"""

import pandas as pd
import pytest
import torch

from toxic_clf.datasets_transformer import (
    LABEL_COLS,
    MAX_LENGTH,
    ToxicCommentTransformerDataset,
    build_tokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


def _make_df():
    return pd.DataFrame(
        {
            "clean_light": ["Hello, world!", "You are terrible and disgusting."],
            "toxic": [0, 1],
            "severe_toxic": [0, 0],
            "obscene": [0, 0],
            "threat": [0, 0],
            "insult": [0, 1],
            "identity_hate": [0, 0],
        }
    )


def test_dataset_length(tokenizer):
    ds = ToxicCommentTransformerDataset(_make_df(), tokenizer, max_length=16)
    assert len(ds) == 2


def test_dataset_getitem_returns_expected_keys_and_shapes(tokenizer):
    ds = ToxicCommentTransformerDataset(_make_df(), tokenizer, max_length=16)
    item = ds[0]
    assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert item["labels"].shape == (len(LABEL_COLS),)


def test_dataset_default_max_length_is_256(tokenizer):
    ds = ToxicCommentTransformerDataset(_make_df(), tokenizer)
    item = ds[0]
    assert item["input_ids"].shape == (MAX_LENGTH,)


def test_dataset_labels_match_dataframe(tokenizer):
    ds = ToxicCommentTransformerDataset(_make_df(), tokenizer, max_length=16)
    item = ds[1]
    assert item["labels"].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_dataset_attention_mask_reflects_padding(tokenizer):
    ds = ToxicCommentTransformerDataset(_make_df(), tokenizer, max_length=16)
    item = ds[0]  # "Hello, world!" is short -> should have padding
    assert item["attention_mask"].sum().item() < 16


def test_dataset_handles_nan_text(tokenizer):
    df = _make_df()
    df.loc[0, "clean_light"] = None
    ds = ToxicCommentTransformerDataset(df, tokenizer, max_length=16)
    item = ds[0]
    assert isinstance(item["input_ids"], torch.Tensor)
