"""Tests for src.toxic_clf.data."""

import pandas as pd

from toxic_clf.data import LABEL_COLS, train_val_split


def _make_dummy_df(n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        is_toxic = i % 5 == 0  # 20% toxic, deterministic
        rows.append(
            {
                "id": str(i),
                "comment_text": f"comment number {i}",
                "toxic": int(is_toxic),
                "severe_toxic": 0,
                "obscene": 0,
                "threat": 0,
                "insult": 0,
                "identity_hate": 0,
            }
        )
    return pd.DataFrame(rows)


def test_split_sizes_approximately_correct():
    df = _make_dummy_df(200)
    train_df, val_df = train_val_split(df, val_size=0.1, random_state=42)
    assert len(train_df) == 180
    assert len(val_df) == 20


def test_split_no_overlap():
    df = _make_dummy_df(200)
    train_df, val_df = train_val_split(df)
    train_ids = set(train_df["id"])
    val_ids = set(val_df["id"])
    assert train_ids.isdisjoint(val_ids)
    assert len(train_ids) + len(val_ids) == len(df)


def test_split_is_reproducible_with_same_seed():
    df = _make_dummy_df(200)
    train_1, val_1 = train_val_split(df, random_state=42)
    train_2, val_2 = train_val_split(df, random_state=42)
    assert set(train_1["id"]) == set(train_2["id"])
    assert set(val_1["id"]) == set(val_2["id"])


def test_split_preserves_label_ratio_roughly():
    df = _make_dummy_df(200)
    train_df, val_df = train_val_split(df, val_size=0.1, random_state=42)

    train_ratio = (train_df[LABEL_COLS].sum(axis=1) > 0).mean()
    val_ratio = (val_df[LABEL_COLS].sum(axis=1) > 0).mean()

    # Both splits should be within a reasonable tolerance of the true 20% rate
    assert abs(train_ratio - 0.2) < 0.05
    assert abs(val_ratio - 0.2) < 0.1
