"""Data loading and splitting utilities for toxic comment classification."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

RAW_TRAIN_PATH = "data/raw/train.csv"
VAL_SIZE = 0.1
RANDOM_STATE = 42


def load_raw_train(path: str = RAW_TRAIN_PATH) -> pd.DataFrame:
    """Load the raw Kaggle training CSV."""
    logger.info("Loading raw training data from %s", path)
    return pd.read_csv(path)


def train_val_split(
    df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into train/val sets, stratified on whether any label is positive.

    True multilabel stratification (e.g. iterative-stratification) is not
    used here -- it's unnecessary complexity for this project. Stratifying
    on a single "any label present" boolean preserves the overall
    clean/toxic ratio across both splits, which is what matters for
    downstream imbalance handling (pos_weight, thresholds).
    """
    has_any_label = (df[LABEL_COLS].sum(axis=1) > 0).astype(int)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=has_any_label,
    )
    logger.info(
        "Split %d rows into train=%d, val=%d", len(df), len(train_df), len(val_df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
