"""PyTorch Dataset for the BiLSTM model."""

import logging

import torch
from torch.utils.data import Dataset

from toxic_clf.vocab import encode

logger = logging.getLogger(__name__)

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


class ToxicCommentDataset(Dataset):
    """Wraps a cleaned DataFrame for BiLSTM training/eval.

    Expects a `clean_heavy` text column and the 6 label columns. Texts
    are encoded to fixed-length integer sequences at __getitem__ time
    (not precomputed), keeping memory usage manageable for large splits.
    """

    def __init__(self, df, vocab, max_len: int = 200, text_col: str = "clean_heavy"):
        self.texts = df[text_col].fillna("").tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = encode(self.texts[idx], self.vocab, self.max_len)
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, labels
