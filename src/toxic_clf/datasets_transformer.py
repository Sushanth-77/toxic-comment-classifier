"""PyTorch Dataset for the DistilBERT model."""

import logging

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

logger = logging.getLogger(__name__)

LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def build_tokenizer(
    model_name: str = TRANSFORMER_MODEL_NAME,
) -> DistilBertTokenizerFast:
    """Load the pretrained DistilBERT tokenizer.

    Downloads and caches tokenizer files from the HuggingFace Hub on
    first call (requires network access once; cached locally after).
    """
    logger.info("Loading tokenizer: %s", model_name)
    return DistilBertTokenizerFast.from_pretrained(model_name)


class ToxicCommentTransformerDataset(Dataset):
    """Wraps a cleaned DataFrame for DistilBERT training/eval.

    Expects a `clean_light` text column and the 6 label columns. Uses
    light_clean'd text (case + punctuation preserved) since the
    tokenizer relies on natural text to produce meaningful subwords.

    Tokenization happens at __getitem__ time, not precomputed -- keeps
    memory usage manageable for large splits.
    """

    def __init__(
        self,
        df,
        tokenizer: DistilBertTokenizerFast,
        max_length: int = MAX_LENGTH,
        text_col: str = "clean_light",
    ):
        self.texts = df[text_col].fillna("").tolist()
        self.labels = df[LABEL_COLS].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        return item
