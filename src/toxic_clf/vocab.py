"""Vocabulary building utilities for the BiLSTM model."""

import json
import logging
from collections import Counter

logger = logging.getLogger(__name__)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def tokenize(text: str) -> list[str]:
    """Whitespace tokenizer.

    Assumes text has already been passed through heavy_clean (lowercased,
    punctuation stripped), so a simple split is sufficient -- no need for
    a heavier tokenizer for the BiLSTM path.
    """
    if not isinstance(text, str) or not text:
        return []
    return text.split()


def build_vocab(texts, vocab_size: int = 50000) -> dict:
    """Build a word->index vocabulary from an iterable of cleaned texts.

    Reserves index 0 for PAD_TOKEN and index 1 for UNK_TOKEN. The
    remaining vocab_size - 2 slots go to the most frequent tokens.
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    most_common = counter.most_common(vocab_size - 2)
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for idx, (token, _) in enumerate(most_common, start=2):
        vocab[token] = idx

    logger.info(
        "Built vocab: %d unique tokens seen, %d kept (incl. PAD/UNK)",
        len(counter),
        len(vocab),
    )
    return vocab


def save_vocab(vocab: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode(text: str, vocab: dict, max_len: int) -> list:
    """Convert text to a fixed-length list of token ids.

    Truncates to max_len tokens, or pads with PAD_IDX if shorter.
    """
    tokens = tokenize(text)
    ids = [vocab.get(tok, UNK_IDX) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids
