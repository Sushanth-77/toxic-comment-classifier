"""One-time script: build BiLSTM vocabulary from train clean_heavy text
and save it to disk.

Run with: python scripts/build_bilstm_vocab.py
"""

import logging

import pandas as pd

from toxic_clf.config import load_params
from toxic_clf.vocab import build_vocab, save_vocab

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

VOCAB_PATH = "models/artifacts/bilstm_vocab.json"


def main() -> None:
    params = load_params()
    vocab_size = params["bilstm"]["vocab_size"]

    train_df = pd.read_csv("data/processed/train.csv")
    train_df["clean_heavy"] = train_df["clean_heavy"].fillna("")

    vocab = build_vocab(train_df["clean_heavy"], vocab_size=vocab_size)
    save_vocab(vocab, VOCAB_PATH)
    logger.info("Saved vocab (%d tokens) to %s", len(vocab), VOCAB_PATH)


if __name__ == "__main__":
    main()
