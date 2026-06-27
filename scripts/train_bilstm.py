"""CLI entrypoint for training the BiLSTM baseline."""

import logging

from toxic_clf.train_bilstm import train

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
