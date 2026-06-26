"""CLI entrypoint for training the Logistic Regression + TF-IDF baseline."""

import logging

from toxic_clf.train_logreg import train

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
