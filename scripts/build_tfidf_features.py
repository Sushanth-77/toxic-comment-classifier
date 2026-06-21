"""One-time script: fit TF-IDF vectorizer on train, transform train/val,
save the fitted vectorizer and sparse feature matrices to disk.

Run with: python scripts/build_tfidf_features.py
"""

import logging

import joblib
import pandas as pd
from scipy import sparse

from toxic_clf.config import load_params
from toxic_clf.features_tfidf import build_tfidf_vectorizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

VECTORIZER_PATH = "models/artifacts/tfidf_vectorizer.joblib"
TRAIN_FEATURES_PATH = "data/processed/tfidf_train.npz"
VAL_FEATURES_PATH = "data/processed/tfidf_val.npz"


def main() -> None:
    params = load_params()

    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")

    # heavy_clean() can legitimately produce "" for comments that are only
    # URLs/digits/punctuation. CSV round-trips serialize "" as an empty
    # field, which pandas reads back as NaN. Restore it to "" here.
    train_df["clean_heavy"] = train_df["clean_heavy"].fillna("")
    val_df["clean_heavy"] = val_df["clean_heavy"].fillna("")

    vectorizer = build_tfidf_vectorizer(params["tfidf"])

    logger.info("Fitting vectorizer on train clean_heavy text (%d rows)", len(train_df))
    X_train = vectorizer.fit_transform(train_df["clean_heavy"])

    logger.info("Transforming val clean_heavy text (%d rows)", len(val_df))
    X_val = vectorizer.transform(val_df["clean_heavy"])

    logger.info("X_train shape: %s, X_val shape: %s", X_train.shape, X_val.shape)

    joblib.dump(vectorizer, VECTORIZER_PATH)
    sparse.save_npz(TRAIN_FEATURES_PATH, X_train)
    sparse.save_npz(VAL_FEATURES_PATH, X_val)

    logger.info("Saved vectorizer to %s", VECTORIZER_PATH)
    logger.info("Saved train/val sparse features to data/processed/")


if __name__ == "__main__":
    main()
