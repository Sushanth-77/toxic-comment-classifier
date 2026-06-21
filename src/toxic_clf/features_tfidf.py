"""TF-IDF feature extraction for the classical ML baseline."""

import logging

from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

TFIDF_CONFIG = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}


def build_tfidf_vectorizer() -> TfidfVectorizer:
    """Return a TfidfVectorizer with the project's locked-in config.

    Config choices:
    - max_features=50000: large enough to capture rare toxic vocabulary,
      small enough to keep the LR baseline fast and the artifact small.
    - ngram_range=(1, 2): unigrams + bigrams to catch short toxic phrases.
    - min_df=2: drops singleton tokens (typos, usernames, noise).
    - sublinear_tf=True: log-scales term frequency, dampens repeated-word
      spam within a single comment.
    """
    logger.info("Building TfidfVectorizer with config: %s", TFIDF_CONFIG)
    return TfidfVectorizer(**TFIDF_CONFIG)
