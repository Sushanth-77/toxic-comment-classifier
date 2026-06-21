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


def build_tfidf_vectorizer(config: dict | None = None) -> TfidfVectorizer:
    """Return a TfidfVectorizer, optionally configured from params.yaml.

    If config is provided (e.g. params["tfidf"] from load_params()), it
    overrides TFIDF_CONFIG. ngram_range is normalized to a tuple since
    YAML deserializes it as a list.
    """
    merged = {**TFIDF_CONFIG, **(config or {})}
    if "ngram_range" in merged:
        merged["ngram_range"] = tuple(merged["ngram_range"])
    logger.info("Building TfidfVectorizer with config: %s", merged)
    return TfidfVectorizer(**merged)
