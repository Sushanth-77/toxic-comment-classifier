"""Tests for src.toxic_clf.features_tfidf."""

from toxic_clf.features_tfidf import build_tfidf_vectorizer

SAMPLE_DOCS = [
    "you are a terrible person and i hate you",
    "thanks for the helpful edit, much appreciated",
    "i will find you and hurt you badly",
    "this article needs more citations please",
]


def test_build_tfidf_vectorizer_returns_configured_instance():
    vec = build_tfidf_vectorizer()
    assert vec.max_features == 50000
    assert vec.ngram_range == (1, 2)
    assert vec.min_df == 2
    assert vec.sublinear_tf is True


def test_fit_transform_produces_expected_shape():
    vec = build_tfidf_vectorizer()
    # min_df=2 requires terms to appear in >=2 docs; use repeated-ish corpus
    docs = SAMPLE_DOCS * 2
    X = vec.fit_transform(docs)
    assert X.shape[0] == len(docs)
    assert X.shape[1] > 0


def test_transform_after_fit_matches_vocab_size():
    vec = build_tfidf_vectorizer()
    docs = SAMPLE_DOCS * 2
    X_train = vec.fit_transform(docs)
    X_new = vec.transform(["a completely new sentence with unseen words"])
    assert X_new.shape[1] == X_train.shape[1]


def test_includes_bigrams_in_vocabulary():
    vec = build_tfidf_vectorizer()
    docs = SAMPLE_DOCS * 2
    vec.fit(docs)
    vocab = vec.get_feature_names_out()
    has_bigram = any(" " in term for term in vocab)
    assert has_bigram
