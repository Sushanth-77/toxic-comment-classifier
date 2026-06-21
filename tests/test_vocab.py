"""Tests for src.toxic_clf.vocab."""

from toxic_clf.vocab import (
    PAD_IDX,
    UNK_IDX,
    build_vocab,
    encode,
    load_vocab,
    save_vocab,
    tokenize,
)


def test_tokenize_splits_on_whitespace():
    assert tokenize("hello world foo") == ["hello", "world", "foo"]


def test_tokenize_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_build_vocab_reserves_pad_and_unk():
    texts = ["hello world", "hello there", "foo bar baz"]
    vocab = build_vocab(texts, vocab_size=10)
    assert vocab["<PAD>"] == PAD_IDX
    assert vocab["<UNK>"] == UNK_IDX


def test_build_vocab_prioritizes_frequent_tokens():
    texts = ["a a a a", "a a b b", "c"]
    vocab = build_vocab(texts, vocab_size=4)  # 2 slots besides PAD/UNK
    assert "a" in vocab
    assert "b" in vocab
    assert "c" not in vocab  # least frequent, should be dropped


def test_encode_pads_short_sequences():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}
    ids = encode("hello world", vocab, max_len=5)
    assert ids == [2, 3, 0, 0, 0]


def test_encode_truncates_long_sequences():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    ids = encode("hello hello hello hello", vocab, max_len=2)
    assert ids == [2, 2]


def test_encode_maps_unknown_tokens_to_unk():
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    ids = encode("hello mystery", vocab, max_len=2)
    assert ids == [2, UNK_IDX]


def test_save_and_load_vocab_roundtrip(tmp_path):
    vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2}
    path = tmp_path / "vocab.json"
    save_vocab(vocab, str(path))
    loaded = load_vocab(str(path))
    assert loaded == vocab
