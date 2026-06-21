"""Tests for src.toxic_clf.config."""

import pytest

from toxic_clf.config import load_params


def test_load_params_returns_dict():
    params = load_params()
    assert isinstance(params, dict)


def test_load_params_has_expected_top_level_keys():
    params = load_params()
    for key in ["split", "tfidf", "bilstm", "transformer", "mlflow"]:
        assert key in params


def test_load_params_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_params("nonexistent_params.yaml")


def test_load_params_values_match_locked_in_decisions():
    params = load_params()
    assert params["split"]["val_size"] == 0.1
    assert params["split"]["random_state"] == 42
    assert params["tfidf"]["max_features"] == 50000
    assert params["bilstm"]["vocab_size"] == 50000
    assert params["bilstm"]["max_len"] == 200
    assert params["transformer"]["max_length"] == 256
