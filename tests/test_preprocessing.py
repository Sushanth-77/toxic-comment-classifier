"""Tests for src.toxic_clf.preprocessing."""

import pandas as pd
import pytest

from toxic_clf.preprocessing import clean_dataframe, heavy_clean, light_clean

# ---- heavy_clean ----


def test_heavy_clean_lowercases():
    assert heavy_clean("HELLO World") == "hello world"


def test_heavy_clean_removes_urls():
    text = "Check this out http://example.com/page now"
    result = heavy_clean(text)
    assert "http" not in result
    assert "example" not in result


def test_heavy_clean_removes_ip_addresses():
    text = "Contact 192.168.1.1 for details"
    result = heavy_clean(text)
    assert "192" not in result


def test_heavy_clean_keeps_apostrophes_strips_other_punctuation():
    text = "Don't do that! It's wrong, seriously??"
    result = heavy_clean(text)
    assert "'" in result
    assert "!" not in result
    assert "," not in result
    assert "?" not in result


def test_heavy_clean_collapses_whitespace():
    text = "too    many     spaces\n\nand newlines"
    result = heavy_clean(text)
    assert "  " not in result


def test_heavy_clean_handles_empty_and_non_string():
    assert heavy_clean("") == ""
    assert heavy_clean(None) == ""


# ---- light_clean ----


def test_light_clean_preserves_case_and_punctuation():
    text = "Don't Yell At Me!"
    result = light_clean(text)
    assert result == "Don't Yell At Me!"


def test_light_clean_decodes_html_entities():
    text = "5 &amp; 3 &lt; 10"
    result = light_clean(text)
    assert "&amp;" not in result
    assert "&" in result


def test_light_clean_collapses_whitespace():
    text = "line one\n\n\nline two     end"
    result = light_clean(text)
    assert "\n\n" not in result
    assert "   " not in result


def test_light_clean_handles_empty_and_non_string():
    assert light_clean("") == ""
    assert light_clean(None) == ""


# ---- clean_dataframe ----


def test_clean_dataframe_heavy_adds_column():
    df = pd.DataFrame({"comment_text": ["HELLO!! Visit http://x.com"]})
    result = clean_dataframe(df, mode="heavy")
    assert "clean_text" in result.columns
    assert result["clean_text"].iloc[0] == "hello visit"
    assert result["comment_text"].iloc[0] == "HELLO!! Visit http://x.com"


def test_clean_dataframe_light_adds_column():
    df = pd.DataFrame({"comment_text": ["Hello!! &amp; goodbye"]})
    result = clean_dataframe(df, mode="light")
    assert result["clean_text"].iloc[0] == "Hello!! & goodbye"


def test_clean_dataframe_invalid_mode_raises():
    df = pd.DataFrame({"comment_text": ["text"]})
    with pytest.raises(ValueError):
        clean_dataframe(df, mode="medium")


def test_clean_dataframe_missing_column_raises():
    df = pd.DataFrame({"other_col": ["text"]})
    with pytest.raises(KeyError):
        clean_dataframe(df, mode="heavy")
