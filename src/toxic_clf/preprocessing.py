"""Text preprocessing utilities for toxic comment classification.

Two preprocessing paths:
- heavy_clean: aggressive normalization for classical ML (TF-IDF) and the
  BiLSTM model, where vocabulary size and noise reduction matter more than
  preserving exact surface form.
- light_clean: minimal normalization for transformer models (DistilBERT),
  where the tokenizer needs natural text (case, punctuation, contractions)
  to produce meaningful subword tokens.
"""

import html
import logging
import re
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
_NON_ALPHA_APOSTROPHE_PATTERN = re.compile(r"[^a-z\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def heavy_clean(text: str) -> str:
    """Aggressively normalize text for classical ML / BiLSTM models.

    Steps: lowercase, strip URLs and IP addresses, decode HTML entities,
    drop punctuation/digits (keep apostrophes for contractions), collapse
    whitespace.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = html.unescape(text)
    text = text.lower()
    text = _URL_PATTERN.sub(" ", text)
    text = _IP_PATTERN.sub(" ", text)
    text = _NON_ALPHA_APOSTROPHE_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def light_clean(text: str) -> str:
    """Minimally normalize text for transformer models (DistilBERT).

    Steps: decode HTML entities, normalize unicode (NFKC), collapse
    excessive whitespace. Case, punctuation, and contractions are
    preserved since the tokenizer relies on them.
    """
    if not isinstance(text, str) or not text:
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    mode: str = "heavy",
    output_col: str = "clean_text",
) -> pd.DataFrame:
    """Apply heavy_clean or light_clean to a DataFrame column.

    Returns a copy of df with a new column (`output_col`) holding the
    cleaned text. The original text_col is left untouched.
    """
    if mode not in ("heavy", "light"):
        raise ValueError(f"mode must be 'heavy' or 'light', got: {mode!r}")
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' not found in DataFrame columns")

    clean_fn = heavy_clean if mode == "heavy" else light_clean
    logger.info("Cleaning column '%s' using mode='%s'", text_col, mode)

    result = df.copy()
    result[output_col] = result[text_col].map(clean_fn)
    return result
