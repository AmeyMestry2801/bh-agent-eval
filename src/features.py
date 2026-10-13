"""
Engineered text features for M3.

All features are computed from the text and the channel only. Nothing here may
read labels, week, template_family or any generator-internal column.
The OOV vocabulary is learned in fit(), so inside a CV pipeline it only ever
sees the training fold.

ALL DATA IS SYNTHETIC.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_TOKEN = re.compile(r"[a-z']+")

NEGATION_WORDS = {
    "not", "no", "never", "none", "nothing", "nobody", "cannot", "without", "hardly",
    "dont", "doesnt", "didnt", "cant", "wont", "wouldnt", "isnt", "arent", "wasnt",
}
HEDGE_WORDS = {"maybe", "perhaps", "guess", "probably", "might", "possibly", "unsure", "dunno"}
HEDGE_PHRASES = re.compile(
    r"\b(?:kind of|sort of|not sure|could be wrong|or something|i don'?t know)\b", re.I)
FIRST_PERSON = {"i", "i'm", "im", "i've", "ive", "i'd", "i'll", "me", "my", "myself", "mine"}

STAT_NAMES = ["negation_count", "hedge_count", "first_person_count",
              "n_words", "is_voice", "oov_rate"]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _is_negation(tok: str) -> bool:
    return tok in NEGATION_WORDS or tok.endswith("n't")


class TextStats(BaseEstimator, TransformerMixin):
    """Six engineered features from a DataFrame with columns ['text', 'channel']."""

    def fit(self, X: pd.DataFrame, y=None) -> "TextStats":
        vocab: set[str] = set()
        for text in X["text"]:
            vocab.update(tokenize(text))
        self.vocab_ = vocab
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows = []
        for text, channel in zip(X["text"], X["channel"]):
            toks = tokenize(text)
            n = max(len(toks), 1)
            rows.append([
                sum(_is_negation(t) for t in toks),
                sum(t in HEDGE_WORDS for t in toks) + len(HEDGE_PHRASES.findall(text)),
                sum(t in FIRST_PERSON for t in toks),
                len(toks),
                1.0 if channel == "voice" else 0.0,
                sum(t not in self.vocab_ for t in toks) / n,
            ])
        return np.asarray(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(STAT_NAMES)
