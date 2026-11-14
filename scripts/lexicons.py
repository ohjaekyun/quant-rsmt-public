"""Sentiment word lists for text features, behind one small interface so a list can be swapped without touching the
feature code (docs/MODELING.md 58.2). Research only.

    lex = load_lexicon("lm")                     # Loughran-McDonald master dictionary (cache file, not in git)
    lex = load_lexicon("csv:/path/words.csv")    # any replacement list: columns word, category
    counts(text, lex)                            # {"n_words": ..., "negative": ..., "positive": ..., ...}

Licence note: the Loughran-McDonald lists are free for academic research; commercial use needs a licence from the
authors (loughranmcdonald@gmail.com). The project uses them for personal, non-commercial research (user decision
2026-10-14). The file is kept in the git-ignored cache (``LM_PATH``), never committed; switching to another list is a
``--lexicon csv:...`` argument, and every feature file / study snapshot records the list's name and version.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

CATEGORIES = ("negative", "positive", "uncertainty", "litigious")
LM_FILE = "LM_MasterDictionary_1993-2023.csv"
_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


@dataclass(frozen=True)
class Lexicon:
    name: str
    version: str
    categories: dict[str, frozenset[str]] = field(default_factory=dict)
    note: str = ""

    @property
    def tag(self) -> str:
        return f"{self.name}:{self.version}"


def lm_path(settings) -> Path:
    return Path(settings.cache_path) / "alt" / "keyed" / "v22" / "lexicon" / LM_FILE


def load_lm(path: Path) -> Lexicon:
    """Loughran-McDonald master dictionary: a word belongs to a category when its column holds a year > 0 (a negative
    year marks a removal)."""
    df = pd.read_csv(path, usecols=["Word", "Negative", "Positive", "Uncertainty", "Litigious"])
    words = df["Word"].astype(str).str.upper()
    cats = {c.lower(): frozenset(words[df[c] > 0]) for c in ("Negative", "Positive", "Uncertainty", "Litigious")}
    return Lexicon("lm", path.stem.split("_")[-1], cats, "Loughran-McDonald (academic-use licence)")


def load_csv(path: Path) -> Lexicon:
    """Generic replacement list: CSV with columns ``word`` and ``category`` (one of CATEGORIES; others ignored)."""
    df = pd.read_csv(path)
    df["word"] = df["word"].astype(str).str.upper()
    df["category"] = df["category"].astype(str).str.lower()
    cats = {c: frozenset(df.loc[df["category"] == c, "word"]) for c in CATEGORIES}
    return Lexicon(f"csv-{path.stem}", str(int(path.stat().st_mtime)), cats, f"custom list {path.name}")


def load_lexicon(spec: str, settings=None) -> Lexicon:
    if spec == "lm":
        if settings is None:
            from quantcio.config import load_settings

            settings = load_settings()
        p = lm_path(settings)
        if not p.exists():
            raise FileNotFoundError(
                f"Loughran-McDonald 사전이 없습니다: {p} (sraf.nd.edu에서 내려받아 이 경로에 두세요)"
            )
        return load_lm(p)
    if spec.startswith("csv:"):
        return load_csv(Path(spec[4:]))
    raise ValueError(f"unknown lexicon {spec!r} (lm | csv:<path>)")


def counts(text: str, lex: Lexicon) -> dict[str, int]:
    toks = [t.upper() for t in _TOKEN.findall(text or "")]
    c = Counter(toks)
    out = {"n_words": len(toks)}
    for cat, words in lex.categories.items():
        out[cat] = int(sum(n for w, n in c.items() if w in words))
    return out


__all__ = ["CATEGORIES", "Lexicon", "counts", "load_lexicon"]
