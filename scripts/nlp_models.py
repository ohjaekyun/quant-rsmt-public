"""Swappable Hugging Face financial-sentiment classifiers (docs/MODELING.md 64). Research only.

    .venv/bin/python scripts/nlp_models.py bench --model finbert     # throughput on this machine

A model is a pinned (repo, revision) pair, run locally (Apple GPU when available, else CPU); weights live in the
Hugging Face cache outside the repo and no remote code is ever executed. To swap a model, add a Spec to MODELS and pass
its key - results are stored per (model, revision), so two models never mix. Sentence scores are cached on disk by
text hash, so re-runs only score new text.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402


@dataclass(frozen=True)
class Spec:
    key: str
    repo: str
    revision: str
    lang: str
    licence: str
    note: str

    @property
    def tag(self) -> str:
        return f"{self.key}@{self.revision[:8]}"


MODELS = {
    "finbert": Spec(
        "finbert",
        "ProsusAI/finbert",
        "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031",
        "en",
        "CC BY-NC-SA 4.0 (non-commercial)",
        "BERT further trained on financial text, fine-tuned on Financial PhraseBank (Araci 2019)",
    ),
    "deberta_fin": Spec(
        "deberta_fin",
        "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
        "9e10915c245a80a89b18d1ac51350e093c7bb35a",
        "en",
        "MIT",
        "DeBERTa-v3 fine-tuned on financial news sentiment (commercial-friendly replacement)",
    ),
    "kr_finbert": Spec(
        "kr_finbert",
        "snunlp/KR-FinBert-SC",
        "f8586286cc3161fb648e9fee09a456069fd846d0",
        "ko",
        "not stated on the model card (personal research only)",
        "KR-BERT further trained on Korean economic news + analyst reports, 50k labelled sentences (SNU)",
    ),
}
DEFAULT = {"en": "finbert", "ko": "kr_finbert"}
_LABELS = {"positive": 0, "negative": 1, "neutral": 2}

_EN_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“(])")
_KO_SPLIT = re.compile(r"(?<=[.!?。])\s+")


def sentences(text: str, lang: str, max_n: int | None = None, min_len: int = 12, max_chars: int = 600) -> list[str]:
    """Rule-based split: paragraphs (blank lines), then terminal punctuation; single line breaks are joined."""
    out = []
    for para in re.split(r"\n\s*\n", text or ""):
        t = re.sub(r"\s+", " ", para).strip()
        for part in (_EN_SPLIT if lang == "en" else _KO_SPLIT).split(t):
            part = part.strip()
            if len(part) >= min_len and re.search(r"[A-Za-z가-힣]", part):
                out.append(part[:max_chars])
                if max_n and len(out) >= max_n:
                    return out
    return out


def device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


class Store:
    """Sentence -> (pos, neg, neu) probabilities, one SQLite file per model revision."""

    def __init__(self, spec: Spec, settings=None):
        if settings is None:
            from quantcio.config import load_settings

            settings = load_settings()
        d = kd.root(settings, "v25") / "scores"
        d.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(d / f"{spec.key}_{spec.revision[:8]}.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS s (h TEXT PRIMARY KEY, pos REAL, neg REAL, neu REAL)")

    @staticmethod
    def h(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, hs: list[str]) -> dict[str, tuple[float, float, float]]:
        out = {}
        for i in range(0, len(hs), 900):
            q = hs[i : i + 900]
            rows = self.db.execute(f"SELECT h, pos, neg, neu FROM s WHERE h IN ({','.join('?' * len(q))})", q)
            out.update({r[0]: r[1:] for r in rows})
        return out

    def put(self, rows: list[tuple[str, float, float, float]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?, ?, ?)", rows)
        self.db.commit()


class Classifier:
    def __init__(self, key: str, settings=None, max_len: int = 128, batch: int = 64, dev: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.spec = MODELS[key]
        self.max_len, self.batch = max_len, batch
        self.dev = dev or device()
        s = self.spec
        self.tok = AutoTokenizer.from_pretrained(s.repo, revision=s.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(s.repo, revision=s.revision)
        self.model.eval().to(self.dev)
        id2 = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.order = [next(i for i, lab in id2.items() if lab.startswith(name)) for name in _LABELS]
        self.store = Store(s, settings)
        self.torch = torch

    def _infer(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        out = np.zeros((len(texts), 3), dtype=np.float32)
        order = np.argsort([len(t) for t in texts])
        for i in range(0, len(texts), self.batch):
            ix = order[i : i + self.batch]
            enc = self.tok(
                [texts[j] for j in ix], padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
            ).to(self.dev)
            with torch.inference_mode():
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
            out[ix] = p[:, self.order]
        return out

    def score(self, texts: list[str], progress=None, chunk: int = 4096) -> np.ndarray:
        """(n, 3) probabilities [positive, negative, neutral]; cached sentences are not re-scored."""
        hs = [Store.h(t) for t in texts]
        have = self.store.get(list(dict.fromkeys(hs)))
        todo = list(dict.fromkeys(t for t, h in zip(texts, hs, strict=True) if h not in have))
        t0 = time.time()
        for i in range(0, len(todo), chunk):
            part = todo[i : i + chunk]
            p = self._infer(part)
            rows = [(Store.h(t), *map(float, r)) for t, r in zip(part, p, strict=True)]
            self.store.put(rows)
            have.update({r[0]: r[1:] for r in rows})
            if progress:
                done, rate = i + len(part), (i + len(part)) / max(time.time() - t0, 1e-9)
                progress(f"{self.spec.key}: {done:,}/{len(todo):,} new sentences ({rate:.0f}/s)")
        return np.array([have[h] for h in hs], dtype=float).reshape(-1, 3)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--model", default="finbert", choices=list(MODELS))
    a = ap.parse_args()
    c = Classifier(a.model)
    lang = c.spec.lang
    demo = (
        ["Revenue rose 12% to a record $4.1 billion, and we raised our full-year outlook.",
         "The company recorded an impairment charge and withdrew its guidance amid weaker demand.",
         "The meeting will be held on Tuesday at 10 a.m. Eastern Time."]
        if lang == "en"
        else ["3분기 영업이익이 전년 대비 35% 증가하며 사상 최대 실적을 기록했다.",
              "수주 부진과 원가 상승으로 적자 전환했다.",
              "정기 주주총회는 3월 20일 본사에서 열린다."]
    )  # fmt: skip
    p = c._infer(demo)
    for t, r in zip(demo, p, strict=True):
        print(f"pos {r[0]:.2f} neg {r[1]:.2f} neu {r[2]:.2f} | {t}")
    texts = [f"{demo[i % 3]} ({i})" for i in range(2048)]
    t0 = time.time()
    c._infer(texts)
    print(f"{c.spec.tag} on {c.dev}: {len(texts) / (time.time() - t0):.0f} sentences/s (max_len {c.max_len})")


if __name__ == "__main__":
    main()
