"""Hugging Face sentiment features (docs/MODELING.md 64, pre-declared 2026-10-16). Research only.

    .venv/bin/python scripts/v25_nlp_features.py score --what us_a     # also us_b, us_c, kr_a, all

`score` runs the classifier over every sentence once (cached by text hash per model revision); the study then builds
point-in-time features from the cached scores. Model keys: scripts/nlp_models.py (swap with --en / --ko).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nlp_models as nm  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v22_tone as tone  # noqa: E402
import v24_ceo_data as cd  # noqa: E402
import v24_ceo_features as cf  # noqa: E402
import v25_news_data as nd  # noqa: E402

say = v13.say
LEAD_A, QUOTE_A, LEAD_B, LEAD_KR = 20, 6, 6, 300
NEG_DOC = -0.3
BS_MAX_AGE = 550
DOCS: dict[str, tuple[str, str]] = {
    "nb_tone": ("실적 발표문 앞 20문장 FinBERT 어조", "earnings release lead FinBERT tone"),
    "nb_neg": ("실적 발표문 부정 문장 비율(FinBERT)", "earnings release negative-sentence share"),
    "nb_pos": ("실적 발표문 긍정 문장 비율(FinBERT)", "earnings release positive-sentence share"),
    "nb_tone_chg": ("실적 발표문 FinBERT 어조 직전 대비 변화", "FinBERT tone change vs previous release"),
    "nb_cq_tone": ("CEO 인용문 FinBERT 어조", "CEO quote FinBERT tone"),
    "pr_n_90": ("90일 회사 보도자료 수(8-K, 실적 외)", "non-earnings press releases, 90 days"),
    "pr_tone_90": ("90일 보도자료 FinBERT 평균 어조", "press release FinBERT tone, 90 days"),
    "pr_neg_90": ("90일 부정 보도자료 수", "negative press releases, 90 days"),
    "pr_tone_30": ("30일 보도자료 FinBERT 평균 어조", "press release FinBERT tone, 30 days"),
    "bs_tone": ("사업의 내용 KR-FinBERT 어조", "business section KR-FinBERT tone"),
    "bs_neg": ("사업의 내용 부정 문장 비율", "business section negative-sentence share"),
    "bs_tone_chg": ("사업의 내용 어조 전년 대비 변화", "business section tone change vs last year"),
    "fh_tone_30": ("30일 뉴스 제목 FinBERT 평균 어조(Finnhub)", "news headline FinBERT tone, 30 days"),
    "fh_neg_30": ("30일 부정 뉴스 제목 수(Finnhub)", "negative news headlines, 30 days"),
    "fh_n_30": ("30일 뉴스 제목 수(Finnhub)", "news headlines, 30 days"),
}


# ================================================================================================ documents


def docs_us_a(settings) -> pd.DataFrame:
    """One row per earnings release: cik, published, lead sentences, CEO-quote sentences."""
    rows = []
    for p in sorted((tone.root(settings) / "ex991").glob("*.jsonl.gz")):
        cik = int(p.name.split(".")[0])
        for r in tone.read_records(p):
            t = r.get("text") or ""
            if not t:
                continue
            q, _ = cf.ceo_quotes(t)
            rows.append(
                {
                    "cik": cik,
                    "acc": r["acc"],
                    "published": pd.Timestamp(r["published"]),
                    "lead": nm.sentences(t, "en", LEAD_A),
                    "quote": nm.sentences(q, "en", QUOTE_A) if q else [],
                }
            )
    return pd.DataFrame(rows)


def docs_us_b(settings) -> pd.DataFrame:
    rows = []
    for p in sorted((nd.root(settings) / "pr").glob("*.jsonl.gz")):
        cik = int(p.name.split(".")[0])
        for r in tone.read_records(p):
            if r.get("text"):
                rows.append(
                    {
                        "cik": cik,
                        "acc": r["acc"],
                        "published": pd.Timestamp(r["published"]),
                        "lead": nm.sentences(r["text"], "en", LEAD_B),
                    }
                )
    return pd.DataFrame(rows)


def docs_kr_a(settings) -> pd.DataFrame:
    rows = []
    for p in sorted((cd.root(settings) / "business").glob("*.jsonl.gz")):
        code = p.name.split(".")[0]
        for r in tone.read_records(p):
            if r.get("text"):
                rows.append(
                    {
                        "entity": code,
                        "year": int(r["year"]),
                        "avail": pd.Timestamp(str(r["rcept_no"])[:8]) + pd.Timedelta(1, "D"),
                        "lead": nm.sentences(r["text"], "ko", LEAD_KR),
                    }
                )
    return pd.DataFrame(rows)


def docs_us_c(settings) -> pd.DataFrame:
    rows, seen = [], set()
    for p in sorted((nd.root(settings) / "finnhub").glob("*.jsonl.gz")):
        for r in tone.read_records(p):
            h = (r.get("headline") or "").strip()
            if not h or (r.get("id"), r.get("entity")) in seen:
                continue
            seen.add((r.get("id"), r.get("entity")))
            rows.append({"entity": r["entity"], "ts": pd.Timestamp(int(r["datetime"]), unit="s"), "headline": h[:300]})
    return pd.DataFrame(rows)


# ================================================================================================ scoring


def _score_lists(clf: nm.Classifier, lists: pd.Series) -> list[np.ndarray]:
    flat = [s for xs in lists for s in xs]
    p = clf.score(flat, progress=say) if flat else np.zeros((0, 3))
    out, i = [], 0
    for xs in lists:
        out.append(p[i : i + len(xs)])
        i += len(xs)
    return out


def _doc_stats(p: np.ndarray) -> tuple[float, float, float]:
    if not len(p):
        return np.nan, np.nan, np.nan
    net = p[:, 0] - p[:, 1]
    return float(net.mean()), float((p[:, 1] > 0.5).mean()), float((p[:, 0] > 0.5).mean())


def scored_us_a(settings, en: str = nm.DEFAULT["en"]) -> tuple[pd.DataFrame, str]:
    d = docs_us_a(settings)
    clf = nm.Classifier(en, settings)
    lead, quote = _score_lists(clf, d["lead"]), _score_lists(clf, d["quote"])
    st = np.array([_doc_stats(p) for p in lead])
    d["nb_tone"], d["nb_neg"], d["nb_pos"] = st[:, 0], st[:, 1], st[:, 2]
    d["nb_cq_tone"] = [_doc_stats(p)[0] for p in quote]
    return d.drop(columns=["lead", "quote"]), clf.spec.tag


def scored_us_b(settings, en: str = nm.DEFAULT["en"]) -> tuple[pd.DataFrame, str]:
    d = docs_us_b(settings)
    clf = nm.Classifier(en, settings)
    d["doc_tone"] = [_doc_stats(p)[0] for p in _score_lists(clf, d["lead"])]
    return d.drop(columns=["lead"]), clf.spec.tag


def scored_kr_a(settings, ko: str = nm.DEFAULT["ko"]) -> tuple[pd.DataFrame, str]:
    d = docs_kr_a(settings)
    clf = nm.Classifier(ko, settings)
    st = np.array([_doc_stats(p) for p in _score_lists(clf, d["lead"])])
    d["bs_tone"], d["bs_neg"] = st[:, 0], st[:, 1]
    return d.drop(columns=["lead"]), clf.spec.tag


def scored_us_c(settings, en: str = nm.DEFAULT["en"]) -> tuple[pd.DataFrame, str]:
    d = docs_us_c(settings)
    clf = nm.Classifier(en, settings)
    p = clf.score(d["headline"].tolist(), progress=say) if len(d) else np.zeros((0, 3))
    d["tone"], d["neg"] = p[:, 0] - p[:, 1], (p[:, 1] > 0.5).astype(float)
    return d.drop(columns=["headline"]), clf.spec.tag


# ================================================================================================ features


def _by_entity(frame: pd.DataFrame, settings, cache, docs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """US docs keyed by CIK -> per entity (any CIK of its chain)."""
    ents = np.unique(frame["entity"].astype(str))
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": ents}))
    by = {int(k): g for k, g in docs.groupby("cik")}
    out = {}
    for e in ents:
        parts = [by[int(c)] for c in chains.get(e, set()) if int(c) in by]
        if parts:
            out[e] = pd.concat(parts).sort_values("published").drop_duplicates("acc")
    return out


def build_us(frame: pd.DataFrame, settings, cache, en: str = nm.DEFAULT["en"]) -> tuple[pd.DataFrame, dict]:
    a, tag = scored_us_a(settings, en)
    b, _ = scored_us_b(settings, en)
    cols_a = ["nb_tone", "nb_neg", "nb_pos", "nb_tone_chg", "nb_cq_tone"]
    cols_b = ["pr_n_90", "pr_tone_90", "pr_neg_90", "pr_tone_30"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols_a + cols_b)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    ga, gb = _by_entity(frame, settings, cache, a), _by_entity(frame, settings, cache, b)
    for e in np.unique(ents):
        ii = np.where(ents == e)[0]
        if e in ga:
            g = ga[e].assign(nb_tone_chg=lambda x: x["nb_tone"] - x["nb_tone"].shift(1))
            g.index = pd.DatetimeIndex(g["published"]) + pd.Timedelta(1, "D")
            g = g[~g.index.duplicated(keep="last")]
            out.iloc[ii, :5] = v13._asof_rows(g[cols_a], dates[ii], tone.TX_MAX_AGE)
        if e in gb:
            g = gb[e].dropna(subset=["doc_tone"])
            ev = g["published"].to_numpy(dtype="datetime64[ns]")
            tv = g["doc_tone"].to_numpy(float)
            n90, s90 = v16._windowed(ev, tv, dates[ii], 90)
            neg90, _ = v16._windowed(ev[tv < NEG_DOC], np.ones(int((tv < NEG_DOC).sum())), dates[ii], 90)
            n30, s30 = v16._windowed(ev, tv, dates[ii], 30)
            with np.errstate(invalid="ignore", divide="ignore"):
                out.iloc[ii, 5] = n90
                out.iloc[ii, 6] = np.where(n90 > 0, s90 / n90, np.nan)
                out.iloc[ii, 7] = neg90
                out.iloc[ii, 8] = np.where(n30 > 0, s30 / n30, np.nan)
        else:
            out.iloc[ii, 5], out.iloc[ii, 7] = 0.0, 0.0
    out.loc[frame["date"].to_numpy() < np.datetime64(nd.START + pd.Timedelta(90, "D")), cols_b] = (
        np.nan
    )  # no window yet
    return out, {"model": tag, "earnings_releases": len(a), "press_releases": len(b), "entities_pr": len(gb)}


def build_kr(frame: pd.DataFrame, settings, ko: str = nm.DEFAULT["ko"]) -> tuple[pd.DataFrame, dict]:
    d, tag = scored_kr_a(settings, ko)
    cols = ["bs_tone", "bs_neg", "bs_tone_chg"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    by = {k: g.sort_values("avail").drop_duplicates("year", keep="last") for k, g in d.groupby("entity")}
    for e in np.unique(ents):
        if e not in by:
            continue
        g = by[e].assign(bs_tone_chg=lambda x: x["bs_tone"] - x["bs_tone"].shift(1))
        ii = np.where(ents == e)[0]
        out.iloc[ii, :] = v16._asof(g, cols, "avail", dates[ii], BS_MAX_AGE)
    return out, {"model": tag, "business_docs": len(d), "entities": len(by)}


def build_us_c(frame: pd.DataFrame, settings, en: str = nm.DEFAULT["en"]) -> tuple[pd.DataFrame, dict]:
    """Descriptive only (one year of headlines): 30-day headline tone / negative count / count per row."""
    d, tag = scored_us_c(settings, en)
    cols = ["fh_tone_30", "fh_neg_30", "fh_n_30"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    for e, g in d.groupby("entity"):
        ii = np.where(ents == e)[0]
        if not len(ii):
            continue
        ev = g["ts"].to_numpy(dtype="datetime64[ns]")
        n, s = v16._windowed(ev, g["tone"].to_numpy(float), dates[ii], 30)
        _, neg = v16._windowed(ev, g["neg"].to_numpy(float), dates[ii], 30)
        out.iloc[ii, 0] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
        out.iloc[ii, 1] = neg
        out.iloc[ii, 2] = n
    first = d["ts"].min() if len(d) else pd.Timestamp.max
    out.loc[frame["date"].to_numpy() < np.datetime64(first + pd.Timedelta(30, "D")), :] = np.nan  # no history yet
    return out, {"model": tag, "headlines": len(d), "first": str(first)}


def main() -> None:
    from quantcio.config import load_settings

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--what", default="all", choices=["all", "us_a", "us_b", "us_c", "kr_a"])
    sc.add_argument("--en", default=nm.DEFAULT["en"], choices=[k for k, s in nm.MODELS.items() if s.lang == "en"])
    sc.add_argument("--ko", default=nm.DEFAULT["ko"], choices=[k for k, s in nm.MODELS.items() if s.lang == "ko"])
    a = ap.parse_args()
    s = load_settings()
    steps = {
        "us_a": lambda: scored_us_a(s, a.en),
        "us_b": lambda: scored_us_b(s, a.en),
        "us_c": lambda: scored_us_c(s, a.en),
        "kr_a": lambda: scored_kr_a(s, a.ko),
    }
    for name, fn in steps.items():
        if a.what in ("all", name):
            d, tag = fn()
            say(f"{name}: {len(d):,} documents scored with {tag}")


if __name__ == "__main__":
    main()
