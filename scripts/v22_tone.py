"""US earnings-release text (docs/MODELING.md 58.2, pre-declared 2026-10-14). Research only.

    .venv/bin/python scripts/v22_tone.py fetch                 # 8-K item 2.02 EX-99.1 texts of the US universe, 2010+
    .venv/bin/python scripts/v22_tone.py features --lexicon lm  # per-release tone (word list: scripts/lexicons.py)

One gzip JSON-lines file per CIK under <cache>/alt/keyed/v22/ex991/ (resumable): accession, published (NY date of
the acceptance time, else the filing date), the exhibit file name and its plain text. Tone features are computed from
these texts with a sentiment word list in a separate step (the list is chosen before any result is looked at).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import lexicons as lx  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402

say = v13.say
START = pd.Timestamp("2010-01-01")


def root(settings) -> Path:
    return kd.root(settings, "v22")


def read_records(p: Path) -> list[dict]:
    """Complete JSON records of one per-CIK file; a truncated tail (a run killed mid-write, or the file being written
    right now) is skipped."""
    out = []
    try:
        with gzip.open(p, "rt") as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
    except (EOFError, gzip.BadGzipFile, OSError):
        pass
    return out


def releases(subs, cik: int) -> pd.DataFrame:
    """8-K filings with item 2.02 of one CIK since START: accessionNumber, published, primaryDocument."""
    from quantcio.data.alt import us_sec

    blocks = us_sec.submission_blocks(subs, cik, START - pd.Timedelta(200, "D"))
    frames = [us_sec._block_frame(b) for b in blocks if b]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[~(df["accessionNumber"].notna() & df["accessionNumber"].duplicated())]
    df = df[(df["form"] == "8-K") & df["items"].astype(str).str.contains(r"\b2\.02\b", regex=True)].copy()
    df["published"] = us_sec.published_dates(df["acceptanceDateTime"], df["filingDate"]).to_numpy()
    return df[df["published"] >= START].reset_index(drop=True)


def fetch(settings, cache) -> dict:
    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet
    from quantcio.data.us import edgar_html_to_text, exhibits_99, parse_filing_index

    d = root(settings) / "ex991"
    d.mkdir(parents=True, exist_ok=True)
    uni = v16.universe("US", settings, cache)
    chains = v16.us_chains(settings, cache, uni)
    ciks = sorted({int(c) for cs in chains.values() for c in cs})
    net = PitNet(settings)
    n_docs, n_req, t0 = 0, 0, time.time()
    for i, cik in enumerate(ciks, 1):
        p = d / f"{cik}.jsonl.gz"
        have = {r["acc"] for r in read_records(p)} if p.exists() else set()
        rel = releases(Submissions(net), cik)
        todo = rel[~rel["accessionNumber"].isin(have)]
        if todo.empty:
            continue
        with gzip.open(p, "at") as fh:
            for r in todo.itertuples(index=False):
                acc = str(r.accessionNumber)
                base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
                try:
                    idx = parse_filing_index(net.http_get(base + f"{acc}-index.htm", "sec").content)
                    n_req += 1
                    ex = exhibits_99(idx)
                    if not ex:
                        rec = {"acc": acc, "published": str(pd.Timestamp(r.published).date()), "doc": "", "text": ""}
                    else:
                        doc = ex[0]
                        text = edgar_html_to_text(net.http_get(doc["url"], "sec").content)
                        n_req += 1
                        rec = {
                            "acc": acc,
                            "published": str(pd.Timestamp(r.published).date()),
                            "doc": doc["document"],
                            "type": doc["type"],
                            "text": text,
                        }
                except Exception as exc:  # noqa: BLE001 - one filing failing is skipped (retried on the next run)
                    say(f"CIK {cik} {acc}: {type(exc).__name__}")
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_docs += 1
        if i % 25 == 0:
            rate = n_req / max(time.time() - t0, 1)
            say(f"EX-99.1 {i}/{len(ciks)} CIKs, {n_docs} releases, {n_req} requests ({rate:.1f}/s)")
    return {"ciks": len(ciks), "releases_new": n_docs, "requests": n_req}


# ================================================================================================ tone features

TX_DOCS = {
    "tx_neg": ("실적 발표문 부정어 비율", "earnings release negative-word share"),
    "tx_pos": ("실적 발표문 긍정어 비율", "earnings release positive-word share"),
    "tx_unc": ("실적 발표문 불확실어 비율", "earnings release uncertainty-word share"),
    "tx_lit": ("실적 발표문 소송어 비율", "earnings release litigious-word share"),
    "tx_tone": ("실적 발표문 어조 (긍정 − 부정) ÷ (긍정 + 부정)", "earnings release tone"),
    "tx_tone_chg": ("직전 발표 대비 어조 변화", "tone change vs the previous release"),
    "tx_len": ("실적 발표문 길이(log 단어 수)", "earnings release length (log words)"),
}
TX_MAX_AGE = 120  # days a release's tone stays in use


def tone_path(settings, lex: lx.Lexicon) -> Path:
    return root(settings) / f"tone_{lex.name}_{lex.version}.parquet"


def features(settings, spec: str) -> dict:
    """Per release: word counts and shares under one word list -> tone_<list>_<version>.parquet."""
    lex = lx.load_lexicon(spec, settings)
    rows = []
    for p in sorted((root(settings) / "ex991").glob("*.jsonl.gz")):
        cik = int(p.name.split(".")[0])
        for r in read_records(p):
            if not r.get("text"):
                continue
            c = lx.counts(r["text"], lex)
            rows.append({"cik": cik, "acc": r["acc"], "published": r["published"], **c})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"releases": 0}
    n = t["n_words"].clip(lower=1)
    t["tx_neg"], t["tx_pos"] = t["negative"] / n, t["positive"] / n
    t["tx_unc"], t["tx_lit"] = t["uncertainty"] / n, t["litigious"] / n
    t["tx_tone"] = (t["positive"] - t["negative"]) / (t["positive"] + t["negative"]).replace(0, np.nan)
    t["tx_len"] = np.log(n)
    t["published"] = pd.to_datetime(t["published"])
    t["lexicon"] = lex.tag
    t.to_parquet(tone_path(settings, lex), index=False)
    return {"releases": len(t), "ciks": int(t["cik"].nunique()), "lexicon": lex.tag}


def block_tx(frame: pd.DataFrame, settings, cache, spec: str) -> tuple[pd.DataFrame, dict]:
    """tx_* per row: the entity's latest release (any CIK of its chain) published before the row date (known the
    next day), at most TX_MAX_AGE days old; tx_tone_chg = its tone minus the previous release's."""
    lex = lx.load_lexicon(spec, settings)
    t = pd.read_parquet(tone_path(settings, lex)).sort_values("published")
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    cols = list(TX_DOCS)
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": np.unique(ents)}))
    by_cik = {int(k): g for k, g in t.groupby("cik")}
    n = 0
    for e in np.unique(ents):
        parts = [by_cik[int(c)] for c in chains.get(e, set()) if int(c) in by_cik]
        if not parts:
            continue
        g = pd.concat(parts).sort_values("published").drop_duplicates("acc")
        g = g.assign(tx_tone_chg=g["tx_tone"] - g["tx_tone"].shift(1))
        g.index = pd.DatetimeIndex(g["published"]) + pd.Timedelta(1, "D")
        g = g[~g.index.duplicated(keep="last")]
        ii = np.where(ents == e)[0]
        out.iloc[ii, :] = v13._asof_rows(g[cols], dates[ii], TX_MAX_AGE)
        n += 1
    return out, {"entities_with_releases": n, "releases": len(t), "lexicon": lex.tag}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    f = sub.add_parser("features")
    f.add_argument("--lexicon", default="lm", help="lm | csv:<path> (scripts/lexicons.py)")
    a = ap.parse_args()
    s = load_settings()
    if a.cmd == "fetch":
        say(f"fetch: {fetch(s, DiskCache(s.cache_path))}")
    else:
        say(f"features: {features(s, a.lexicon)}")


if __name__ == "__main__":
    main()
