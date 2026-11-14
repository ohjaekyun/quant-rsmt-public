"""US text-based peers and peer-return diffusion (docs/MODELING.md 67.2, pre-declared 2026-10-17). Research only.

Peers at a month-end t: every universe firm's latest 10-K filed before t (18 months), text = Item 1A + Item 7 sections
cached by scripts/us_v11_data.py; binary word vectors, words kept by their document frequency among those documents
only (5 <= df <= 25%), cosine similarity, top 10 other firms. The peer lists of month-end t are used on every date up to
the next month-end; peer returns are the frame's own return columns of the same date (same timing as ret_1m).
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v16_data as v16  # noqa: E402

K = 10
MAX_AGE_DAYS = 548
DF_MIN, DF_MAX = 5, 0.25
_WORD = re.compile(r"[a-z]{3,}")
DOCS: dict[str, tuple[str, str]] = {
    "tp_ret_1w": ("사업 내용 동종 10개사 1주 수익(유사도 가중)", "text-peer 1-week return"),
    "tp_ret_1m": ("동종 10개사 1개월 수익", "text-peer 1-month return"),
    "tp_mom_12_1": ("동종 10개사 12-1개월 모멘텀", "text-peer 12-1 momentum"),
    "tp_gap_1m": ("동종 1개월 수익 − 자기 1개월 수익(따라잡기)", "peer minus own 1-month return"),
    "tp_sim": ("동종 10개사 평균 유사도(제품 시장 경쟁)", "mean similarity to the top-10 peers"),
}


def sections_root(settings) -> Path:
    return Path(settings.cache_path) / "alt" / "keyed" / "v11" / "tenk_sections"


def load_docs(settings) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    """One row per 10-K (cik, filed) and the binary doc x word matrix over every word seen (indexing only - which
    words count is decided per date from past documents)."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    vocab: dict[str, int] = {}
    rows, indptr, indices = [], [0], []
    for p in sorted(sections_root(settings).glob("*.json.gz")):
        cik = int(p.name.split(".")[0])
        try:
            d = json.loads(gzip.open(p).read())
        except (OSError, EOFError, json.JSONDecodeError):
            continue
        for acc, x in d.items():
            text = f"{x.get('s1a') or ''} {x.get('s7') or ''}".lower()
            words = {w for w in _WORD.findall(text) if w not in ENGLISH_STOP_WORDS}
            if len(words) < 200 or not x.get("filed"):
                continue
            ids = sorted(vocab.setdefault(w, len(vocab)) for w in words)
            indices += ids
            indptr.append(len(indices))
            rows.append({"cik": cik, "acc": acc, "filed": pd.Timestamp(x["filed"])})
    m = sparse.csr_matrix((np.ones(len(indices), dtype=np.float32), indices, indptr), shape=(len(rows), len(vocab)))
    return pd.DataFrame(rows), m


def build(frame: pd.DataFrame, settings, cache) -> tuple[pd.DataFrame, dict]:
    docs, mat = load_docs(settings)
    docs["_row"] = np.arange(len(docs))
    ents_all = np.unique(frame["entity"].astype(str))
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": ents_all}))
    ent_ciks = {e: {int(c) for c in cs} for e, cs in chains.items()}
    f = frame[["date", "entity", "ret_1w", "ret_1m", "mom_12_1"]].copy()
    f["entity"] = f["entity"].astype(str)
    dates = np.sort(f["date"].unique())
    me = pd.DatetimeIndex(dates).to_series().groupby(pd.DatetimeIndex(dates).to_period("M")).max().to_numpy()
    out = pd.DataFrame(np.nan, index=frame.index, columns=list(DOCS))
    n_lists, sims = 0, []
    by_date = {d: g for d, g in f.groupby("date")}
    for i, t in enumerate(me):
        t = pd.Timestamp(t)
        members = sorted(by_date[np.datetime64(t)]["entity"].unique())
        live = docs[(docs["filed"] < t) & (docs["filed"] >= t - pd.Timedelta(MAX_AGE_DAYS, "D"))]
        latest = live.sort_values("filed").groupby("cik").tail(1).set_index("cik")
        ents, rws = [], []
        for e in members:
            cands = [c for c in ent_ciks.get(e, ()) if c in latest.index]
            if cands:
                c = max(cands, key=lambda c: latest.loc[c, "filed"])
                ents.append(e)
                rws.append(int(latest.loc[c, "_row"]))
        if len(ents) < K + 2:
            continue
        x = mat[rws]
        df = np.asarray((mat[latest["_row"].to_numpy()] > 0).sum(axis=0)).ravel()  # df among the live documents
        keep = (df >= DF_MIN) & (df <= DF_MAX * len(latest))
        x = x[:, np.where(keep)[0]]
        norm = np.sqrt(np.asarray(x.sum(axis=1)).ravel())
        norm[norm == 0] = 1.0
        xn = sparse.diags(1.0 / norm) @ x
        s = (xn @ xn.T).toarray()
        np.fill_diagonal(s, -1.0)
        by_cik: dict[int, list[int]] = {}
        for j, e in enumerate(ents):
            for c in ent_ciks.get(e, ()):
                by_cik.setdefault(c, []).append(j)
        for js in by_cik.values():  # the same firm under two entities (a CIK shared by their chains)
            if len(js) > 1:
                s[np.ix_(js, js)] = -1.0
        top = np.argsort(-s, axis=1)[:, :K]
        w = np.take_along_axis(s, top, axis=1).clip(min=0)
        n_lists += 1
        sims.append(float(w.mean()))
        pos = {e: j for j, e in enumerate(ents)}
        nxt = me[i + 1] if i + 1 < len(me) else np.datetime64(t)  # the last month-end's rows use the previous lists
        for d in dates[(dates > np.datetime64(t)) & (dates <= nxt)]:
            g = by_date[d]
            g = g[g["entity"].isin(pos)]
            if g.empty:
                continue
            vals = {c: pd.Series(np.nan, index=ents) for c in ("ret_1w", "ret_1m", "mom_12_1")}
            for c in vals:
                vals[c].loc[g["entity"].to_numpy()] = g[c].to_numpy(float)
            for c, col in (("ret_1w", "tp_ret_1w"), ("ret_1m", "tp_ret_1m"), ("mom_12_1", "tp_mom_12_1")):
                v = vals[c].to_numpy()[top]
                ok = np.isfinite(v) & (w > 0)
                num = np.where(ok, v * w, 0.0).sum(axis=1)
                den = np.where(ok, w, 0.0).sum(axis=1)
                peer = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
                ii = g.index.to_numpy()
                out.loc[ii, col] = peer[[pos[e] for e in g["entity"]]]
            ii = g.index.to_numpy()
            own = g["ret_1m"].to_numpy(float)
            out.loc[ii, "tp_gap_1m"] = out.loc[ii, "tp_ret_1m"].to_numpy() - own
            out.loc[ii, "tp_sim"] = w.mean(axis=1)[[pos[e] for e in g["entity"]]]
    return out, {"docs": len(docs), "month_lists": n_lists, "mean_top10_sim": float(np.mean(sims)) if sims else None}
