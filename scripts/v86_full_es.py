"""Every stored research block + early-stopped boosting / MLP twins (docs/MODELING.md 86, pre-declared 2026-11-03).
Research only: src/quantcio and the served models are not changed.

    .venv/bin/python scripts/v86_full_es.py pool --market KR     # build the full pool on the frozen data (offline)
    .venv/bin/python scripts/v86_full_es.py study --market KR    # arms A (current inputs) and F (full pool)

Pool: the 84.6 frozen dataset snapshot + every research block from sections 13-85, each built by its own section's
code from stored data only (sockets and proxies are disabled while the blocks are built). Each block's new columns are
kept under .cache/quantcio/research/v86/<M>/ so a rerun reuses them.
Candidates: 14 grid models + 9 early-stopped twins (``*_es``: same grid / scheme / learning rate, the tree or epoch
count chosen on a time-ordered, purged validation tail of each training window, then a refit on the whole window).
Writes reports/modeling/<M>_v86.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling import candidates as cmod  # noqa: E402
from quantcio.modeling import features as feat  # noqa: E402
from quantcio.modeling import metrics as mt  # noqa: E402
from quantcio.modeling import selection as sel  # noqa: E402
from quantcio.modeling.cli import guard_xsh  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
SNAP = {"KR": "d0afe28c7c6013bbcc292bab9343f637", "US": "b9b116b96c483c8bd34b42706d3c2e16"}  # 86.1: the 84.6 data
BASE = (
    "logit_w",
    "enet_logit",
    "ridge_rg",
    "rf",
    "et",
    "lgbm_clf",
    "lgbm_rank",
    "lgbm_reg_w",
    "xgb_clf",
    "xgb_clf_w",
    "xgb_rank",
    "cat_clf",
    "cat_rank",
    "mlp",
)
ES_BASES = {  # twin -> library
    "lgbm_clf": "lgbm",
    "lgbm_rank": "lgbm",
    "lgbm_reg_w": "lgbm",
    "xgb_clf": "xgb",
    "xgb_clf_w": "xgb",
    "xgb_rank": "xgb",
    "cat_clf": "cat",
    "cat_rank": "cat",
    "mlp": "mlp",
}
V85_CANDS = ("logit_w", "et", "lgbm_rank", "lgbm_clf", "xgb_rank", "cat_clf")  # 86.3 reproduction check
ES_VERSION = 1
ES_MAX, ES_PATIENCE, ES_MIN = 2000, 50, 10  # trees: cap, rounds without improvement, floor of the refit
MLP_MAX, MLP_PATIENCE = 200, 10  # epochs
ES_VAL_FRAC, ES_VAL_MIN, ES_VAL_MAX = 0.2, 3, 12  # validation tail: 20% of the training months, 3..12 months
NDCG_FRAC = 0.2  # NDCG@k with k = 20% of the median names per date
ARM = ["-"]  # the arm being run (ES log tag)
PRE: dict[str, pd.DataFrame] = {}  # inputs fetched before going offline (86.4)
ES_LOG: list[Path] = []


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def research_dir(settings, m: str) -> Path:
    p = settings.cache_path / "research" / "v86" / m / SNAP[m][:12]
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_frozen(m: str, settings, cache):
    ds = cache.get("modeling_dataset", SNAP[m], None)
    if ds is None:
        raise SystemExit(f"dataset snapshot {SNAP[m]} is not cached")
    return guard_xsh(ds, settings, say)


# ------------------------------------------------------------------------------------------------ offline


_NET: dict = {}  # the original socket functions / proxy variables while offline
_PROXY = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "NO_PROXY", "no_proxy")
ONLINE_MARKETS = {"US"}  # 86.4: the US blocks refresh SEC submissions / FRED / Federal Register inside their own code


def offline() -> None:
    """No network while the blocks are built: every block comes from stored data (86.1)."""

    def deny(*_a, **_k):
        raise OSError("v86: network disabled while building the pool")

    if _NET:
        return
    _NET["sock"] = {k: getattr(socket.socket, k) for k in ("connect", "connect_ex")}
    _NET["mod"] = {k: getattr(socket, k) for k in ("create_connection", "getaddrinfo")}
    _NET["env"] = {k: os.environ.get(k) for k in (*_PROXY, "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    for k in _NET["sock"]:
        setattr(socket.socket, k, deny)
    for k in _NET["mod"]:
        setattr(socket, k, deny)
    for k in _PROXY[:6]:
        os.environ[k] = "http://127.0.0.1:9"  # libcurl (curl_cffi / yfinance) does not use Python sockets
    os.environ.pop("NO_PROXY", None)
    os.environ.pop("no_proxy", None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def online() -> None:
    if not _NET:
        return
    for k, v in _NET["sock"].items():
        setattr(socket.socket, k, v)
    for k, v in _NET["mod"].items():
        setattr(socket, k, v)
    for k, v in _NET["env"].items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    _NET.clear()


# ------------------------------------------------------------------------------------------------ pool


def redact(text: str) -> str:
    """Error text without credentials: URL query strings (API keys travel there) and key=value secrets."""
    text = re.sub(r"\?[^\s'\")]*", "?…", text)
    return re.sub(r"(?i)(api[_-]?key|token|secret|password|crtfc_key|servicekey)=\S+", r"\1=***", text)


def _df_block(f: pd.DataFrame, df: pd.DataFrame, docs: dict, regime=()) -> tuple[list[str], list[str]]:
    """Raw columns + per-date rank-gauss model inputs, exactly as the 22-32 studies add a block."""
    xs, reg = [], []
    for c in df.columns:
        vals = df[c].astype(float)
        if vals.notna().sum() == 0 or vals.nunique() < 2:
            continue
        f[c] = vals.to_numpy()
        feat.FEATURE_DOCS[c] = docs.get(c, (c, c))
        if c in regime:
            reg.append(c)
        else:
            f[feat.x_col(c)] = feat.rank_gauss(f[c], f["date"])
            xs.append(feat.x_col(c))
    return xs, reg


def _raw_x(raw: list[str]) -> list[str]:
    return [feat.x_col(c) for c in raw]


def _step(tag: str, m: str, ds, uni, s, cache) -> tuple[list[str], list[str], dict]:
    """One block of the 86.1 pool: (model columns, regime columns, info). Each = its own section's code."""
    f = ds.frame
    if tag == "us_v4v8K":  # 24
        import full_opt_study as fo

        return fo.us_blocks(ds, s, cache, universe=uni)
    if tag == "E":  # 25
        import us_v9_features as v9f

        ranked, prod, docs = v9f.engineer(f)
        feat.FEATURE_DOCS.update(docs)
        return _raw_x([*ranked, *prod]), [], {}
    if tag in ("T", "F", "S"):  # 31
        import v13_study as v13s

        raw, reg, info = v13s.add_block(tag, m, ds, uni, s)
        return _raw_x(raw), reg, info
    if tag in ("C", "D", "H"):  # 32
        import v14_study as v14s

        raw, reg, info = v14s.add_block(tag, m, ds, uni, s, cache)
        return _raw_x(raw), reg, info
    if tag == "L":  # 38
        import location_study as v15s

        raw, reg, info = v15s.add_block(m, ds, s, cache)
        return _raw_x(raw), reg, info
    if tag in ("X", "J", "B", "R"):  # 39
        import v16_study as v16s

        raw, reg, info = v16s.add_block(tag, m, ds, s, cache)
        return _raw_x(raw), reg, info
    if tag == "V":  # 43
        import v17_study as v17s

        raw, reg, info = v17s.add_block("V", m, ds, s, cache)
        return _raw_x(raw), reg, info
    if tag == "W":  # 40 (US)
        import us_weekly_study as wk

        w, info = wk.block_w(ds, uni, s, cache)
        xs, reg = _df_block(f, w, wk.W_DOCS)
        return xs, reg, info
    if tag == "53":
        import v19_study as v19s

        raw, reg, info = v19s.add_blocks(m, ds, s, cache)
        keep = [c for c in raw if c not in set(info["leak_suspect"])]
        return _raw_x(keep), reg, {k: v for k, v in info.items() if k in ("leak_suspect", "empty")}
    if tag == "54":
        import v20_study as v20s

        blocks, reg, info = v20s.extra_blocks(m, ds, uni, s, cache)
        return [x for b in blocks.values() for x in b], reg, {"blocks": {k: len(v) for k, v in blocks.items()}}
    if tag == "55":
        import v21_study as v21s

        xs, reg, info = v21s.add_blocks(m, ds, s, cache)
        return xs, reg, {k: v for k, v in info.items() if k == "empty"}
    if tag == "TX":  # 58.2 (US)
        import v22_tone as tone

        df, info = tone.block_tx(f, s, cache, "lm")
        xs, reg = _df_block(f, df, tone.TX_DOCS)
        return xs, reg, {}
    if tag == "SI":  # 61 (KR)
        import v23_short_index as v23

        df, info = v23.build_kr(f, s)
        xs, reg = _df_block(f, df, v23.DOCS, v23.REGIME)
        return xs, reg, {}
    if tag == "CEO":  # 63
        import v24_ceo_features as v24

        df, info = v24.block(m, f, s, cache)
        xs, reg = _df_block(f, df, v24.DOCS)
        return xs, reg, {}
    if tag == "NLP":  # 64
        import nlp_models as nm
        import v25_nlp_features as v25

        df, info = v25.build_us(f, s, cache, nm.DEFAULT["en"]) if m == "US" else v25.build_kr(f, s, nm.DEFAULT["ko"])
        xs, reg = _df_block(f, df, v25.DOCS)
        return xs, reg, {}
    if tag == "KRX":  # 66 (KR)
        import v26_features as v26

        df, info = v26.block(f, s, cache)
        xs, reg = _df_block(f, df, v26.DOCS)
        return xs, reg, {}
    if tag == "TP":  # 67.2 (US)
        import v27_peers as v27

        df, info = v27.build(f, s, cache)
        xs, reg = _df_block(f, df, v27.DOCS)
        return xs, reg, {}
    if tag in ("73", "75", "76"):
        import v30_features as v30
        import v31_features as v31
        import v32_features as v32

        mod = {"73": v30, "75": v31, "76": v32}[tag]
        df, regime, info = mod.block(m, f, s)
        xs, reg = _df_block(f, df, mod.DOCS, regime)
        return xs, reg, {}
    if tag == "85":
        import v85_competition as v85

        rec = PRE.pop("85", None)
        rec = rec if rec is not None else (v85.kr_records(s) if m == "KR" else v85.us_records(s))
        df, info = v85.block(f, rec, m)
        xs, reg = _df_block(f, df[list(v85.FEATURES)], v85.DOCS)
        return xs, reg, {}
    raise ValueError(tag)


STEPS = {
    "KR": ["T", "F", "S", "C", "D", "H", "L", "X", "J", "B", "R", "V", "53", "54", "55"]
    + ["SI", "CEO", "NLP", "KRX", "73", "75", "76", "85"],
    "US": ["us_v4v8K", "E", "T", "F", "S", "C", "D", "H", "L", "X", "J", "B", "R", "V", "W", "53", "54", "55"]
    + ["TX", "CEO", "NLP", "TP", "73", "75", "85"],
}


def _hash(f: pd.DataFrame, cols: list[str]) -> str:
    return str(int(pd.util.hash_pandas_object(f[cols], index=False).to_numpy().sum(dtype=np.uint64)))


def _coverage(f: pd.DataFrame, xs: list[str], m: str) -> dict:
    lab = f["labelled"].to_numpy(bool) & f["is_me"].to_numpy(bool)
    me = f.loc[lab]
    out = {"all": round(float(np.median([me[c].notna().mean() for c in xs])), 3) if xs else None}
    if m == "KR" and xs:
        top = me.groupby("date")["mcap"].rank(ascending=False, method="first") <= 200
        out["top200"] = round(float(np.median([me.loc[top, c].notna().mean() for c in xs])), 3)
        out["201_400"] = round(float(np.median([me.loc[~top, c].notna().mean() for c in xs])), 3)
    return out


def _load_step(rd: Path, tag: str, f: pd.DataFrame) -> dict | None:
    pq, js = rd / f"{tag}.parquet", rd / f"{tag}.json"
    if not (pq.exists() and js.exists()):
        return None
    info = json.loads(js.read_text())
    if info.get("error"):
        return info
    d = pd.read_parquet(pq)
    if not (d["date"].to_numpy() == f["date"].to_numpy()).all() or not (d["entity"] == f["entity"]).all():
        raise SystemExit(f"{tag}: stored rows do not match the frame")
    for c in d.columns:
        if c not in ("date", "entity"):
            f[c] = d[c].to_numpy()
    return info


def build_pool(m: str) -> dict:
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = load_frozen(m, s, cache)
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params

    n, start = universe_params(Market(m), s)
    uni = load_pit_universe(Market(m), s, cache, n, start)  # the src loader (data vintages) before going offline
    rd = research_dir(s, m)
    if m not in ONLINE_MARKETS and not (rd / "85.parquet").exists():  # 86.4: section 85 re-downloads FX crosses
        import v85_competition as v85

        PRE["85"] = v85.kr_records(s) if m == "KR" else v85.us_records(s)
    net_off = m not in ONLINE_MARKETS
    if net_off:
        offline()
    f = ds.frame
    base_xs, base_reg = list(ds.xs_cols), list(ds.regime_cols)
    base_cols = [*base_xs, *base_reg, "date", "entity", "y_e", "y_cls", "y_q", "y_dm", "w_date"]
    base_cols = [c for c in dict.fromkeys(base_cols) if c in f]
    h0 = _hash(f, base_cols)
    steps: dict[str, dict] = {}
    deadline = float(os.environ.get("V86_DEADLINE", "inf"))  # epoch: stop before the next new block (resumable)
    for tag in STEPS[m]:
        info = _load_step(rd, tag, f)
        if info is None and time.time() > deadline:
            say(f"deadline reached before step {tag}: stopping (rerun resumes from the stored blocks)")
            raise SystemExit(3)
        if info is not None:
            say(f"step {tag}: stored ({len(info.get('xs', []))} features, {len(info.get('reg', []))} regime)")
            steps[tag] = info
            continue
        before = set(f.columns)
        t0 = time.monotonic()
        try:
            xs, reg, binfo = _step(tag, m, ds, uni, s, cache)
            err = None
        except Exception as exc:  # noqa: BLE001 - a block that needs the network is reported, not fatal
            xs, reg, binfo, err = [], [], {}, redact(f"{type(exc).__name__}: {exc}")[:500]
        f = ds.frame
        ds.xs_cols, ds.regime_cols = list(base_xs), list(base_reg)
        new = [c for c in f.columns if c not in before and pd.api.types.is_numeric_dtype(f[c])]
        if _hash(f, base_cols) != h0:
            raise SystemExit(f"step {tag} changed the current inputs (86.1 hash check)")
        xs = [c for c in dict.fromkeys(xs) if c in f and c not in base_xs]
        reg = [c for c in dict.fromkeys(reg) if c in f and c not in base_reg]
        info = {
            "tag": tag,
            "xs": xs,
            "reg": reg,
            "seconds": round(time.monotonic() - t0, 1),
            "error": err,
            "coverage": _coverage(f, xs, m),
            "info": json.loads(json.dumps(binfo, default=str))
            if len(json.dumps(binfo, default=str)) < 20000
            else "(long)",
        }
        if err is None:
            d = f[["date", "entity"]].copy()
            for c in dict.fromkeys([*new, *xs, *reg, *(x[2:] for x in xs if x[2:] in f)]):
                d[c] = f[c].astype(np.float32) if pd.api.types.is_float_dtype(f[c]) else f[c]
            d.to_parquet(rd / f"{tag}.parquet")
        (rd / f"{tag}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
        say(
            f"step {tag}: +{len(xs)} features, +{len(reg)} regime, {info['seconds']}s, cover {info['coverage']}"
            + (f", ERROR {err}" if err else "")
        )
        steps[tag] = info
    # assemble (86.1): dedupe, drop empty / constant, leak screen
    import v20_study as v20s

    xs_all = [x for t in STEPS[m] for x in steps[t].get("xs", [])]
    reg_all = [r for t in STEPS[m] for r in steps[t].get("reg", [])]
    xs_all = [
        x for x in dict.fromkeys(xs_all) if x in f and x not in base_xs and f[x].notna().any() and f[x].nunique() > 1
    ]
    reg_all = [
        r for r in dict.fromkeys(reg_all) if r in f and r not in base_reg and f[r].notna().any() and f[r].nunique() > 1
    ]
    aucs, sus = v20s.leak_screen(ds, xs_all)
    final = [x for x in xs_all if x not in set(sus)]
    summary = {
        "market": m,
        "snapshot": ds.snapshot,
        "base": {"xs": len(base_xs), "regime": len(base_reg)},
        "steps": {
            t: {k: v for k, v in steps[t].items() if k not in ("xs", "reg", "info")}
            | {"n_xs": len(steps[t].get("xs", [])), "n_reg": len(steps[t].get("reg", []))}
            for t in STEPS[m]
        },
        "candidates_xs": len(xs_all),
        "leak_suspect": sus,
        "missing_auc": aucs,
        "final_xs": final,
        "final_reg": reg_all,
        "coverage_final": _coverage(f, final, m),
    }
    (rd / "pool.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    say(f"pool {m}: +{len(final)} features (+{len(reg_all)} regime), leak suspects {len(sus)}: {sus}")
    return summary


def load_pool(m: str, ds, settings) -> tuple[list[str], list[str], dict]:
    rd = research_dir(settings, m)
    summary = json.loads((rd / "pool.json").read_text())
    if summary["snapshot"] != ds.snapshot:
        raise SystemExit("pool was built on another snapshot")
    for tag in STEPS[m]:
        _load_step(rd, tag, ds.frame)
    return summary["final_xs"], summary["final_reg"], summary


# ------------------------------------------------------------------------------------------------ early stopping


def es_split(train: pd.DataFrame, ctx) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """(fit rows, validation rows, info): the last 20% of the training months (3..12) validate; fit rows whose label
    window could reach the first validation date (h + 1 sessions before it) are dropped (86.2)."""
    dates = pd.DatetimeIndex(np.sort(train["date"].unique()))
    if len(dates) < 2:
        return None
    months = (dates[-1] - dates[0]).days / 30.44
    k = int(np.clip(round(months * ES_VAL_FRAC), ES_VAL_MIN, ES_VAL_MAX))
    v0 = dates[-1] - pd.DateOffset(months=k)
    val = train[train["date"] > v0]
    if val.empty:
        return None
    cal = pd.DatetimeIndex(ctx.calendar)
    pos = int(cal.searchsorted(val["date"].min()))
    fit_end = cal[max(pos - int(ctx.horizon) - 1, 0)]
    fit = train[train["date"] <= fit_end]
    if fit["date"].nunique() < 3 or val["date"].nunique() < 2:
        return None
    return fit, val, {"val_months": k, "n_fit": len(fit), "n_val": len(val), "fit_end": str(fit_end.date())}


def _x(d: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return d[cols].to_numpy(dtype=np.float64)


def _ranked(d: pd.DataFrame) -> pd.DataFrame:
    return d[np.isfinite(d["y_q"].to_numpy(dtype=float))].sort_values("date", kind="stable")


def _ndcg_k(val: pd.DataFrame) -> int:
    return max(1, int(round(NDCG_FRAC * float(val.groupby("date").size().median()))))


def _es_lgbm(base: str, fit, val, cols, params, scheme, ctx) -> tuple[int, float]:
    import lightgbm as lgb

    p = {**params, "n_estimators": ES_MAX}
    cb = [lgb.early_stopping(ES_PATIENCE, first_metric_only=True, verbose=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if base == "lgbm_rank":
            fit, val = _ranked(fit), _ranked(val)
            kw = cmod._lgbm_common(p, len(fit), ctx, float(np.var(fit["y_q"].to_numpy(dtype=float))))
            kw["min_child_samples"] = cmod._frac_leaf(len(fit), float(p.get("leaf_frac", 0.01)))
            m = lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=10_000, metric="ndcg", **kw)
            m.fit(
                _x(fit, cols),
                fit["y_q"].to_numpy(dtype=int),
                group=fit.groupby("date", sort=True).size().to_numpy(),
                sample_weight=cmod.row_weights(fit, scheme, ctx),
                eval_set=[(_x(val, cols), val["y_q"].to_numpy(dtype=int))],
                eval_group=[val.groupby("date", sort=True).size().to_numpy()],
                eval_sample_weight=[cmod.row_weights(val, scheme, ctx)],
                eval_at=[_ndcg_k(val)],
                callbacks=cb,
            )
        else:
            if base == "lgbm_clf":
                yf, yv = fit["y_cls"].to_numpy(dtype=float), val["y_cls"].to_numpy(dtype=float)
            else:  # lgbm_reg_w
                yf, yv = cmod.reg_target(fit, scheme), cmod.reg_target(val, scheme)
            of, ov = np.isfinite(yf), np.isfinite(yv)
            wf, wv = cmod.row_weights(fit, scheme, ctx)[of], cmod.row_weights(val, scheme, ctx)[ov]
            if base == "lgbm_clf":
                kw = cmod._lgbm_common(p, int(of.sum()), ctx)
                m = lgb.LGBMClassifier(objective="binary", metric="binary_logloss", **kw)
                yf, yv = yf.astype(int), yv.astype(int)
            else:
                kw = cmod._lgbm_common(p, int(of.sum()), ctx, float(np.var(yf[of])) if of.any() else 1.0)
                m = lgb.LGBMRegressor(objective="regression", metric="l2", **kw)
            m.fit(
                _x(fit, cols)[of],
                yf[of],
                sample_weight=wf,
                eval_set=[(_x(val, cols)[ov], yv[ov])],
                eval_sample_weight=[wv],
                callbacks=cb,
            )
    best = int(m.best_iteration_ or 0)
    score = next(iter(m.best_score_.get("valid_0", {}).values()), float("nan"))
    return best, float(score)


def _es_xgb(base: str, fit, val, cols, params, scheme, ctx) -> tuple[int, float]:
    import xgboost as xgb

    p = {**params, "n_estimators": ES_MAX}
    if base == "xgb_rank":
        fit, val = _ranked(fit), _ranked(val)
        qf, gwf = cmod._date_groups(fit, scheme, ctx)
        qv, gwv = cmod._date_groups(val, scheme, ctx)
        kw = cmod._xgb_common(p, len(fit), ctx)
        kw["min_child_weight"] = 1.0
        m = xgb.XGBRanker(
            objective="rank:pairwise", eval_metric=f"ndcg@{_ndcg_k(val)}", early_stopping_rounds=ES_PATIENCE, **kw
        )
        m.fit(
            _x(fit, cols),
            fit["y_q"].to_numpy(dtype=int),
            qid=qf,
            sample_weight=gwf,
            eval_set=[(_x(val, cols), val["y_q"].to_numpy(dtype=int))],
            eval_qid=[qv],
            sample_weight_eval_set=[gwv],
            verbose=False,
        )
    else:  # xgb_clf, xgb_clf_w
        yf, yv = fit["y_cls"].to_numpy(dtype=float), val["y_cls"].to_numpy(dtype=float)
        of, ov = np.isfinite(yf), np.isfinite(yv)
        kw = cmod._xgb_common(p, int(of.sum()), ctx, 0.25)
        m = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=ES_PATIENCE, **kw
        )
        m.fit(
            _x(fit, cols)[of],
            yf[of].astype(int),
            sample_weight=cmod.row_weights(fit, scheme, ctx)[of],
            eval_set=[(_x(val, cols)[ov], yv[ov].astype(int))],
            sample_weight_eval_set=[cmod.row_weights(val, scheme, ctx)[ov]],
            verbose=False,
        )
    return int(m.best_iteration) + 1, float(m.best_score)


def _es_cat(base: str, fit, val, cols, params, scheme, ctx) -> tuple[int, float]:
    from catboost import CatBoostClassifier, CatBoostRanker, Pool

    kw = cmod._cat_common({**params, "iterations": ES_MAX}, ctx)
    if base == "cat_rank":
        fit, val = _ranked(fit), _ranked(val)
        qf = pd.factorize(fit["date"], sort=True)[0]
        qv = pd.factorize(val["date"], sort=True)[0]
        m = CatBoostRanker(loss_function="YetiRank", eval_metric=f"NDCG:top={_ndcg_k(val)}", **kw)
        m.fit(
            Pool(_x(fit, cols), fit["y_q"].to_numpy(dtype=float), group_id=qf),
            eval_set=Pool(_x(val, cols), val["y_q"].to_numpy(dtype=float), group_id=qv),
            early_stopping_rounds=ES_PATIENCE,
        )
    else:  # cat_clf
        yf, yv = fit["y_cls"].to_numpy(dtype=float), val["y_cls"].to_numpy(dtype=float)
        of, ov = np.isfinite(yf), np.isfinite(yv)
        m = CatBoostClassifier(loss_function="Logloss", eval_metric="Logloss", **kw)
        m.fit(
            _x(fit, cols)[of],
            yf[of].astype(int),
            sample_weight=cmod.row_weights(fit, scheme, ctx)[of],
            eval_set=Pool(_x(val, cols)[ov], yv[ov].astype(int), weight=cmod.row_weights(val, scheme, ctx)[ov]),
            early_stopping_rounds=ES_PATIENCE,
        )
    best = m.get_best_iteration()
    score = next(iter(m.get_best_score().get("validation", {}).values()), float("nan"))
    return int(best if best is not None else 0) + 1, float(score)


_ES = {"lgbm": _es_lgbm, "xgb": _es_xgb, "cat": _es_cat}


def _log(rec: dict) -> None:
    if ES_LOG:
        with ES_LOG[0].open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


def _make_es_fitter(base: str):
    lib = ES_BASES[base]
    n_key = "iterations" if lib == "cat" else "n_estimators"

    def fit_es(c, train, params, scheme, ctx):
        rec = {
            "arm": ARM[0],
            "cand": c.name,
            "cutoff": str(pd.Timestamp(ctx.cutoff).date()) if ctx.cutoff is not None else None,
            "params": params,
            "scheme": scheme,
            "n_train": len(train),
        }
        split = es_split(train, ctx)
        best = None
        if split is not None:
            fit, val, sinfo = split
            rec.update(sinfo)
            best, score = _ES[lib](base, fit, val, cmod._tree_cols(ctx, train), params, scheme, ctx)
            rec["val_score"] = score
        n = max(ES_MIN, int(best)) if best else None
        rec["best"] = best
        rec["n_final"] = n
        out = cmod._FITTERS[base](c, train, {**params, n_key: n} if n else params, scheme, ctx)
        out.extra["es"] = {k: rec.get(k) for k in ("best", "n_final", "val_months", "n_fit", "n_val")}
        _log(rec)
        return out

    return fit_es


def _fit_mlp_es(c, train, params, scheme, ctx):
    """``mlp`` with a time-ordered, purged validation tail instead of sklearn's random 10% (86.2): epochs by
    partial_fit until the weighted validation MSE stops improving, then a refit of exactly that many epochs."""
    from sklearn.neural_network import MLPRegressor

    def make(max_iter: int) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=tuple(int(x) for x in str(params.get("hidden", "64-32")).split("-")),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params.get("lr_init", 1e-3)),
            batch_size=512,
            max_iter=max_iter,
            early_stopping=False,
            tol=0.0,
            n_iter_no_change=max_iter + 1,
            random_state=int(ctx.seed),
        )

    xs = list(ctx.xs_cols)
    rec = {
        "arm": ARM[0],
        "cand": c.name,
        "cutoff": str(pd.Timestamp(ctx.cutoff).date()) if ctx.cutoff is not None else None,
        "params": params,
        "scheme": scheme,
        "n_train": len(train),
    }
    best = None
    split = es_split(train, ctx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if split is not None:
            fit, val, sinfo = split
            rec.update(sinfo)
            yf, yv = cmod._rg_target(fit, scheme), cmod._rg_target(val, scheme)
            of, ov = np.isfinite(yf), np.isfinite(yv)
            xf = np.nan_to_num(fit[xs].to_numpy(dtype=np.float64), nan=0.0)[of]
            xv = np.nan_to_num(val[xs].to_numpy(dtype=np.float64), nan=0.0)[ov]
            wf, wv = cmod.row_weights(fit, scheme, ctx)[of], cmod.row_weights(val, scheme, ctx)[ov]
            m, best_loss, bad = make(1), np.inf, 0
            for ep in range(1, MLP_MAX + 1):
                m.partial_fit(xf, yf[of], sample_weight=wf)
                loss = float(np.average((m.predict(xv) - yv[ov]) ** 2, weights=wv))
                if loss < best_loss:
                    best, best_loss, bad = ep, loss, 0
                else:
                    bad += 1
                    if bad >= MLP_PATIENCE:
                        break
            rec["val_score"] = best_loss
        n = max(1, int(best)) if best else 200
        y = cmod._rg_target(train, scheme)
        ok = np.isfinite(y)
        X = np.nan_to_num(train[xs].to_numpy(dtype=np.float64), nan=0.0)
        model = make(n).fit(X[ok], y[ok], sample_weight=cmod.row_weights(train, scheme, ctx)[ok])
    rec["best"], rec["n_final"] = best, n
    _log(rec)
    out = cmod.Fitted(c.name, params, scheme, model, xs, "mlp")
    out.extra["es"] = {k: rec.get(k) for k in ("best", "n_final", "val_months", "n_fit", "n_val")}
    return out


def candidates() -> list[cmod.Candidate]:
    by = {c.name: c for c in cmod.default_candidates()}
    missing = [n for n in ("mom_12_1", *BASE) if n not in by]
    if missing:
        raise SystemExit(f"missing candidates {missing}")
    es = []
    for n in ES_BASES:
        cmod._FITTERS[f"{n}_es"] = _fit_mlp_es if n == "mlp" else _make_es_fitter(n)
        es.append(
            replace(
                by[n],
                name=f"{n}_es",
                fitter=f"{n}_es",
                label_ko=f"{by[n].label_ko} + 조기 종료",
                label_en=f"{by[n].label_en} + early stopping",
            )
        )
    return [by["mom_12_1"], *(by[n] for n in BASE), *es]


def patch_selection() -> None:
    sel.COMPOSITE_CANDIDATES = (cmod.ENSEMBLE, cmod.ENSEMBLE_W)
    sel.COMPOSITES = (cmod.ENSEMBLE.name, cmod.ENSEMBLE_W.name)
    sel.DIAGNOSTIC_PAIRS = ()


# ------------------------------------------------------------------------------------------------ analysis


def _per_date(res, name: str, part: str, cost: float) -> pd.Series:
    frame = res.oos if part == "outer" else res.holdout_preds
    return mt.per_date_primary(frame, name, cost) if frame is not None and name in frame else pd.Series(dtype=float)


def averaged(pairs: list[tuple[str, str]], res_x, res_y, cost: float) -> dict:
    """Per-date mean over pairs of (name_x in res_x) - (name_y in res_y); per pair and averaged (86.3)."""
    out: dict = {"per_pair": {}}
    for part in ("outer", "holdout"):
        diffs = []
        for nx, ny in pairs:
            a, b = _per_date(res_x, nx, part, cost), _per_date(res_y, ny, part, cost)
            common = a.index.intersection(b.index)
            if len(common):
                diffs.append((a.reindex(common) - b.reindex(common)).rename(nx))
                out["per_pair"].setdefault(nx if nx == ny else f"{nx} − {ny}", {})[part] = mt.paired_difference(a, b)
        avg = pd.concat(diffs, axis=1).mean(axis=1) if diffs else pd.Series(dtype=float)
        out[f"mean_{part}"] = mt.summarize(avg.to_numpy(), 0.0)
    mo, mh = out["mean_outer"], out["mean_holdout"]
    out["helps"] = bool((mo["lo"] if mo["lo"] is not None else -1) > 0 and (mh["mean"] is not None and mh["mean"] >= 0))
    return out


def table(res) -> dict:
    out = {}
    for c in res.candidates:
        o, h = c["outer"], c.get("holdout") or {}

        def g(d, k, f="mean"):
            return (d.get(k) or {}).get(f)

        out[c["name"]] = {
            "outer": g(o, sel.PRIMARY),
            "outer_lo": g(o, sel.PRIMARY, "lo"),
            "outer_hi": g(o, sel.PRIMARY, "hi"),
            "outer_ic": g(o, "ic"),
            "outer_auc": g(o, "auc_median"),
            "holdout": g(h, sel.PRIMARY),
            "holdout_ic": g(h, "ic"),
            "holdout_auc": g(h, "auc_median"),
            "vs_mom": (c.get("vs_momentum_auc") or {}).get("mean"),
            "vs_mom_lo": (c.get("vs_momentum_auc") or {}).get("lo"),
        }
    return out


def insample(res, ds, names: list[str]) -> dict:
    """Final-context models scored on their own training month-ends (86.3 overfitting diagnostic)."""
    data = sel._Data(ds)
    _, _, final, _ = sel.make_contexts(data)
    rows = data.train(final.train_end, "M")
    out = {}
    for n in names:
        fitted = res.final.fitted.get(n) if res.final is not None else None
        if fitted is None:
            continue
        fr = rows[["date", "entity", "y_e", "r"]].assign(score=fitted.predict(rows))
        out[n] = {
            "ic": float(mt.per_date_ic(fr, "score").mean()),
            "auc": float(mt.per_date_auc_median(fr, "score").mean()),
            "es": fitted.extra.get("es"),
        }
    return out


def es_stats(path: Path, arm: str, outer_starts: set[str]) -> dict:
    if not path.exists():
        return {}
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    recs = [r for r in recs if r["arm"] == arm]
    seen, uniq = set(), []
    for r in recs:  # a context rerun after an interruption logs its fits twice
        k = (r["cand"], r["cutoff"], json.dumps(r["params"], sort_keys=True), r["scheme"], r["n_train"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    out = {}
    for name in sorted({r["cand"] for r in uniq}):
        rr = [r for r in uniq if r["cand"] == name]
        b = np.array([r["n_final"] for r in rr if r.get("n_final")], dtype=float)
        refit = np.array([r["n_final"] for r in rr if r.get("n_final") and r["cutoff"] in outer_starts], dtype=float)
        out[name] = {
            "fits": len(rr),
            "fallback": sum(1 for r in rr if not r.get("best")),
            "median": float(np.median(b)) if len(b) else None,
            "p10": float(np.percentile(b, 10)) if len(b) else None,
            "p90": float(np.percentile(b, 90)) if len(b) else None,
            "outer_refit_median": float(np.median(refit)) if len(refit) else None,
        }
    return out


def _f(x, spec: str = "+.4f") -> str:
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def write_md(m: str, r: dict) -> Path:
    ta, tf = r["arm_a"]["table"], r["arm_f"]["table"]
    pool = r["pool"]
    lines = [
        f"# {m} 전체 변수 + 조기 종료 (docs/MODELING.md 86)",
        "",
        f"- 데이터셋 `{r['snapshot']}` (84.6 고정 자료), 비용 {r['cost_bps']:.0f}bp, "
        "지표: 월 상위 20% 베타 조정 순초과",
        f"- 입력: A {pool['base']['xs']}개 + 국면 {pool['base']['regime']}개, F = A + {len(pool['final_xs'])}개 "
        f"+ 국면 {len(pool['final_reg'])}개 (누출 의심 {len(pool['leak_suspect'])}개 제외)",
    ]
    if r.get("variant"):
        gl = pool.get("group_leak") or {}
        lines.append(
            f"- **F′ (86.6):** 시가총액 그룹(상위 200 / 201~400위) 안 결측 누출 검사로 {len(gl)}개를 더 뺐습니다 "
            f"(아래 표의 F 열 = F′): " + ", ".join(f"`{k}` {v}" for k, v in gl.items())
        )
    lines += [
        "",
        "## 판정",
        "",
    ]
    for key, label in (
        ("es_f", "조기 종료 효과(F, 9쌍)"),
        ("es_a", "조기 종료 효과(A, 9쌍)"),
        ("full", "전체 변수 효과(F − A, 23개)"),
    ):
        e = r[key]
        mo, mh = e["mean_outer"], e["mean_holdout"]
        lines.append(
            f"- {label}: 외부 {_f(mo['mean'])} [{_f(mo['lo'])}, {_f(mo['hi'])}] (t {_f(mo['t'], '.2f')}), "
            f"홀드아웃 {_f(mh['mean'])} → **{'도움' if e['helps'] else '도움 없음'}**"
        )
    lines += [
        f"- 선정(F): `{r['arm_f']['selected']}` (게이트 {'통과' if r['arm_f']['gate'].get('passed') else '실패'}), "
        f"홀드아웃 1위(F): `{r['arm_f']['holdout_best']}`; "
        f"참고 A: `{r['arm_a']['selected']}` / `{r['arm_a']['holdout_best']}`",
        f"- 재현(A의 85절 6개 = 85절 A): {r['repro']}",
        "",
        "## 후보별 (외부·홀드아웃, 월)",
        "",
        "| 모델 | A 외부 | A 홀드아웃 | F 외부 [95%] | F 홀드아웃 | F 외부 IC | F 학습 IC "
        "| F 학습 AUC | F 홀드아웃 AUC | 나무 수(F 최종) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    ins = r["arm_f"]["insample"]
    for n in tf:
        a, b, i = ta.get(n, {}), tf[n], ins.get(n, {})
        es = (i.get("es") or {}).get("n_final")
        lines.append(
            f"| `{n}` | {_f(a.get('outer'))} | {_f(a.get('holdout'))} | {_f(b['outer'])} [{_f(b['outer_lo'])}, "
            f"{_f(b['outer_hi'])}] | {_f(b['holdout'])} | {_f(b['outer_ic'])} | {_f(i.get('ic'))} | "
            f"{_f(i.get('auc'), '.3f')} | {_f(b['holdout_auc'], '.3f')} | {es if es else ''} |"
        )
    lines += [
        "",
        "## 쌍둥이별 조기 종료 효과 (ES − 원래, 외부 [95%] / 홀드아웃)",
        "",
        "| 쌍 | F | A |",
        "|---|---|---|",
    ]
    for n in ES_BASES:
        pf = r["es_f"]["per_pair"].get(f"{n}_es − {n}", {})
        pa = r["es_a"]["per_pair"].get(f"{n}_es − {n}", {})

        def cell(p):
            o, h = p.get("outer", {}), p.get("holdout", {})
            return f"{_f(o.get('mean'))} [{_f(o.get('lo'))}, {_f(o.get('hi'))}] / {_f(h.get('mean'))}"

        lines.append(f"| `{n}` | {cell(pf)} | {cell(pa)} |")
    lines += [
        "",
        "## 조기 종료로 고른 나무 수(F, 모든 학습)",
        "",
        "| 모델 | 학습 수 | 중앙값 | 10%~90% | 바깥 재학습 중앙값 | 대체 |",
        "|---|---|---|---|---|---|",
    ]
    for n, st in r["arm_f"]["es_stats"].items():
        lines.append(
            f"| `{n}` | {st['fits']} | {_f(st['median'], '.0f')} | {_f(st['p10'], '.0f')}~{_f(st['p90'], '.0f')} | "
            f"{_f(st['outer_refit_median'], '.0f')} | {st['fallback']} |"
        )
    lines += [
        "",
        "## 묶음 (채움 = 월말 행의 특징별 채움 비율 중앙값)",
        "",
        "| 묶음 | 특징 | 국면 | 채움 | 오류 |",
        "|---|---|---|---|---|",
    ]
    for t, st in pool["steps"].items():
        lines.append(f"| {t} | {st['n_xs']} | {st['n_reg']} | {st['coverage']} | {st['error'] or ''} |")
    p = OUT / f"{m}_v86{r.get('variant', '')}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def group_leak(ds, xs: list[str], m: str) -> dict[str, float]:
    """86.6: the missing-indicator AUC inside each market-cap group (KR top 200 / 201-400 by the date's rank). The
    pooled screen kept investor-flow features fetched only around old top-200 spells (present in 201-400 = about to
    enter the top 200). Returns {feature: largest |AUC - 0.5|} for the features above the 31.2 threshold."""
    import v13_study as v13s

    if m != "KR":
        return {}
    f = ds.frame
    me = f[f["labelled"].to_numpy(bool) & f["is_me"].to_numpy(bool)]
    top = (me.groupby("date")["mcap"].rank(ascending=False, method="first") <= 200).to_numpy()
    out = {}
    for x in xs:
        devs = [abs(a - 0.5) for g in (me[top], me[~top]) if (a := v13s.missing_auc(g, x)) is not None]
        if devs and max(devs) > v13s.LEAK_AUC:
            out[x] = round(max(devs), 4)
    return out


def _arm(res, ds, cands, es_path: Path, arm: str, outer_starts: set[str]) -> dict:
    import serve_override as so

    d = res.to_dict()
    names = [c.name for c in cands if c.name != sel.BASELINE]
    return {
        "selected": d["selected"],
        "gate": d["gate"],
        "holdout_best": so.holdout_best(res),
        "table": table(res),
        "insample": insample(res, ds, names),
        "es_stats": es_stats(es_path, arm, outer_starts),
    }


def study(m: str, variant: str = "") -> None:
    """Arms A and F (variant "p": F' = F minus the group leak screen's features, 86.6; arm A comes from the cache)."""
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = load_frozen(m, s, cache)
    patch_selection()
    cands = candidates()
    rd = research_dir(s, m)
    ES_LOG[:] = [rd / "es_log.jsonl"]
    snap0, xs0, reg0 = ds.snapshot, list(ds.xs_cols), list(ds.regime_cols)
    cost = float(ds.cost_bps)
    names = [c.name for c in cands if c.name != sel.BASELINE]
    data = sel._Data(ds)
    outer, holdout, final, _ = sel.make_contexts(data)
    outer_starts = {str(pd.Timestamp(c.test_dates[0]).date()) for c in [*outer, holdout, final]}
    del data
    # ---- arm A: current inputs
    ARM[0] = "A"
    ds.snapshot = f"{snap0}|v86A|es{ES_VERSION}"
    say(f"arm A: {len(xs0)} features, {len(cands)} candidates")
    res_a = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    arm_a = _arm(res_a, ds, cands, ES_LOG[0], "A", outer_starts)
    (rd / "arm_a.json").write_text(json.dumps(arm_a, ensure_ascii=False, indent=1, default=str))
    # ---- arm F: + the full pool
    arm = "F" + variant
    ARM[0] = arm
    ds.snapshot = snap0  # the pool was built on the guarded snapshot
    new_xs, new_reg, pool = load_pool(m, ds, s)
    if variant == "p":
        drop = group_leak(ds, new_xs, m)
        say(f"group leak screen: {len(drop)} dropped {drop}")
        new_xs = [x for x in new_xs if x not in drop]
        pool = {**pool, "group_leak": drop}
    ds.xs_cols, ds.regime_cols = [*xs0, *new_xs], [*reg0, *new_reg]
    ds.snapshot = f"{snap0}|v86{arm}|es{ES_VERSION}|{len(new_xs)}|{len(new_reg)}"
    say(f"arm {arm}: {len(ds.xs_cols)} features, {len(ds.regime_cols)} regime")
    res_f = sel.run_selection(ds, s, cache, candidates=cands, threads=8, use_cache=True, progress=say)
    arm_f = _arm(res_f, ds, cands, ES_LOG[0], arm, outer_starts)
    pairs = [(f"{n}_es", n) for n in ES_BASES]
    v85 = json.loads((OUT / f"{m}_v85.json").read_text())["arm_a"]
    repro = {n: (round(arm_a["table"][n]["outer"], 6), round(v85[n], 6)) for n in V85_CANDS}
    r = {
        "market": m,
        "variant": variant,
        "snapshot": snap0,
        "cost_bps": cost,
        "es": {
            "version": ES_VERSION,
            "max": ES_MAX,
            "patience": ES_PATIENCE,
            "min": ES_MIN,
            "val": [ES_VAL_FRAC, ES_VAL_MIN, ES_VAL_MAX],
        },
        "pool": {k: v for k, v in pool.items() if k != "missing_auc"},
        "arm_a": arm_a,
        "arm_f": arm_f,
        "es_f": averaged(pairs, res_f, res_f, cost),
        "es_a": averaged(pairs, res_a, res_a, cost),
        "full": averaged([(n, n) for n in names], res_f, res_a, cost),
        "repro": "같음" if all(abs(a - b) < 1e-9 for a, b in repro.values()) else f"다름 {repro}",
    }
    js = OUT / f"{m}_v86{variant}.json"
    js.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    say(f"wrote {write_md(m, r)} and {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "study"])
    ap.add_argument("--market", choices=["KR", "US"], required=True)
    ap.add_argument("--variant", choices=["", "p"], default="", help="p: F' = F minus the group leak screen (86.6)")
    a = ap.parse_args()
    if a.cmd == "pool":
        build_pool(a.market)
    else:
        study(a.market, a.variant)


if __name__ == "__main__":
    main()
