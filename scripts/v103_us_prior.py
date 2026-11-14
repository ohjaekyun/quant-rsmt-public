"""US 103 (docs/MODELING.md 103, pre-declared 2026-11-13 22:45): pick signals on the long, survivorship-free history
of the Kenneth French data library (big-cap rows of the size x characteristic sorts), combine the adopted ones with
equal weights (no fitting) and check only the direction on our US PIT data. Research only: src and serving unchanged.

    .venv/bin/python scripts/v103_us_prior.py fetch     # download the library files (kept under .cache)
    .venv/bin/python scripts/v103_us_prior.py french    # rule 10 on the library (long 1963-07..2011-12, recent 2012+)
    .venv/bin/python scripts/v103_us_prior.py ours      # composite on the US dataset: rules 11 and 12

Rule 10: big-row favourable-end portfolio regressed on the big-row average (both minus RF), monthly value-weighted
returns; adopted when the long-window alpha t >= 2.8 and the recent alpha > 0.
Composite: equal-weight mean over adopted groups of the dataset's per-date rank scores (x_*; missing = 0).
Writes reports/modeling/US_v103_{french,prior}.{json,md}.
"""

from __future__ import annotations

import os

for _k in ("KRX_ID", "KRX_PW"):  # before anything imports pykrx
    os.environ.pop(_k, None)

import argparse  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "modeling"
RAW = ROOT / ".cache" / "quantcio" / "research" / "french"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
UA = "Mozilla/5.0 (compatible; research bot)"
FACTORS = "F-F_Research_Data_Factors"
# name: (file, favourable end, group, our column, sign) - the last three are None for report-only sorts
SIGNALS = {
    "momentum": ("25_Portfolios_ME_Prior_12_2", "high", "momentum", "x_mom_12_1", 1),
    "st_reversal": ("25_Portfolios_ME_Prior_1_0", "low", "reversal", "x_ret_1m", -1),
    "low_var": ("25_Portfolios_ME_VAR_5x5", "low", "risk", "x_vol60", -1),
    "low_resvar": ("25_Portfolios_ME_RESVAR_5x5", "low", "risk", "x_ivol60", -1),
    "low_beta": ("25_Portfolios_ME_BETA_5x5", "low", "risk", "x_beta", -1),
    "profitability": ("25_Portfolios_ME_OP_5x5", "high", "profitability", "x_roe", 1),
    "value_bm": ("25_Portfolios_5x5", "high", "value", "x_bp", 1),
    "value_ep": ("6_Portfolios_ME_EP_2x3", "high", "value", "x_ep", 1),
    "net_issues": ("25_Portfolios_ME_NI_5x5", "low", None, None, None),
    "investment": ("25_Portfolios_ME_INV_5x5", "low", None, None, None),
    "accruals": ("25_Portfolios_ME_AC_5x5", "low", None, None, None),
    "lt_reversal": ("25_Portfolios_ME_Prior_60_13", "low", None, None, None),
}
LONG = ("1963-07", "2011-12")
RECENT_FROM = "2012-01"
T_ADOPT = 2.8
SNAPSHOT = "32176fe480acdc5f319e9053ec51bfe2"  # the 95 US dataset (as_of 2026-11-06, frozen digest "match")
VW_MONTHLY = "value weight returns monthly"  # words of the value-weighted monthly table's title
WIDTH = {25: 5, 6: 3, 35: 7}  # columns -> portfolios per size group (net issues: Neg, Zero, Lo, 2, 3, 4, Hi)


def say(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------ French library


def fetch() -> None:
    import requests

    RAW.mkdir(parents=True, exist_ok=True)
    for name in [FACTORS, *dict.fromkeys(v[0] for v in SIGNALS.values())]:
        p = RAW / f"{name}_CSV.zip"
        if p.exists() and p.stat().st_size > 1000:
            say(f"{name}: cached")
            continue
        r = requests.get(f"{BASE}{name}_CSV.zip", headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
        say(f"{name}: {len(r.content):,} bytes")
        time.sleep(2.0)
    (RAW / "fetched_at.txt").write_text(datetime.now().isoformat(timespec="seconds"))


def read_table(name: str, title: str | None) -> pd.DataFrame:
    """One monthly table of a library file (percent returns), index = month period. ``title`` None = the first
    monthly block (the factors file)."""
    with zipfile.ZipFile(RAW / f"{name}_CSV.zip") as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = 0
    if title is not None:  # the files spell it "Average Value Weighted Returns" or "Value Weight Average Returns"
        keys = title.lower().split()
        hit = [i for i, ln in enumerate(lines) if all(k in ln.lower() for k in keys)]
        if not hit:
            raise SystemExit(f"{name}: no table '{title}'")
        start = hit[0] + 1
    while start < len(lines) and not lines[start].lstrip().startswith(","):
        start += 1
    rows = []
    for ln in lines[start + 1 :]:
        a = ln.split(",")
        k = a[0].strip()
        if not (len(k) == 6 and k.isdigit()):
            break
        rows.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([lines[start], *rows])), index_col=0)
    df.columns = [c.strip() for c in df.columns]
    df.index = pd.PeriodIndex([str(i).strip() for i in df.index], freq="M")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.mask(df <= -99.0)


def ols_alpha(y: pd.Series, x: pd.Series) -> dict:
    d = pd.concat([y, x], axis=1).dropna()
    yy, xx = d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float)
    X = np.column_stack([np.ones(len(xx)), xx])
    b, *_ = np.linalg.lstsq(X, yy, rcond=None)
    e = yy - X @ b
    se = np.sqrt(np.diag(e @ e / (len(yy) - 2) * np.linalg.inv(X.T @ X)))
    return {"alpha_ann": float(12 * b[0]), "beta": float(b[1]), "t": float(b[0] / se[0]), "months": int(len(yy))}


def french() -> None:
    rf_all = read_table(FACTORS, None)["RF"]
    out = {"long": list(LONG), "recent_from": RECENT_FROM, "t_adopt": T_ADOPT, "signals": {}}
    out["fetched_at"] = (RAW / "fetched_at.txt").read_text() if (RAW / "fetched_at.txt").exists() else None
    for sig, (name, end, group, col, _sign) in SIGNALS.items():
        t = read_table(name, VW_MONTHLY)
        rf = rf_all.reindex(t.index)
        n = t.shape[1]
        if n not in WIDTH:
            raise SystemExit(f"{name}: {n} columns (expected one of {sorted(WIDTH)})")
        w = WIDTH[n]
        rows = {"big": t.iloc[:, n - w :], "me4": t.iloc[:, n - 2 * w : n - w] if n != 6 else None}
        res = {"file": name, "favourable": end, "group": group, "our_column": col, "big_columns": list(rows["big"])}
        for rk, r in rows.items():
            if r is None:
                continue
            bench = r.mean(axis=1, skipna=False)
            top = r.iloc[:, -1] if end == "high" else r.iloc[:, 0]
            bot = r.iloc[:, 0] if end == "high" else r.iloc[:, -1]
            for wk, (a, b) in {"long": LONG, "recent": (RECENT_FROM, str(t.index[-1]))}.items():
                m = (t.index >= pd.Period(a, "M")) & (t.index <= pd.Period(b, "M"))
                x = (bench - rf)[m]
                res[f"{rk}_{wk}_top"] = ols_alpha((top - rf)[m], x)
                res[f"{rk}_{wk}_bottom"] = ols_alpha((bot - rf)[m], x)
                res[f"{rk}_{wk}_span"] = [str(t.index[m][0]), str(t.index[m][-1])]
        a, b = res["big_long_top"], res["big_recent_top"]
        res["adopted"] = bool(col is not None and a["t"] >= T_ADOPT and b["alpha_ann"] > 0)
        res["report_only"] = col is None
        out["signals"][sig] = res
        say(
            f"{sig:14s} big top: long alpha {a['alpha_ann']:+.2f}%/yr t {a['t']:+.2f} ({a['months']} mo) | recent "
            f"{b['alpha_ann']:+.2f}% t {b['t']:+.2f} ({b['months']} mo) -> {'ADOPT' if res['adopted'] else '-'}"
        )
    out["adopted"] = [s for s, r in out["signals"].items() if r["adopted"]]
    out["groups"] = sorted({SIGNALS[s][2] for s in out["adopted"]})
    (OUT / "US_v103_french.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [
        "# US 103.1: 프렌치 자료 대형주 줄 — 롱 쪽 알파 (월간 시가가중, 기준 = 같은 줄 평균)",
        "",
        f"긴 과거 {LONG[0]}~{LONG[1]}, 최근 {RECENT_FROM}~. 채택 = 긴 과거 t ≥ {T_ADOPT} 이고 최근 알파 > 0.",
        "",
        "| 신호 | 긴 과거 상위 알파(연) | t | 최근 상위 알파 | t | 긴 과거 하위 알파 | t | 최근 하위 알파 | t "
        "| ME4 긴 과거 t | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s, r in out["signals"].items():
        a, b, c, d = r["big_long_top"], r["big_recent_top"], r["big_long_bottom"], r["big_recent_bottom"]
        m4 = f"{r['me4_long_top']['t']:+.2f}" if "me4_long_top" in r else "—"
        verdict = "채택" if r["adopted"] else ("보고만" if r["report_only"] else "탈락")
        lines.append(
            f"| {s} | {a['alpha_ann']:+.2f}% | {a['t']:+.2f} | {b['alpha_ann']:+.2f}% | {b['t']:+.2f} | "
            f"{c['alpha_ann']:+.2f}% | {c['t']:+.2f} | {d['alpha_ann']:+.2f}% | {d['t']:+.2f} | {m4} | {verdict} |"
        )
    lines += ["", f"채택: {', '.join(out['adopted']) or '없음'} (묶음: {', '.join(out['groups']) or '없음'})"]
    (OUT / "US_v103_french.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"adopted: {out['adopted']}")


# ------------------------------------------------------------------------------------------------ our data


def composite(frame: pd.DataFrame, adopted: list[str]) -> pd.Series:
    groups: dict[str, list[pd.Series]] = {}
    for s in adopted:
        _, _, group, col, sign = SIGNALS[s]
        groups.setdefault(group, []).append(sign * frame[col].astype(float).fillna(0.0))
    return pd.concat([pd.concat(v, axis=1).mean(axis=1) for v in groups.values()], axis=1).mean(axis=1)


def book(d: pd.DataFrame, score: str, lo: float, hi: float, cost: float) -> pd.DataFrame:
    """Per month-end: equal-weight mean simple return of the names with lo <= percentile < hi, one-way turnover."""
    g = d.groupby("date")
    u = (g[score].rank(method="average") - 0.5) / g[score].transform("size")
    pick = d[(u >= lo) & (u < hi)]
    dates = sorted(d["date"].unique())
    names = pick.groupby("date")["entity"].apply(set).reindex(dates)
    turn, prev = [], None
    for x in names:
        if not isinstance(x, set) or not x:
            turn.append(np.nan)
            prev = None
            continue
        turn.append(1.0 if prev is None else len(x - prev) / len(x))
        prev = x
    r = pick.groupby("date")["R"].mean().reindex(dates)
    out = pd.DataFrame({"gross": r.to_numpy(), "turn": turn}, index=pd.DatetimeIndex(dates))
    out["net"] = out["gross"] - cost * out["turn"]
    return out


def describe(r: pd.Series, u: pd.Series, turn: pd.Series | None = None) -> dict:
    x = r.fillna(0.0)
    nav = np.cumprod(1 + x.to_numpy())
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    e = (r - u).fillna(0.0)
    out = {
        "ann": float(nav[-1] ** (12 / len(x)) - 1),
        "mdd": float((nav / peak - 1).min()),
        "t_vs_universe": float(e.mean() / e.std(ddof=1) * np.sqrt(len(e))),
        **ols_alpha(r, u),
    }
    if turn is not None:
        out["turnover"] = float(turn.iloc[1:].mean())
    return out


def ours() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache
    from quantcio.modeling import metrics as mt
    from quantcio.modeling.cli import guard_xsh

    fr = json.loads((OUT / "US_v103_french.json").read_text())
    adopted = fr["adopted"]
    s = load_settings()
    cache = DiskCache(s.cache_path)
    ds = cache.get("modeling_dataset", SNAPSHOT, None)
    if ds is None:
        raise SystemExit(f"dataset {SNAPSHOT} is not cached")
    dig = (ds.coverage.get("vintage") or {}).get("digest") or {}
    if dig.get("status") != "match":
        raise SystemExit(f"frozen-window digest is not a match: {dig}")
    ds = guard_xsh(ds, s, say)
    f = ds.frame
    d = f[f["is_me"].astype(bool) & f["labelled"].astype(bool)].dropna(subset=["r"]).copy()
    d = d[d.groupby("date")["entity"].transform("size") >= mt.MIN_NAMES]
    d["R"] = np.expm1(d["r"].to_numpy(float))
    cost = float(ds.cost_bps) / 1e4
    uni = d.groupby("date")["R"].mean()
    uni.index = pd.DatetimeIndex(uni.index)
    months = len(uni)
    half = (months + 1) // 2
    say(
        f"US {ds.snapshot[:8]} as_of {ds.as_of.date()}: {months} month-ends {uni.index[0].date()}.."
        f"{uni.index[-1].date()}, median {int(d.groupby('date').size().median())} names, "
        f"cost {1e4 * cost:.0f} bp, adopted {adopted}"
    )
    out = {
        "snapshot": ds.snapshot,
        "as_of": str(ds.as_of.date()),
        "months": months,
        "span": [str(uni.index[0].date()), str(uni.index[-1].date())],
        "cost_bps": 1e4 * cost,
        "adopted": adopted,
        "groups": fr["groups"],
        "universe_ann": float((1 + uni).prod() ** (12 / months) - 1),
        "books": {},
        "rule": {},
    }
    scores = {}
    if adopted:
        d["composite"] = composite(d, adopted)
        scores["composite"] = "composite"
        for sig in adopted:
            d[f"s_{sig}"] = SIGNALS[sig][4] * d[SIGNALS[sig][3]].astype(float).fillna(0.0)
            scores[sig] = f"s_{sig}"
    d["s_mom_only"] = d["x_mom_12_1"].astype(float).fillna(0.0)
    scores.setdefault("mom_12_1", "s_mom_only")
    for name, col in scores.items():
        for bk, (lo, hi) in {"Q5": (0.80, 1.01), "D20": (0.95, 1.01), "Q1": (0.0, 0.20)}.items():
            b = book(d, col, lo, hi, cost)
            e = {"gross": describe(b["gross"], uni), "net": describe(b["net"], uni, b["turn"])}
            e["net_first_half"] = describe(b["net"].iloc[:half], uni.iloc[:half])
            e["net_second_half"] = describe(b["net"].iloc[half:], uni.iloc[half:])
            out["books"][f"{name}/{bk}"] = e
            say(
                f"{name:14s} {bk:3s}: net {100 * e['net']['ann']:+5.1f}%/yr beta {e['net']['beta']:.2f} alpha "
                f"{100 * e['net']['alpha_ann']:+5.1f}% t {e['net']['t']:+.2f} | halves "
                f"{100 * e['net_first_half']['alpha_ann']:+.1f}% / {100 * e['net_second_half']['alpha_ann']:+.1f}% | "
                f"turn {100 * e['net']['turnover']:.0f}% mdd {100 * e['net']['mdd']:.0f}%"
            )
    if adopted:
        q = out["books"]["composite/Q5"]
        out["rule"]["11_significant_in_our_data"] = bool(q["net"]["t"] >= 2)
        out["rule"]["12_show_as_reference"] = bool(
            q["net_first_half"]["alpha_ann"] > 0 and q["net_second_half"]["alpha_ann"] > 0
        )
        out["half_months"] = [half, months - half]
    else:
        out["rule"] = {"11_significant_in_our_data": False, "12_show_as_reference": False, "note": "nothing adopted"}
    (OUT / "US_v103_prior.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    lines = [
        f"# US 103.3: 합성 점수의 우리 자료 확인 ({out['span'][0]}~{out['span'][1]}, {months}개월)",
        "",
        f"채택 신호: {', '.join(adopted) or '없음'}. 전체(동일가중) 연 {100 * out['universe_ann']:+.1f}%, 비용 "
        f"{out['cost_bps']:.0f}bp × 교체.",
        f"판정 11(Q5 알파 t ≥ 2): {out['rule']['11_significant_in_our_data']} · 판정 12(앞·뒤 절반 알파 모두 양수): "
        f"{out['rule']['12_show_as_reference']}",
        "",
        "| 장부 | 비용 후 연 | 베타 | 알파(연) | 알파 t | 앞 절반 알파 | 뒤 절반 알파 | 월 교체 | 최대 낙폭 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, e in out["books"].items():
        n = e["net"]
        lines.append(
            f"| `{k}` | {100 * n['ann']:+.1f}% | {n['beta']:.2f} | {100 * n['alpha_ann']:+.1f}% | {n['t']:+.2f} | "
            f"{100 * e['net_first_half']['alpha_ann']:+.1f}% | {100 * e['net_second_half']['alpha_ann']:+.1f}% | "
            f"{100 * n['turnover']:.0f}% | {100 * n['mdd']:.0f}% |"
        )
    (OUT / "US_v103_prior.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"rules: {out['rule']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("fetch", "french", "ours"))
    a = ap.parse_args()
    {"fetch": fetch, "french": french, "ours": ours}[a.cmd]()


if __name__ == "__main__":
    main()
