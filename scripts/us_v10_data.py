"""US v10 block M (docs/MODELING.md section 28): analyst rating / price-target changes, FINRA short interest,
off-exchange volume share, shareholder yield, Piotroski F-score, operating accruals, peer lead-lag momentum and the
CBOE equity put/call ratio. Plus forward ledgers (option-implied signals, EPS-estimate trends): no free history.
Research code outside ``src/quantcio``.
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import us_v4_data as v4  # noqa: E402

from quantcio.data.pit.us_net import PitNet  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (research; quantcio)"}
FINRA_SI = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
SI_LAG_DAYS = 14
CBOE_CSV = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
CBOE_DAILY = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
M_CONCEPTS = {
    "buyback": ["PaymentsForRepurchaseOfCommonStock"],
    "issuance": ["ProceedsFromIssuanceOfCommonStock", "ProceedsFromStockOptionsExercised"],
    "dividends": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
    "op_income": ["OperatingIncomeLoss"],
    "cfo": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "assets": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liab_current": ["LiabilitiesCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "gross_profit": ["GrossProfit"],
    "revenue": list(v4.REVENUE),
    "shares": ["WeightedAverageNumberOfSharesOutstandingBasic"],
}
M_INSTANT = {"assets", "assets_current", "liab_current", "lt_debt"}
FEATURES = [
    "an_rev_30",
    "an_rev_90",
    "an_pt_30",
    "an_pt_90",
    "an_activity",
    "si_dtc",
    "si_chg",
    "si_float",
    "offex_20",
    "offex_chg",
    "sh_yield",
    "f_score",
    "op_accruals",
    "peer_lead_5",
    "peer_lead_10",
    "peer_gap_10",
]
DOCS = {
    "an_rev_30": ("애널리스트 순상향 비율(30일)", "analyst net upgrades (30d)"),
    "an_rev_90": ("애널리스트 순상향 비율(90일)", "analyst net upgrades (90d)"),
    "an_pt_30": ("목표가 변경률(30일, log)", "price-target revisions (30d)"),
    "an_pt_90": ("목표가 변경률(90일, log)", "price-target revisions (90d)"),
    "an_activity": ("애널리스트 의견 건수(90일, log)", "analyst activity (90d)"),
    "si_dtc": ("공매도 잔고 ÷ 평균 거래량(Days-to-Cover)", "days to cover"),
    "si_chg": ("공매도 잔고 변화(log)", "short-interest change"),
    "si_float": ("공매도 잔고 ÷ 발행주식수", "short interest / shares"),
    "offex_20": ("장외(TRF) 거래 비중(20일)", "off-exchange volume share (20d)"),
    "offex_chg": ("장외 거래 비중 변화(20일 − 120일)", "off-exchange share change"),
    "sh_yield": ("주주환원 수익률(배당 + 순자사주 매입)", "shareholder yield"),
    "f_score": ("Piotroski F-Score(0~9)", "Piotroski F-score"),
    "op_accruals": ("영업 발생액((영업이익 − 영업현금흐름) ÷ 자산)", "operating accruals"),
    "peer_lead_5": ("동종 산업 상위 3개사 5일 수익률", "peer leaders 5-day return"),
    "peer_lead_10": ("동종 산업 상위 3개사 10일 수익률", "peer leaders 10-day return"),
    "peer_gap_10": ("자기 10일 수익률 − 동종 선도 10일 수익률", "own minus peer-leader 10-day return"),
}
REGIME = ["r_cboe_pc"]


def _say(_m: str) -> None:
    return None


# ================================================================================================ analysts


def fetch_analyst(settings, tickers: list[str], say=_say) -> dict:
    """yfinance upgrades / downgrades history per ticker (cached parquet; refreshed after 7 days)."""
    import yfinance as yf

    d = kd.root(settings, "analyst")
    n = 0
    for i, t in enumerate(sorted(set(tickers)), 1):
        p = d / f"{re.sub(r'[^A-Za-z0-9._-]', '_', t)}.parquet"
        if p.exists() and time.time() - p.stat().st_mtime < 7 * 86400:
            continue
        for attempt in range(4):
            try:
                ud = yf.Ticker(t.replace(".", "-")).upgrades_downgrades
                break
            except Exception:  # noqa: BLE001 - Yahoo throttling: wait and retry
                ud = None
                time.sleep(5 * (attempt + 1))
        df = ud.reset_index() if ud is not None and len(ud) else pd.DataFrame(columns=["GradeDate", "Action"])
        df.to_parquet(p, index=False)
        n += 1
        time.sleep(0.3)
        if i % 100 == 0:
            say(f"analyst {i}/{len(set(tickers))}")
    return {"tickers": len(set(tickers)), "downloaded": n}


def analyst_events(settings, tickers: set[str]) -> pd.DataFrame:
    parts = []
    for t in tickers:
        p = kd.root(settings, "analyst") / f"{re.sub(r'[^A-Za-z0-9._-]', '_', t)}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["when", "action", "pt"])
    df = pd.concat(parts, ignore_index=True)
    when = pd.to_datetime(df["GradeDate"]).dt.tz_localize(None)
    cur = pd.to_numeric(df.get("currentPriceTarget"), errors="coerce")
    pri = pd.to_numeric(df.get("priorPriceTarget"), errors="coerce")
    pt = np.log(cur / pri).where((cur > 0) & (pri > 0))
    out = pd.DataFrame({"when": when, "action": df["Action"].astype(str).str.lower(), "pt": pt})
    return out.drop_duplicates().sort_values("when").reset_index(drop=True)


def analyst_features(dates: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
    """an_rev_30, an_rev_90, an_pt_30, an_pt_90, an_activity at each date (events strictly before the date)."""
    out = np.full((len(dates), 5), np.nan)
    if ev.empty:
        return out
    w = ev["when"].to_numpy(dtype="datetime64[ns]")
    up = (ev["action"] == "up").to_numpy(float)
    down = (ev["action"] == "down").to_numpy(float)
    pt = ev["pt"].to_numpy(float)
    has_pt = np.isfinite(pt).astype(float)
    cs = {
        k: np.concatenate([[0.0], np.cumsum(v)])
        for k, v in (("up", up), ("down", down), ("n", np.ones(len(w))), ("pt", np.nan_to_num(pt)), ("npt", has_pt))
    }
    for i, d in enumerate(dates):
        hi = int(np.searchsorted(w, d, "left"))
        for j, days in ((0, 30), (1, 90)):
            lo = int(np.searchsorted(w, d - np.timedelta64(days, "D"), "left"))
            n = cs["n"][hi] - cs["n"][lo]
            if n > 0:
                out[i, j] = (cs["up"][hi] - cs["up"][lo] - (cs["down"][hi] - cs["down"][lo])) / n
            npt = cs["npt"][hi] - cs["npt"][lo]
            if npt > 0:
                out[i, 2 + j] = (cs["pt"][hi] - cs["pt"][lo]) / npt
        lo90 = int(np.searchsorted(w, d - np.timedelta64(90, "D"), "left"))
        out[i, 4] = np.log1p(cs["n"][hi] - cs["n"][lo90]) if hi > 0 and w[0] <= d - np.timedelta64(365, "D") else np.nan
    return out


# ================================================================================================ short interest


def fetch_short_interest(settings, tickers: list[str], say=_say) -> dict:
    """FINRA consolidated short interest per symbol, 2017 .. (cached parquet per symbol, refreshed after 7 days)."""
    d = kd.root(settings, "finra_si")
    n = 0
    for i, t in enumerate(sorted(set(tickers)), 1):
        p = d / f"{re.sub(r'[^A-Za-z0-9._-]', '_', t)}.parquet"
        if p.exists() and time.time() - p.stat().st_mtime < 7 * 86400:
            continue
        body = {
            "limit": 1000,
            "compareFilters": [{"fieldName": "symbolCode", "fieldValue": t.replace("-", "."), "compareType": "EQUAL"}],
            "dateRangeFilters": [
                {"fieldName": "settlementDate", "startDate": "2017-01-01", "endDate": str(date.today())}
            ],
        }
        r = None
        for attempt in range(6):  # transient resets / throttling: wait and retry
            try:
                r = requests.post(FINRA_SI, json=body, headers={**UA, "Accept": "application/json"}, timeout=60)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code in (200, 204):
                break
            time.sleep(10 * (attempt + 1))
        if r is None or r.status_code not in (200, 204):
            say(f"FINRA short interest {t}: skipped (retry next run)")
            continue
        rows = r.json() if r.status_code == 200 and r.text.strip() else []
        df = pd.DataFrame(rows)
        keep = [
            "settlementDate",
            "currentShortPositionQuantity",
            "previousShortPositionQuantity",
            "averageDailyVolumeQuantity",
            "daysToCoverQuantity",
        ]
        df = df[[c for c in keep if c in df]] if len(df) else pd.DataFrame(columns=keep)
        df.to_parquet(p, index=False)
        n += 1
        time.sleep(0.2)
        if i % 100 == 0:
            say(f"FINRA short interest {i}/{len(set(tickers))}")
    return {"tickers": len(set(tickers)), "downloaded": n}


def short_interest(settings, tickers: set[str]) -> pd.DataFrame:
    parts = []
    for t in tickers:
        p = kd.root(settings, "finra_si") / f"{re.sub(r'[^A-Za-z0-9._-]', '_', t)}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if len(df):
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["avail", "si", "si_prev", "dtc"])
    df = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame(
        {
            "avail": pd.to_datetime(df["settlementDate"]) + pd.Timedelta(SI_LAG_DAYS, "D"),
            "si": pd.to_numeric(df["currentShortPositionQuantity"], errors="coerce"),
            "si_prev": pd.to_numeric(df["previousShortPositionQuantity"], errors="coerce"),
            "dtc": pd.to_numeric(df["daysToCoverQuantity"], errors="coerce"),
        }
    )
    return out.dropna(subset=["si"]).sort_values("avail").drop_duplicates("avail", keep="last").reset_index(drop=True)


# ====================================================================== XBRL (shareholder yield, F-score, accruals)


def fetch_facts_m(net: PitNet, ciks, say=_say) -> dict[int, dict]:
    want = {c for cs in M_CONCEPTS.values() for c in cs}
    out = {}
    ciks = sorted({int(c) for c in ciks})
    for i, cik in enumerate(ciks, 1):
        rel = f"facts_m/{cik}.json.gz"
        j = net.read_json("sec", rel)
        if j is None:
            r = net.http_get(COMPANYFACTS.format(cik=cik), "sec", ok=(200, 404))
            facts = {}
            if r.status_code == 200:
                js = r.json().get("facts") or {}
                for ns in ("us-gaap", "dei"):
                    for c, v in (js.get(ns) or {}).items():
                        if c in want:
                            units = v.get("units") or {}
                            lst = units.get("USD") or units.get("shares") or []
                            facts[c] = [{k: f.get(k) for k in ("start", "end", "val", "form", "filed")} for f in lst]
            j = {"cik": cik, "facts": facts}
            net.write_json("sec", rel, j)
        out[cik] = j
        if i % 100 == 0:
            say(f"XBRL (block M) {i}/{len(ciks)}")
    return out


def annual_table(facts: dict) -> pd.DataFrame:
    """One row per fiscal-year end (first-filed annual values) with every block-M item and the latest filed date."""
    old = dict(v4.CONCEPTS)
    old_inst = set(v4.INSTANT)
    try:
        v4.CONCEPTS.clear()
        v4.CONCEPTS.update(M_CONCEPTS)
        v4.INSTANT.clear()
        v4.INSTANT.update(M_INSTANT)
        tabs = {k: v4.first_filed(facts, k, annual=True) for k in M_CONCEPTS}
    finally:
        v4.CONCEPTS.clear()
        v4.CONCEPTS.update(old)
        v4.INSTANT.clear()
        v4.INSTANT.update(old_inst)
    base = tabs["assets"]
    recs = []
    for r in base.itertuples(index=False):
        rec = {"end": r.end, "filed": r.filed, "assets": r.value}
        for k, t in tabs.items():
            if k == "assets":
                continue
            m = v4._match(t, r.end)
            rec[k] = float(m["value"]) if m is not None else np.nan
            if m is not None:
                rec["filed"] = max(rec["filed"], m["filed"])
        recs.append(rec)
    return pd.DataFrame(recs).sort_values("end").reset_index(drop=True) if recs else pd.DataFrame()


def fundamental_features(t: pd.DataFrame) -> pd.DataFrame:
    """Per fiscal year: payout (dividends + buyback - issuance, $), f_score, op_accruals; usable from filed + 1 day."""
    if t.empty:
        return pd.DataFrame(columns=["avail", "end", "payout", "f_score", "op_accruals"])
    t = t.copy()
    p = t.shift(1)
    a = t["assets"]
    roa = t["net_income"] / a
    roa_p = p["net_income"] / p["assets"]
    cfo_a = t["cfo"] / a
    lev = t["lt_debt"].fillna(0) / a
    lev_p = p["lt_debt"].fillna(0) / p["assets"]
    cur = t["assets_current"] / t["liab_current"]
    cur_p = p["assets_current"] / p["liab_current"]
    gm = t["gross_profit"] / t["revenue"]
    gm_p = p["gross_profit"] / p["revenue"]
    turn = t["revenue"] / a
    turn_p = p["revenue"] / p["assets"]
    sig = [
        roa > 0,
        cfo_a > 0,
        roa > roa_p,
        t["cfo"] > t["net_income"],
        lev < lev_p,
        cur > cur_p,
        ~(t["shares"] > p["shares"] * 1.005),
        gm > gm_p,
        turn > turn_p,
    ]
    known = [
        roa.notna(),
        cfo_a.notna(),
        roa_p.notna(),
        t["cfo"].notna() & t["net_income"].notna(),
        lev_p.notna(),
        cur.notna() & cur_p.notna(),
        t["shares"].notna() & p["shares"].notna(),
        gm.notna() & gm_p.notna(),
        turn_p.notna(),
    ]
    n_known = sum(k.astype(int) for k in known)
    score = sum((s & k).astype(int) for s, k in zip(sig, known, strict=True))
    f = (score * 9.0 / n_known).where(n_known >= 6)  # scaled to 0..9 when a few items are missing
    payout = t["dividends"].fillna(0) + t["buyback"].fillna(0) - t["issuance"].fillna(0)
    payout = payout.where(t["cfo"].notna())
    acc = ((t["op_income"] - t["cfo"]) / a).where(t["op_income"].notna() & t["cfo"].notna())
    return pd.DataFrame(
        {
            "avail": t["filed"] + pd.Timedelta(1, "D"),
            "end": t["end"],
            "payout": payout,
            "f_score": f,
            "op_accruals": acc,
        }
    )


# ================================================================================================ CBOE equity put/call


def fetch_cboe(settings, say=_say) -> pd.Series:
    """Daily CBOE equity put/call ratio: the historical CSV (2006 .. 2019-10) + the daily statistics files after it."""
    d = kd.root(settings, "cboe")
    hist = d / "equitypc.csv"
    if not hist.exists():
        hist.write_bytes(requests.get(CBOE_CSV, headers=UA, timeout=120).content)
    lines = [ln for ln in hist.read_text(errors="replace").splitlines() if re.match(r"\s*\d{1,2}/\d{1,2}/\d{4}", ln)]
    vals = {}
    for ln in lines:
        p = [x.strip() for x in ln.split(",")]
        try:
            vals[pd.Timestamp(pd.to_datetime(p[0], format="%m/%d/%Y"))] = float(p[4])
        except (ValueError, IndexError):
            continue
    daily = d / "daily.json"
    got = json.loads(daily.read_text()) if daily.exists() else {}
    start = max(vals) + timedelta(1) if vals else pd.Timestamp("2019-10-07")
    for day in pd.bdate_range(start, date.today() - timedelta(1)):
        k = day.strftime("%Y-%m-%d")
        if k in got:
            continue
        r = requests.get(CBOE_DAILY.format(d=k), headers=UA, timeout=60)
        if r.status_code == 200:
            try:
                ratios = {x["name"]: x["value"] for x in r.json().get("ratios", [])}
                v = ratios.get("EQUITY PUT/CALL RATIO")
                got[k] = float(v) if v not in (None, "") else None
            except ValueError:
                got[k] = None
        else:
            got[k] = None
        time.sleep(0.15)
    daily.write_text(json.dumps(got))
    for k, v in got.items():
        if v is not None:
            vals[pd.Timestamp(k)] = v
    s = pd.Series(vals).sort_index()
    say(f"CBOE equity P/C {s.index.min().date()}..{s.index.max().date()} ({len(s)} days)")
    return s


def cboe_regime(pc: pd.Series) -> pd.Series:
    """z of the 5-day mean vs the trailing 3 years, usable from the next day (daily calendar)."""
    m5 = pc.rolling(5, min_periods=4).mean()
    z = (m5 - m5.rolling(756, min_periods=252).mean()) / m5.rolling(756, min_periods=252).std()
    z.index = z.index + pd.Timedelta(1, "D")
    return z.asfreq("D").ffill(limit=5)


# ================================================================================================ forward ledgers


def _done_today(path: Path, today: str) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for ln in path.read_text().splitlines()[-20000:]:
        if f'"date": "{today}"' in ln:
            try:
                out.add(json.loads(ln)["ticker"])
            except (ValueError, KeyError):
                continue
    return out


def options_ledger(tickers: list[str], say=_say) -> dict:
    """Today's option-implied signals per ticker (yfinance chain of the first expiry >= 20 days):
    IV skew = IV(put, strike ~ 0.9 S) - IV(call, strike ~ 1.1 S); ATM IV; put/call volume and open-interest ratios."""
    import yfinance as yf

    today = date.today()
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = datetime.now(ZoneInfo("America/New_York")).date()
    try:
        last = yf.Ticker("SPY").history(period="7d").index[-1].date()
    except Exception:  # noqa: BLE001
        last = None
    if last is None or (ny - last).days > 1:  # weekend / holiday: Yahoo serves placeholder IVs
        say(f"옵션 기록 건너뜀: 직전 미국 거래일 {last} (뉴욕 {ny})")
        return {"skipped": "미국 장이 쉬는 날"}
    out = kd.ledger("us_options.jsonl")
    done = _done_today(out, str(today))
    n = 0
    with open(out, "a") as fh:
        for t in tickers:
            if t in done:
                continue
            try:
                tk = yf.Ticker(t.replace(".", "-"))
                exps = [e for e in tk.options if (pd.Timestamp(e).date() - today).days >= 20]
                if not exps:
                    continue
                ch = tk.option_chain(exps[0])
                spot = float(tk.history(period="5d")["Close"].iloc[-1])
            except Exception:  # noqa: BLE001 - skip names Yahoo cannot serve today
                time.sleep(1.0)
                continue
            c, p = ch.calls, ch.puts

            def iv_near(df, k):
                df = df[(df["impliedVolatility"] > 0.05) & (df["bid"].gt(0) | df["lastPrice"].gt(0))]
                if df.empty:
                    return None
                return float(df.iloc[(df["strike"] - k).abs().argsort()[:1]]["impliedVolatility"].iloc[0])

            rec = {
                "date": str(today),
                "ticker": t,
                "expiry": exps[0],
                "spot": spot,
                "iv_put90": iv_near(p, 0.9 * spot),
                "iv_call110": iv_near(c, 1.1 * spot),
                "iv_atm": iv_near(c, spot),
                "pc_volume": float(p["volume"].fillna(0).sum() / max(c["volume"].fillna(0).sum(), 1)),
                "pc_oi": float(p["openInterest"].fillna(0).sum() / max(c["openInterest"].fillna(0).sum(), 1)),
            }
            fh.write(json.dumps(rec) + "\n")
            n += 1
            time.sleep(0.3)
    say(f"옵션 기록 {n}종목")
    return {"recorded": n}


def eps_trend_ledger(tickers: list[str], say=_say) -> dict:
    """Today's EPS-estimate trend (current / 7 / 30 / 60 / 90 days ago) and up / down revision counts per ticker."""
    import yfinance as yf

    today = str(date.today())
    out = kd.ledger("us_eps_trend.jsonl")
    done = _done_today(out, today)
    n = 0
    with open(out, "a") as fh:
        for t in tickers:
            if t in done:
                continue
            try:
                tk = yf.Ticker(t.replace(".", "-"))
                tr, rv = tk.eps_trend, tk.eps_revisions
            except Exception:  # noqa: BLE001
                time.sleep(1.0)
                continue
            rec = {"date": today, "ticker": t}
            if tr is not None and len(tr):
                rec["trend"] = json.loads(tr.to_json(orient="index"))
            if rv is not None and len(rv):
                rec["revisions"] = json.loads(rv.to_json(orient="index"))
            fh.write(json.dumps(rec) + "\n")
            n += 1
            time.sleep(0.3)
    say(f"EPS 추정치 기록 {n}종목")
    return {"recorded": n}
