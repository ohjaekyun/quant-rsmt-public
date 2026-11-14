"""kr-market data layer (verified + corrected 2026-09-24 KST, keyless).

Corrections vs the probe's kr_market.py:
  * Naver fchart caps at 3000 rows (~12y, from 2014-07-04), not ~24y -> warn if `start` is truncated.
  * pd.Timedelta(days=N) raises a numpy DeprecationWarning (numpy 2.5 / pandas 2.3) -> pd.Timedelta(N, "D").
  * official_close_fix: lookback grows with the NXT-integrated window (fixed 30 would silently stop
    fixing 2026-09-14.. once >30 bars old); relative tolerance so adjusted-price rounding noise
    (<=2 KRW, <=1.6bp) is not "fixed"; last day from Daum quote `regularTradePrice` (official
    regular-session close), yfinance only as guarded fallback (must lie inside the bar's High/Low).
  * naver_snapshot / naver_financials / wisereport_* raise on invalid codes instead of returning
    all-None / TypeError / AttributeError; snapshot adds adjusted 52w high/low.
  * band_valuation_percentile: PER percentile over months with a defined PER (NaN months excluded).
  * kind_listing: de-duplicates 43 codes that KIND lists twice (old/new region name).
"""
from __future__ import annotations

import re
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")
H = {"User-Agent": UA}
TIMEOUT = 20
NXT_SINCE = pd.Timestamp("2026-09-01")  # Naver/Daum closes are KRX+NXT integrated from ~2026-09-14
_CODE_RE = re.compile(r"[0-9A-Z]{6}")   # KRX short codes can be alphanumeric (e.g. 0126Z0)


def _check_code(code: str) -> str:
    if not _CODE_RE.fullmatch(code or ""):
        raise ValueError(f"not a KRX 6-char code: {code!r} (US tickers go to the US data layer)")
    return code


def _num(s):
    if s is None:
        return None
    s = str(s).replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _krw_korean(s):
    if not s:
        return None
    s = s.replace(",", "").replace(" ", "")
    jo = re.search(r"(\d+(?:\.\d+)?)조", s)
    eok = re.search(r"(\d+(?:\.\d+)?)억", s)
    v = (float(jo.group(1)) * 1e12 if jo else 0) + (float(eok.group(1)) * 1e8 if eok else 0)
    return v or None


# ------------------------------------------------------------------ prices
def naver_ohlcv(symbol: str, start: str = "2019-01-01", count: int = 3000) -> pd.DataFrame:
    """Adjusted (split/bonus/rights, NOT dividend) daily OHLCV from Naver fchart. tz-naive index.
    symbol: 6-char code or 'KOSPI' | 'KOSDAQ' | 'KPI200'. Server caps at 3000 bars (~12y).
    Index Volume is in 1,000 shares. Recent stock Close/High/Low = KRX+NXT integrated."""
    r = requests.get("https://fchart.stock.naver.com/sise.nhn",
                     params=dict(timeframe="day", count=count, requestType=0, symbol=symbol),
                     headers=H, timeout=TIMEOUT)
    r.raise_for_status()
    rows = re.findall(r'<item data="(.*?)" />', r.text)
    if not rows:
        raise ValueError(f"naver fchart: no data for {symbol!r}")
    df = pd.read_csv(StringIO("\n".join(rows)), sep="|", header=None, dtype={0: str},
                     names=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index().astype(float)
    df = df[~df.index.duplicated(keep="last")]
    z = df["Open"] <= 0   # trading-halt rows: O/H/L=0, Vol=0 -> flat bar at Close
    df.loc[z, ["Open", "High", "Low"]] = np.repeat(df.loc[z, ["Close"]].values, 3, axis=1)
    if len(df) >= count and df.index[0] > pd.Timestamp(start):
        warnings.warn(f"naver fchart {symbol}: capped at {len(df)} bars, history starts "
                      f"{df.index[0].date()} (> requested {start})")
    return df.loc[pd.Timestamp(start):]


def kr_indices(start: str = "2019-01-01") -> pd.DataFrame:
    out = {}
    for name, ysym in [("KOSPI", "^KS11"), ("KOSDAQ", "^KQ11"), ("KPI200", "^KS200")]:
        try:
            out[name] = naver_ohlcv(name, start)["Close"]
        except Exception as e:  # noqa: BLE001
            import yfinance as yf
            print(f"naver index {name} failed ({e}); yfinance {ysym} (has missing days)")
            h = yf.Ticker(ysym).history(start=start)
            h.index = h.index.tz_localize(None).normalize()
            out[name] = h["Close"]
    return pd.DataFrame(out)


def _daum_headers(code: str) -> dict:
    return {**H, "Referer": f"https://finance.daum.net/quotes/A{code}"}


def daum_official_close(code: str, limit: int = 30) -> pd.Series:
    """Official KRX close for day t = Daum base price of t+1 (tradePrice - changePrice).
    The last available day is NOT included. limit <= 2000 (back to 2018-07)."""
    r = requests.get(f"https://finance.daum.net/api/charts/A{code}/days",
                     params={"limit": min(limit, 1999) + 1, "adjusted": "true"},
                     headers=_daum_headers(code), timeout=TIMEOUT)
    r.raise_for_status()
    d = pd.DataFrame(r.json()["data"])
    d["Date"] = pd.to_datetime(d["date"])
    d = d.set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="last")]
    return (d["tradePrice"] - d["changePrice"]).shift(-1).dropna().rename("OfficialClose")


def daum_quote(code: str) -> dict:
    """Daum quote snapshot. regularTradePrice = KRX regular-session (official) close of tradeDate;
    tradePrice = integrated KRX+NXT last. Referer is mandatory (403 without)."""
    r = requests.get(f"https://finance.daum.net/api/quotes/A{code}",
                     params={"summary": "false", "changeStatistics": "true"},
                     headers=_daum_headers(code), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def yf_ohlcv(code: str, start: str = "2019-01-01", market: str | None = None) -> pd.DataFrame:
    import yfinance as yf
    market = market or kr_market_of(code)["market"]
    sym = f"{code}.{'KQ' if market == 'KOSDAQ' else 'KS'}"
    h = yf.Ticker(sym).history(start=start, auto_adjust=False)
    if h.empty:
        raise ValueError(f"yfinance: empty history for {sym}")
    h.index = h.index.tz_localize(None).normalize()
    h = h[~h.index.duplicated(keep="last")]
    return h[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def official_close_fix(df: pd.DataFrame, code: str, since: pd.Timestamp = NXT_SINCE) -> pd.DataFrame:
    """Replace Naver's integrated (KRX+NXT) closes with KRX official closes. Adds CloseSrc.
    NOTE: High/Low/Open of recent bars stay integrated (they can exceed the KRX range)."""
    out = df.copy()
    out["CloseSrc"] = "naver"
    lookback = min(2000, max(30, int((out.index >= since).sum()) + 5))
    try:
        off = daum_official_close(code, lookback)
        idx = out.index.intersection(off.index)
        gap = (out.loc[idx, "Close"] - off.loc[idx]).abs()
        tol = np.maximum(1.0, 3e-4 * off.loc[idx])   # < 1 KRX tick; > adjusted-price rounding noise
        diff = idx[(gap >= tol).to_numpy()]
        out.loc[diff, "Close"] = off.loc[diff].round()
        out.loc[diff, "CloseSrc"] = "krx_official(daum)"
    except Exception as e:  # noqa: BLE001
        print(f"official_close_fix: daum chart failed: {e}")
    last = out.index[-1]
    done = False
    try:
        q = daum_quote(code)
        px = q.get("regularTradePrice")
        if px and pd.Timestamp(q["date"]) == last:
            if float(px) != out.loc[last, "Close"]:
                out.loc[last, "Close"] = float(px)
                out.loc[last, "CloseSrc"] = "krx_official(daum_quote)"
            done = True
    except Exception as e:  # noqa: BLE001
        print(f"official_close_fix: daum quote failed: {e}")
    if not done:
        try:
            y = yf_ohlcv(code, start=(last - pd.Timedelta(10, "D")).strftime("%Y-%m-%d"))
            yc = float(y.loc[last, "Close"]) if last in y.index else None
            # guard against Yahoo's stale closes: official close must lie inside the integrated range
            if yc and out.loc[last, "Low"] <= yc <= out.loc[last, "High"] and yc != out.loc[last, "Close"]:
                out.loc[last, "Close"] = yc
                out.loc[last, "CloseSrc"] = "krx_official(yahoo)"
        except Exception as e:  # noqa: BLE001
            print(f"official_close_fix: yahoo failed: {e}")
    out["High"] = out[["High", "Close"]].max(axis=1)
    out["Low"] = out[["Low", "Close"]].min(axis=1)
    return out


def kr_prices(code: str, start: str = "2019-01-01") -> pd.DataFrame:
    _check_code(code)
    try:
        df = naver_ohlcv(code, start)
        return official_close_fix(df, code)
    except Exception as e:  # noqa: BLE001
        print(f"naver failed ({e}); falling back to yfinance")
        y = yf_ohlcv(code, start)
        ratio = y["Adj Close"] / y["Close"]   # dividend-adjusted -> never splice with Naver
        return pd.DataFrame({c: y[c] * ratio for c in ["Open", "High", "Low", "Close"]} | {
            "Volume": y["Volume"], "CloseSrc": "yahoo_adj"})


# ------------------------------------------------------------------ identity / listing
def kr_market_of(code: str) -> dict:
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/basic", headers=H, timeout=TIMEOUT)
    if r.status_code != 200:  # invalid code / US ticker -> 409 {"code":"StockConflict"}
        raise ValueError(f"naver basic {code}: HTTP {r.status_code} {r.text[:80]}")
    b = r.json()
    ex = b["stockExchangeType"]["code"]            # 'KS' | 'KQ' | 'KN'(konex, untested)
    market = {"KS": "KOSPI", "KQ": "KOSDAQ", "KN": "KONEX"}.get(ex, b["stockExchangeType"]["name"])
    return {"code": code, "name": b["stockName"], "market": market,
            "yahoo": f"{code}.{'KQ' if ex == 'KQ' else 'KS'}", "sosok": b.get("sosok")}


def kind_listing() -> pd.DataFrame:
    frames = []
    for mkt, label in [("stockMkt", "KOSPI"), ("kosdaqMkt", "KOSDAQ"), ("konexMkt", "KONEX")]:
        r = requests.get("https://kind.krx.co.kr/corpgeneral/corpList.do",
                         params={"method": "download", "searchType": "13", "marketType": mkt},
                         headers=H, timeout=TIMEOUT)
        r.raise_for_status()
        t = pd.read_html(StringIO(r.content.decode("euc-kr", errors="replace")),
                         converters={"종목코드": str})[0]
        t["Market"] = label
        frames.append(t)
    df = pd.concat(frames, ignore_index=True).rename(columns={
        "회사명": "Name", "종목코드": "Code", "업종": "Industry", "주요제품": "Products",
        "상장일": "ListingDate", "결산월": "SettleMonth", "대표자명": "CEO",
        "홈페이지": "HomePage", "지역": "Region"}).drop(columns=["시장구분"])
    df["Code"] = df["Code"].str.zfill(6)
    # KIND lists ~43 companies twice (old + new region name, e.g. 전라남도 / 전남광주통합특별시)
    return df.drop_duplicates("Code", keep="last").reset_index(drop=True)


# ------------------------------------------------------------------ fundamentals
def naver_snapshot(code: str) -> dict:
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/integration", headers=H, timeout=TIMEOUT)
    if r.status_code != 200:
        raise ValueError(f"naver integration {code}: HTTP {r.status_code} {r.text[:80]}")
    j = r.json()
    ti = {x["code"]: x for x in j.get("totalInfos", [])}
    g = lambda k: _num(ti.get(k, {}).get("value"))  # noqa: E731
    cons = j.get("consensusInfo") or {}
    ind_code = j.get("industryCode")
    ind_name = None
    if ind_code:
        try:
            ind_name = requests.get(f"https://m.stock.naver.com/api/stocks/industry/{ind_code}",
                                    params={"page": 1, "pageSize": 1}, headers=H,
                                    timeout=TIMEOUT).json()["groupInfo"]["name"]
        except Exception:  # noqa: BLE001
            pass
    return {
        "code": code, "name": j.get("stockName"),
        "prev_close": g("lastClosePrice"),     # official close of the day BEFORE the latest bar
        "per": g("per"), "eps": g("eps"), "pbr": g("pbr"), "bps": g("bps"),   # per/pbr use integrated price
        "per_basis": ti.get("per", {}).get("valueDesc"),
        "fwd_per": g("cnsPer"), "fwd_eps": g("cnsEps"),
        "div_yield_pct": g("dividendYieldRatio"), "dps": g("dividend"),
        "foreign_rate_pct": g("foreignRate"),
        "market_cap_krw": _krw_korean(ti.get("marketValue", {}).get("value")),
        "high_52w": g("highPriceOf52Weeks"), "low_52w": g("lowPriceOf52Weeks"),   # raw, KRX+NXT integrated
        # *Adjusted keys only exist after a capital event -> fall back to raw
        "high_52w_adj": g("highPriceOf52WeeksAdjusted") or g("highPriceOf52Weeks"),
        "low_52w_adj": g("lowPriceOf52WeeksAdjusted") or g("lowPriceOf52Weeks"),
        "target_mean": _num(cons.get("priceTargetMean")), "recomm_mean": _num(cons.get("recommMean")),
        "consensus_date": cons.get("createDate"),
        "industry_code": ind_code, "industry_name": ind_name,
        "peers": [p["itemCode"] for p in (j.get("industryCompareInfo") or [])],
        "researches": [{"date": x["wdt"], "broker": x["bnm"], "title": x["tit"], "id": x["id"]}
                       for x in (j.get("researches") or [])],
        "investor_flow": [{"date": x["bizdate"], "foreign_net": _num(x["foreignerPureBuyQuant"]),
                           "inst_net": _num(x["organPureBuyQuant"]),
                           "indiv_net": _num(x["individualPureBuyQuant"]),
                           "foreign_hold_pct": _num(x["foreignerHoldRatio"])}
                          for x in (j.get("dealTrendInfos") or [])],
    }


def naver_financials(code: str, period: str = "annual") -> pd.DataFrame:
    r = requests.get(f"https://m.stock.naver.com/api/stock/{code}/finance/{period}", headers=H, timeout=TIMEOUT)
    fi = r.json().get("financeInfo") if r.status_code == 200 else None
    if not fi:
        raise ValueError(f"naver finance/{period} {code}: HTTP {r.status_code} {r.text[:80]}")
    cols = {t["key"]: t["key"] + ("(E)" if t["isConsensus"] == "Y" else "") for t in fi["trTitleList"]}
    data = {row["title"]: {cols[k]: _num(v["value"]) for k, v in row["columns"].items()}
            for row in fi["rowList"]}
    return pd.DataFrame(data).T[list(cols.values())]


def _wise_session(code: str):
    s = requests.Session()
    s.headers.update(H)
    page = s.get("https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx",
                 params={"cmp_cd": code}, timeout=TIMEOUT).text
    return s, page


def wisereport_snapshot(code: str) -> dict:
    from bs4 import BeautifulSoup
    s, page = _wise_session(code)
    soup = BeautifulSoup(page, "lxml")
    txt = lambda el: re.sub(r"\s+", " ", el.get_text(" ", strip=True))  # noqa: E731
    head = [txt(x) for x in soup.select("td.cmp-table-cell dt")]
    if not head:
        raise ValueError(f"wisereport {code}: no company header (invalid code?)")
    hd = {}
    for h in head:
        for k in ["PER", "업종PER", "PBR", "EPS", "BPS", "현금배당수익률", "WICS"]:
            if h.startswith(k + " ") or h.startswith(k + " :"):
                hd[k] = h.split(":", 1)[1].strip() if k == "WICS" else h[len(k):].strip()
    krx_sector = next((h.split(":", 1)[1].strip() for h in head
                       if h.startswith(("KOSPI :", "KOSDAQ :", "KONEX :"))), None)
    q = {}
    tb = soup.find("table", id="cTB11")
    for tr in tb.find_all("tr") if tb else []:
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            q[txt(th)] = txt(td)
    cons = {}
    ct = soup.find("table", id="cTB15")
    if ct:
        vals = [txt(td) for td in ct.find_all("tr")[-1].find_all("td")]
        if len(vals) >= 5:
            cons = dict(zip(["opinion", "target", "eps", "per", "n_brokers"], map(_num, vals[:5])))
    price_part = q.get("주가/전일대비/수익률", "").split("/")
    rets = [_num(x) for x in q.get("수익률 (1M/3M/6M/1Y)", "").split("/")]
    hl = q.get("52Weeks 최고/최저", "/").split("/")
    return {
        "code": code,
        "official_close": _num(price_part[0]) if price_part else None,
        "per": _num(hd.get("PER")), "sector_per": _num(hd.get("업종PER")),
        "pbr": _num(hd.get("PBR")), "eps": _num(hd.get("EPS")), "bps": _num(hd.get("BPS")),
        "div_yield_pct": _num(hd.get("현금배당수익률")),
        "wics": hd.get("WICS"), "krx_sector": krx_sector,
        "beta_52w": _num(q.get("52주베타")),
        "high_52w_adj": _num(hl[0]), "low_52w_adj": _num(hl[-1]),
        "market_cap_krw": (_num(q.get("시가총액")) or 0) * 1e8 or None,
        "float_pct": _num(q.get("발행주식수/유동비율", "/").split("/")[-1]),
        "foreign_pct": _num(q.get("외국인지분율")),
        "ret_1m_3m_6m_1y_pct": rets if len(rets) == 4 else None,
        "consensus": cons,
    }


def wisereport_annual(code: str, quarterly: bool = False) -> pd.DataFrame:
    s2 = requests.Session()
    s2.headers.update(H)
    p4 = s2.get("https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx",
                params={"cmp_cd": code}, timeout=TIMEOUT).text
    m = re.search(r"encparam:\s*'([^']+)'", p4)
    if not m:
        raise ValueError(f"wisereport c1040001 {code}: encparam not found (invalid code or page change)")
    j = s2.get("https://navercomp.wisereport.co.kr/v2/company/cF4002.aspx",
               params=dict(cmp_cd=code, frq="1" if quarterly else "0", rpt=5, finGubun="MAIN",
                           frqTyp=1 if quarterly else 0, cn="", encparam=m.group(1)),
               headers={"Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1040001.aspx?cmp_cd={code}"},
               timeout=TIMEOUT).json()
    periods = [re.sub(r"<br\s*/?>.*", "", y) for y in j["YYMM"]]
    keep = [i for i, p in enumerate(periods) if re.match(r"\d{4}/\d{2}", p)]
    rows = {}
    for row in j["DATA"]:
        nm = row["ACC_NM"]
        if nm.startswith("보통주.수정주가"):
            nm = "수정주가"
        if nm in ("EPS", "BPS", "PER", "PBR", "PCR", "PSR", "EV/EBITDA", "DPS", "현금배당수익률", "수정주가") and nm not in rows:
            rows[nm] = [row.get(f"DATA{i + 1}") for i in keep]
    df = pd.DataFrame(rows, index=[periods[i] for i in keep])
    df["is_est"] = df.index.str.contains(r"\(E\)")
    df.index = pd.to_datetime(df.index.str[:7], format="%Y/%m") + pd.offsets.MonthEnd(0)
    return df.astype({c: float for c in df.columns if c != "is_est"})


def valuation_history(code: str, prices: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time TTM PER/PBR (FY data usable FY-end+90d, quarters +45d). ~4.5y deep.
    Between the last FY knot and the first 4-quarter TTM knot the EPS is the (stale) FY value."""
    a = wisereport_annual(code)
    a = a[~a["is_est"]]
    knots = pd.DataFrame({"EPS_ttm": a["EPS"], "BPS": a["BPS"]})
    knots.index = knots.index + pd.Timedelta(90, "D")
    q = wisereport_annual(code, quarterly=True)
    q = q[~q["is_est"]].sort_index()
    ttm = q["EPS"].rolling(4).sum()
    qk = pd.DataFrame({"EPS_ttm": ttm, "BPS": q["BPS"]}).dropna()
    qk = qk[qk.index.month != 12]
    qk.index = qk.index + pd.Timedelta(45, "D")
    knots = pd.concat([knots, qk]).sort_index()
    knots = knots[~knots.index.duplicated(keep="last")]
    v = pd.merge_asof(prices[["Close"]].sort_index(), knots, left_index=True, right_index=True)
    v = v.dropna(subset=["EPS_ttm"])
    v["PER"] = v["Close"] / v["EPS_ttm"].where(v["EPS_ttm"] > 0)
    v["PBR"] = v["Close"] / v["BPS"]
    v["PER_pctile"] = v["PER"].rank(pct=True)
    v["PBR_pctile"] = v["PBR"].rank(pct=True)
    return v


def wisereport_band(code: str) -> pd.DataFrame:
    s, _ = _wise_session(code)
    j = s.get("https://navercomp.wisereport.co.kr/common/BandChart3.aspx",
              params={"cmp_cd": code, "gubun": "0"},
              headers={"Referer": f"https://navercomp.wisereport.co.kr/v2/company/c1010001.aspx?cmp_cd={code}"},
              timeout=TIMEOUT).json()
    out = {}
    for key, lab in [("bandChart1", "EPS"), ("bandChart2", "BPS")]:
        b = j[key]
        # use the first band line with a non-zero multiple (VAL4 is often '0.0배')
        vk, mult = next(((f"val{i}", _num(b["name"][f"VAL{i}"])) for i in range(1, 5)
                         if _num(b["name"].get(f"VAL{i}"))), (None, None))
        ts = pd.to_datetime([p["x"] for p in b["price"]], unit="ms", utc=True).tz_convert("Asia/Seoul").tz_localize(None).normalize()
        out["Price"] = pd.Series([p["y"] for p in b["price"]], index=ts)
        out[lab] = pd.Series([p["y"] / mult if (mult and p["y"] is not None) else None for p in b[vk]] if vk
                             else [None] * len(ts), index=ts)
    df = pd.DataFrame(out).astype(float)
    df["PER"] = df["Price"] / df["EPS"].where(df["EPS"] > 0)
    df["PBR"] = df["Price"] / df["BPS"].where(df["BPS"] > 0)
    return df


def band_valuation_percentile(code: str) -> dict:
    b = wisereport_band(code).dropna(subset=["Price"])
    cur = b.iloc[-1]
    f = lambda x: None if pd.isna(x) else round(float(x), 4)  # noqa: E731
    per_ok, pbr_ok = b["PER"].dropna(), b["PBR"].dropna()
    return {"asof_month": b.index[-1].date().isoformat(), "months": len(b), "per_months": len(per_ok),
            "per": f(cur["PER"]),
            "per_pctile": f((per_ok <= cur["PER"]).mean()) if pd.notna(cur["PER"]) and len(per_ok) else None,
            "pbr": f(cur["PBR"]),
            "pbr_pctile": f((pbr_ok <= cur["PBR"]).mean()) if pd.notna(cur["PBR"]) and len(pbr_ok) else None,
            "per_min": f(per_ok.min()), "per_med": f(per_ok.median()), "per_max": f(per_ok.max()),
            "pbr_min": f(pbr_ok.min()), "pbr_med": f(pbr_ok.median()), "pbr_max": f(pbr_ok.max())}
