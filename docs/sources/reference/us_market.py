"""US market data layer for quantcio -- re-verified 2026-09-24 (independent verifier).

Primary: yfinance 1.7 (curl_cffi + crumb).  Fallbacks: raw Yahoo v8 chart (requests, short UA),
Nasdaq API (independent provider, split-adjusted only), Google News RSS, Yahoo RSS (curl_cffi), Naver FX API.

Changes vs. the probe version (all verified live):
  * unfinished (live, intraday) daily bar is dropped by default (drop_live=True) in every OHLCV/benchmark
    function -- during the regular session Yahoo/Nasdaq return a half-formed bar for today;
  * pd.Timedelta(days=N) -> pd.Timedelta(N, unit="D"): with numpy 2.5.3 + pandas 2.3.3 the keyword form emits
    "DeprecationWarning: The 'generic' unit for NumPy timedelta is deprecated, and will raise an error";
  * news / earnings: an empty answer is legitimate -> no 16 s retry loop (allow_empty=True), Yahoo RSS fallback
    also triggers when yfinance news is merely empty; get_earnings_dates(limit) is now honoured;
  * Nasdaq history is split-adjusted (NVDA 2021 4:1 and 2024 10:1 splits are applied) but NOT dividend-adjusted;
  * Google News query uses the cleaned company name ("Apple" instead of the exact phrase "Apple Inc.").
"""
from __future__ import annotations

import random
import re
import time
from email.utils import parsedate_to_datetime

import feedparser
import pandas as pd
import requests
import yfinance as yf
from curl_cffi import requests as creq

# ---- yfinance global config (set once at import) ------------------------------------------
yf.config.network.retries = 2          # yf.set_config(retries=) is deprecated in 1.7
yf.config.debug.hide_exceptions = False  # default True => errors silently become empty frames

UA_BROWSER = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")
# Yahoo returns 429 to plain requests/curl with a *full* Chrome UA (TLS-fingerprint mismatch),
# but 200 to this short UA (the one FinanceDataReader uses). Fragile; keep as fallback only.
UA_YAHOO_SHORT = "Mozilla/5.0 AppleWebKit/537.36"


# ---- retry / backoff ------------------------------------------------------------------------
def with_backoff(fn, *, tries: int = 4, base: float = 2.0, allow_empty: bool = False):
    """Call fn(); retry on rate-limit / transient errors (and empty results unless allow_empty)
    with exp backoff+jitter. 404 / 'not found' / delisted errors are NOT retried."""
    last = None
    for i in range(tries):
        try:
            out = fn()
            if not allow_empty and (out is None or (hasattr(out, "empty") and out.empty)
                                    or (isinstance(out, (list, dict)) and len(out) == 0)):
                raise ValueError("empty result")
            return out
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "404" in msg or "Not Found" in msg or "delisted" in msg:
                raise
            last = e
            if i < tries - 1:
                is_rl = isinstance(e, yf.exceptions.YFRateLimitError) or "429" in msg or "Too Many" in msg
                time.sleep(base * (2 ** i) * (3 if is_rl else 1) + random.uniform(0, 1))
    raise last


# ---- unfinished-bar guard -----------------------------------------------------------------
_DEFAULT_CLOSE = {"America/New_York": (16, 0), "America/Chicago": (15, 15),
                  "Asia/Seoul": (15, 30), "Asia/Tokyo": (15, 30)}


def _drop_unfinished_bar(df: pd.DataFrame, tz: str, regular_end: pd.Timestamp | None = None,
                         now: pd.Timestamp | None = None, buffer_min: int = 20) -> pd.DataFrame:
    """df has a tz-naive, exchange-local normalized date index. If its last row is *today* (exchange-local)
    and the regular session has not closed yet (+buffer), that bar is still forming -> drop it."""
    if df.empty:
        return df
    now = (now if now is not None else pd.Timestamp.now(tz="UTC")).tz_convert(tz)
    today = now.tz_localize(None).normalize()
    if df.index[-1] != today:
        return df
    if regular_end is None or regular_end.tz_convert(tz).tz_localize(None).normalize() != today:
        h, m = _DEFAULT_CLOSE.get(tz, (16, 0))
        regular_end = (today + pd.Timedelta(h * 60 + m, unit="m")).tz_localize(tz)
    if now < regular_end + pd.Timedelta(buffer_min, unit="m"):
        return df.iloc[:-1]
    return df


# ---- OHLCV ------------------------------------------------------------------------------------
_OHLCV = ["open", "high", "low", "close", "volume"]


def us_ohlcv_yf(ticker: str, period: str = "7y", drop_live: bool = True) -> pd.DataFrame:
    """Adjusted (split+dividend) daily OHLCV. Index: tz-naive exchange-local date."""
    tk = yf.Ticker(ticker)
    df = with_backoff(lambda: tk.history(period=period, auto_adjust=True, actions=False))
    tz = str(df.index.tz)
    df.index = df.index.tz_localize(None).normalize()
    df.index.name = "date"
    df = df.rename(columns=str.lower)[_OHLCV]
    df = df[~df.index.duplicated(keep="last")]
    if drop_live:
        try:
            reg_end = tk.history_metadata["currentTradingPeriod"]["regular"]["end"]
            reg_end = pd.Timestamp(reg_end) if reg_end is not None else None
        except Exception:  # noqa: BLE001
            reg_end = None
        df = _drop_unfinished_bar(df, tz, reg_end)
    df.attrs.update(source="yfinance", adjusted=True, ticker=ticker, tz=tz)
    return df


def us_ohlcv_yahoo_chart(ticker: str, years: int = 7, drop_live: bool = True) -> pd.DataFrame:
    """Fallback without yfinance: Yahoo v8 chart JSON via plain requests; OHLC adjusted by
    adjclose/close ratio. Dates computed in the exchange timezone (correct for FX too)."""
    end = int(time.time())
    start = end - int(years * 365.25 * 86400)
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = dict(period1=start, period2=end, interval="1d", includeAdjustedClose="true", events="div,splits")

    def _get():
        r = requests.get(url, params=params, headers={"User-Agent": UA_YAHOO_SHORT}, timeout=20)
        r.raise_for_status()
        return r.json()["chart"]["result"][0]

    res = with_backoff(_get)
    tz = res["meta"]["exchangeTimezoneName"]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).tz_localize(None).normalize()
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in _OHLCV}, index=idx, dtype="float64")
    adj = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype="float64")
    ratio = adj / df["close"]
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c] * ratio
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["close"])
    df.index.name = "date"
    if drop_live:
        reg = (res["meta"].get("currentTradingPeriod") or {}).get("regular") or {}
        reg_end = pd.Timestamp(reg["end"], unit="s", tz="UTC") if reg.get("end") else None
        df = _drop_unfinished_bar(df, tz, reg_end)
    df.attrs.update(source="yahoo_chart_raw", adjusted=True, ticker=ticker, tz=tz)
    return df


def us_ohlcv_nasdaq(ticker: str, years: int = 7, assetclass: str = "stocks",
                    drop_live: bool = True) -> pd.DataFrame:
    """Independent provider (api.nasdaq.com), keyless, ~10y max. SPLIT-adjusted, NOT dividend-adjusted
    (close == Yahoo raw 'Close'; ~4% below Yahoo adjusted close 7y back for AAPL).
    Volume is Nasdaq's consolidated figure (differs from Yahoo by up to ~0.3%).
    assetclass: 'stocks' | 'etf' | 'index' (index symbols: COMP, NDX; SPX/VIX NOT available)."""
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    params = dict(assetclass=assetclass, fromdate=(today - pd.DateOffset(years=years)).strftime("%Y-%m-%d"),
                  todate=today.strftime("%Y-%m-%d"), limit=9999)

    def _get():
        r = requests.get(f"https://api.nasdaq.com/api/quote/{ticker}/historical", params=params,
                         headers={"User-Agent": UA_BROWSER, "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        if not j.get("data"):
            raise ValueError(f"nasdaq: {j.get('status')}")
        return j["data"]["tradesTable"]["rows"]

    rows = with_backoff(_get)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
    for c in _OHLCV:
        # '$337.02', '31,658,820', '--' (index volume) -> float/NaN
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    df = df.set_index("date").sort_index()[_OHLCV]
    df = df[~df.index.duplicated(keep="last")]
    if drop_live:
        df = _drop_unfinished_bar(df, "America/New_York")
    df.attrs.update(source="nasdaq", adjusted=False, split_adjusted=True, ticker=ticker, tz="America/New_York")
    return df


def us_ohlcv(ticker: str, years: int = 7) -> pd.DataFrame:
    """Primary -> fallback chain. Check df.attrs['adjusted'] (False for the Nasdaq fallback)."""
    for fn in (lambda: us_ohlcv_yf(ticker, f"{years}y"), lambda: us_ohlcv_yahoo_chart(ticker, years),
               lambda: us_ohlcv_nasdaq(ticker, years)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            print(f"[us_ohlcv] {ticker} source failed: {type(e).__name__}: {e}")
    raise RuntimeError(f"all OHLCV sources failed for {ticker}")


# ---- benchmarks -----------------------------------------------------------------------------
def us_benchmarks(period: str = "7y", drop_live: bool = True) -> pd.DataFrame:
    """Close panel for SPY, QQQ, ^GSPC, ^IXIC, ^VIX aligned to the SPY (NYSE) calendar.
    Yahoo index series (^GSPC/^IXIC/^VIX) occasionally miss a day (2026-09-22 observed);
    ETFs don't. Use SPY/QQQ for returns/beta; VIX is a level -> ffill is safe."""
    syms = ["SPY", "QQQ", "^GSPC", "^IXIC", "^VIX"]
    df = with_backoff(lambda: yf.download(syms, period=period, auto_adjust=True, progress=False, threads=True))
    close = df["Close"].copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    close = close[close["SPY"].notna()]            # NYSE trading days only
    if drop_live:                                  # SPY itself has a half-formed bar during the session
        close = _drop_unfinished_bar(close, "America/New_York")
    close.attrs["missing_index_days"] = close[["^GSPC", "^IXIC", "^VIX"]].isna().sum().to_dict()
    close["^VIX"] = close["^VIX"].ffill()
    return close


# ---- fundamentals / valuation / analyst --------------------------------------------------
def us_fundamentals(ticker: str) -> dict:
    """Normalized snapshot. Ratios are fractions (0.25 = 25%). NOTE Yahoo units:
    info['dividendYield'] is in PERCENT (0.32 => 0.32%), info['debtToEquity'] in PERCENT (78 => 0.78x)."""
    tk = yf.Ticker(ticker)
    info = with_backoff(lambda: tk.info)
    g = info.get
    px = g("currentPrice") or g("regularMarketPrice")
    out = {
        "ticker": ticker, "name": g("shortName"), "currency": g("currency"), "exchange": g("exchange"),
        "sector": g("sector"), "industry": g("industry"), "price": px, "market_cap": g("marketCap"),
        "beta": g("beta"), "pe_trailing": g("trailingPE"), "pe_forward": g("forwardPE"),
        "pb": g("priceToBook"), "ev_ebitda": g("enterpriseToEbitda"), "ps_ttm": g("priceToSalesTrailing12Months"),
        "peg": g("trailingPegRatio") or g("pegRatio"),
        "roe": g("returnOnEquity"), "roa": g("returnOnAssets"), "gross_margin": g("grossMargins"),
        "op_margin": g("operatingMargins"), "net_margin": g("profitMargins"),
        "rev_growth_yoy": g("revenueGrowth"), "eps_growth_yoy": g("earningsGrowth"),
        "debt_to_equity": (g("debtToEquity") / 100) if g("debtToEquity") is not None else None,
        "div_yield": (g("dividendYield") / 100) if g("dividendYield") is not None else None,
        "eps_ttm": g("trailingEps"), "eps_fwd": g("forwardEps"),
        "high_52w": g("fiftyTwoWeekHigh"), "low_52w": g("fiftyTwoWeekLow"),
        "target_mean": g("targetMeanPrice"), "target_median": g("targetMedianPrice"),
        "target_high": g("targetHighPrice"), "target_low": g("targetLowPrice"),
        "rec_mean": g("recommendationMean"),  # 1=Strong Buy ... 5=Strong Sell
        "rec_key": g("recommendationKey"), "n_analysts": g("numberOfAnalystOpinions"),
        "next_earnings_utc": (pd.Timestamp(g("earningsTimestampStart"), unit="s", tz="UTC")
                              if g("earningsTimestampStart") else None),
        "earnings_date_is_estimate": g("isEarningsDateEstimate"),
    }
    out["upside_to_mean_target"] = (out["target_mean"] / px - 1) if (px and out["target_mean"]) else None
    try:  # rating distribution (current month) + revisions momentum
        rs = tk.recommendations_summary
        cur = rs.loc[rs["period"] == "0m"].iloc[0]
        out["ratings"] = {k: int(cur[k]) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]}
        et = tk.eps_trend
        out["eps_fwd1y_rev_30d"] = float(et.loc["+1y", "current"] / et.loc["+1y", "30daysAgo"] - 1)
    except Exception as e:  # noqa: BLE001
        out["ratings_error"] = f"{type(e).__name__}: {e}"
    return out


def us_earnings_history(ticker: str, limit: int = 12) -> pd.DataFrame:
    """EPS estimate/actual/surprise per report, newest first (future rows have NaN actual).
    Index tz=America/New_York. 'Surprise(%)' is in percent. Empty frame (fast) for ETFs / no data."""
    df = with_backoff(lambda: yf.Ticker(ticker).get_earnings_dates(limit=limit), allow_empty=True)
    if df is None or df.empty:
        return pd.DataFrame(columns=["EPS Estimate", "Reported EPS", "Surprise(%)"])
    return df.sort_index(ascending=False).head(limit)   # yfinance ignores limit (12 -> 25 rows)


# ---- news -----------------------------------------------------------------------------------
_NEWS_COLS = ["published_utc", "title", "publisher", "summary", "url", "source", "relevant"]


def _clean_company(company: str | None) -> str:
    if not company:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[,.]|\b(inc|corp|corporation|company|co|ltd|plc|holdings|class [ab])\b",
                                      "", company, flags=re.I)).strip()


def _names_for(ticker: str, company: str | None) -> list[str]:
    names = {ticker.lower()}
    base = _clean_company(company)
    if base:
        names.add(base.lower())
        names.add(base.split()[0].lower())   # NOTE: generic first words ("American ...") over-match
    return sorted(names)


def _relevant(text: str, names: list[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(rf"\b{re.escape(n)}\b", t) for n in names)


def _news_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=_NEWS_COLS)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True)
    return df.sort_values("published_utc", ascending=False, ignore_index=True)


def us_news_yf(ticker: str, company: str | None = None, count: int = 40) -> pd.DataFrame:
    """yfinance 1.x news: list of {'id','content':{...}}. Yahoo's ticker feed is loose (market-wide
    stories mixed in) -> 'relevant' flag = ticker/company name appears in title or summary."""
    items = with_backoff(lambda: yf.Ticker(ticker).get_news(count=count, tab="news"), allow_empty=True) or []
    names = _names_for(ticker, company)
    rows = []
    for it in items:
        c = it.get("content") or {}
        url = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        text = f"{c.get('title', '')} {c.get('summary', '')}"
        rows.append(dict(published_utc=pd.Timestamp(c.get("pubDate")), title=c.get("title"),
                         publisher=(c.get("provider") or {}).get("displayName"), summary=c.get("summary"),
                         url=url, source="yahoo", relevant=_relevant(text, names)))
    return _news_frame(rows)


def us_news_google(ticker: str, company: str | None = None, days: int = 7) -> pd.DataFrame:
    """Google News RSS (keyless). <=100 items, relevance-ordered -> we sort by time. No real summary;
    URL is a news.google.com redirect. Title has ' - Publisher' suffix (stripped)."""
    base = _clean_company(company)
    q = f'"{base}" OR {ticker}' if base else f"{ticker} stock"
    url = "https://news.google.com/rss/search"
    params = dict(q=f"{q} when:{days}d", hl="en-US", gl="US", ceid="US:en")

    def _get():
        r = requests.get(url, params=params, headers={"User-Agent": UA_BROWSER}, timeout=20)
        r.raise_for_status()
        return feedparser.parse(r.content).entries

    entries = with_backoff(_get, allow_empty=True)
    names = _names_for(ticker, company)
    rows = []
    for e in entries:
        pub = (e.get("source") or {}).get("title")
        title = e.get("title", "")
        if pub and title.endswith(f" - {pub}"):
            title = title[: -len(pub) - 3]
        rows.append(dict(published_utc=pd.Timestamp(parsedate_to_datetime(e["published"])),
                         title=title, publisher=pub, summary=None, url=e.get("link"), source="google_news",
                         relevant=_relevant(title, names)))
    return _news_frame(rows)


def us_news_yahoo_rss(ticker: str, company: str | None = None) -> pd.DataFrame:
    """Yahoo Finance RSS (~16-20 items, real summaries, direct URLs). Plain requests+Chrome UA -> 429;
    curl_cffi impersonate='chrome' -> 200. Same item pool as yfinance news (not independent)."""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"

    def _get():
        r = creq.get(url, impersonate="chrome", timeout=20)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:80]}")
        return feedparser.parse(r.content).entries

    entries = with_backoff(_get, allow_empty=True)
    names = _names_for(ticker, company)
    rows = [dict(published_utc=pd.Timestamp(parsedate_to_datetime(e["published"])),
                 title=e.get("title"), publisher=None, summary=e.get("summary"), url=e.get("link"),
                 source="yahoo_rss", relevant=_relevant(f"{e.get('title', '')} {e.get('summary', '')}", names))
            for e in entries]
    return _news_frame(rows)


def us_news(ticker: str, company: str | None = None, days: int = 7, only_relevant: bool = True) -> pd.DataFrame:
    """Merge yfinance + Google News (+ Yahoo RSS if yfinance failed or was empty); dedupe by normalized title."""
    frames = []
    yf_ok = False
    try:
        f = us_news_yf(ticker, company)
        frames.append(f)
        yf_ok = not f.empty
    except Exception as e:  # noqa: BLE001
        print(f"[us_news] {ticker} yfinance news failed: {type(e).__name__}: {e} -> Yahoo RSS")
    if not yf_ok:
        try:
            frames.append(us_news_yahoo_rss(ticker, company))
        except Exception as e2:  # noqa: BLE001
            print(f"[us_news] {ticker} yahoo_rss failed: {type(e2).__name__}: {e2}")
    try:
        frames.append(us_news_google(ticker, company, days))
    except Exception as e:  # noqa: BLE001
        print(f"[us_news] {ticker} google news failed: {type(e).__name__}: {e}")
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _news_frame([])
    df = pd.concat(frames, ignore_index=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days, unit="D")   # not Timedelta(days=..): numpy 2.5 warns
    df = df[df["published_utc"] >= cutoff]
    if only_relevant:
        df = df[df["relevant"].astype(bool)]
    df = df.copy()
    key = df["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.slice(0, 60)
    df["_k"] = key.where(key.str.len() > 0, df["url"].fillna(""))
    df = df.sort_values(["published_utc"], ascending=False).drop_duplicates("_k").drop(columns="_k")
    return df.reset_index(drop=True)


# ---- FX: USD/KRW --------------------------------------------------------------------------
def usdkrw_yf(period: str = "7y") -> pd.Series:
    """KRW per 1 USD, Yahoo 'KRW=X'. Index = Yahoo's own bar date (London tz, tz stripped).
    The last row is a LIVE intraday bar for today. Trades on US holidays (more rows than NYSE)."""
    h = with_backoff(lambda: yf.Ticker("KRW=X").history(period=period, auto_adjust=True, actions=False))
    s = h["Close"].copy()
    s.index = s.index.tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    s.name = "usdkrw"
    return s


def usdkrw_naver(pages: int = 1, page_size: int = 60) -> pd.Series:
    """Independent source: Naver (Hana Bank first-notice rate, KST date). Keyless JSON. pageSize<=60."""
    rows = []
    for p in range(1, pages + 1):
        r = requests.get("https://api.stock.naver.com/marketindex/exchange/FX_USDKRW/prices",
                         params=dict(page=p, pageSize=page_size), headers={"User-Agent": UA_BROWSER}, timeout=20)
        r.raise_for_status()
        j = r.json()
        if not j:
            break
        rows += j
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["closePrice"].astype(str).str.replace(",", "")).values,
                  index=pd.to_datetime(df["localTradedAt"]), name="usdkrw").sort_index()
    return s[~s.index.duplicated()]


def usdkrw_latest() -> tuple[float, str]:
    try:
        s = usdkrw_yf("1mo")
        return float(s.iloc[-1]), f"yahoo KRW=X {s.index[-1].date()}"
    except Exception:  # noqa: BLE001
        s = usdkrw_naver(1, 10)
        return float(s.iloc[-1]), f"naver/hana {s.index[-1].date()}"


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("error", DeprecationWarning)
    for t, comp in [("AAPL", "Apple Inc."), ("NVDA", "NVIDIA Corporation")]:
        d = us_ohlcv(t, 7)
        print(t, d.shape, d.attrs, d.index[-1].date(), "unique", d.index.is_unique)
        f = us_fundamentals(t)
        print(t, {k: f[k] for k in ("price", "pe_forward", "target_mean", "upside_to_mean_target", "rec_key")})
        e = us_earnings_history(t, 8)
        print(t, "earnings", e.shape, e.index[0])
        n = us_news(t, comp, days=7)
        print(t, "news", n.shape, n.source.value_counts().to_dict())
    b = us_benchmarks("7y")
    print("bench", b.shape, b.attrs, b.index[-1].date())
    print("usdkrw", usdkrw_latest())
    for t in ["005930.KS", "247540.KQ"]:   # KR via Yahoo suffixes (KOSPI .KS / KOSDAQ .KQ)
        k = us_ohlcv_yf(t, "7y")
        print(t, k.shape, k.attrs, k.index[-1].date(), "zero-volume rows", int((k.volume == 0).sum()))
