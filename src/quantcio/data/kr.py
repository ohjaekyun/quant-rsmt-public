"""Korean equities provider (KOSPI / KOSDAQ): prices, benchmark, fundamentals, valuation history,
news, filings, FX and the RSMT-Q training universe.

Ported from the live-verified references ``docs/sources/reference/kr_market.py`` and ``kr_text.py``
(verified 2026-09-24; gotchas in ``docs/sources/kr-*.json``). Source chain per method:

=================  ===========================================================================
resolve            Naver m.stock basic -> Daum quote -> KIND listing (name / KOSPI|KOSDAQ);
                   sector = Naver industry group (WICS-style) -> Daum WICS -> cached KIND KSIC
prices             Naver fchart (split/rights adjusted, NOT dividend adjusted, <= 3000 bars)
                   + KRX official-close fix (Daum base prices; last day Daum ``regularTradePrice``,
                   guarded yfinance) -> fallback yfinance .KS/.KQ (dividend adjusted, never spliced)
benchmark          Naver fchart KOSPI / KOSDAQ (index volume is in 1,000 shares) -> yfinance ^KS11/^KQ11
fundamentals       Naver m.stock integration + WiseReport (FnGuide) snapshot + Naver finance
                   annual/quarter (+ yfinance ``.info`` for the analyst target range only);
                   next_earnings_date from recent KRX 안내공시 ('결산실적공시 예고', earnings-call
                   '기업설명회(IR) 개최') - None when nothing is announced (never extrapolated)
valuation_history  WiseReport cF4002 annual/quarterly EPS/BPS knots -> point-in-time TTM PER/PBR
news               Naver m.stock stock news JSON + Google News RSS, de-duplicated, relevance-filtered
filings            OpenDART (``DART_API_KEY``) or keyless DART web search + viewer text,
                   + Naver broker-research summaries (the KR earnings-call substitute)
fx_rate            shared with the US provider (``quantcio.data.us.cached_usdkrw``): last COMPLETED
                   Yahoo KRW=X bar aligned to the last completed US session -> Naver FX_USDKRW
                   prices (flagged); one cache entry, so KR and US tickers use the same rate
universe           Naver m.stock market-cap ranking (KOSPI+KOSDAQ) -> FinanceDataReader KRX listing
=================  ===========================================================================

Definitions used for fundamentals (the KR sources disagree, see kr-market gotchas):
- ``per`` / ``pbr``: trailing (TTM EPS / latest BPS) on the KRX official close (WiseReport); Naver's
  TTM figure (NXT-integrated price) is the fallback. Never the fiscal-year (Daum) or band PER.
- ``forward_per`` / ``forward_eps``: consensus for the current fiscal year (Naver ``cnsPer``/``cnsEps``).
- ``sector_per``: WiseReport 업종PER (TTM basis). Daum ``sectorPer`` is FY-basis and is NOT used.
- ``roe``: TTM EPS / BPS; ``operating_margin``: sum of the last 4 reported quarters.
- ``revenue_growth_yoy`` / ``earnings_growth_yoy``: latest reported quarter vs the same quarter a year
  earlier (earnings = controlling net income); None when the base is <= 0.
- ``analyst_target_price``: consensus mean; ``analyst_rating``: consensus mean on the 1 (sell) .. 5
  (strong buy) scale, normalised, e.g. "Buy (4.00/5)".

Observability (never fail silently): every fallback or partial result is logged at WARNING, recorded
in :meth:`KRProvider.data_warnings` (Korean, report-ready) and, for frames, copied into
``df.attrs["warnings"]`` (always a list) next to ``df.attrs["source"]``. Mandatory data (resolve /
prices / benchmark) raise instead of returning empty frames. Price frames also carry
``attrs["split_adjusted"]`` / ``attrs["dividend_adjusted"]`` (Naver: True / False; Yahoo fallback:
True / True); ``attrs["adjusted"]`` is not set because the shared loader treats ``False`` as a
degraded fallback.
"""

from __future__ import annotations

import html as html_lib
import io
import logging
import math
import re
import threading
import time
import urllib.parse
import warnings
import zipfile
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from io import StringIO
from typing import Any, TypeVar
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning

from quantcio.config import Settings
from quantcio.data.base import MarketProvider, normalise_ticker
from quantcio.data.cache import DiskCache
from quantcio.data.http import PoliteSession
from quantcio.i18n import lang_of
from quantcio.i18n import tr as _tr
from quantcio.models import (
    PRICE_COLUMNS,
    FilingItem,
    Fundamentals,
    Instrument,
    Market,
    NewsItem,
    text_id,
)

# bs4 emits these for DART's XHTML fragments and very short titles; they are cosmetic.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

log = logging.getLogger(__name__)
T = TypeVar("T")

# ----------------------------------------------------------------------------------- constants

KST = ZoneInfo("Asia/Seoul")
#: Naver/Daum closes are the KRX+NXT integrated 20:00 last trade from ~2026-09-14 (checked from 09-01).
NXT_SINCE = pd.Timestamp("2026-09-01")
FCHART_MAX_BARS = 3000
#: Before this KST time a bar dated today is treated as partial (KRX closes 15:30; buffer for the
#: closing auction publication and Yahoo's ~20 min delay).
SESSION_FINAL = dtime(16, 0)
#: NXT after-market ends at 20:00 KST; until then Naver's "close" of today keeps moving.
NXT_CLOSE = dtime(20, 0)
#: Date-only timestamps (DART, research) are pinned to 12:00 KST so the calendar date is identical in
#: KST and UTC.
DATE_ONLY_HOUR_KST = 12
OFFICIAL_REL_TOL = 3e-4  # < 1 KRX tick, > adjusted-price rounding noise (<= 1.6bp observed)

NAVER_API = "https://m.stock.naver.com/api"
NAVER_FCHART = "https://fchart.stock.naver.com/sise.nhn"
DAUM_API = "https://finance.daum.net/api"
WISE = "https://navercomp.wisereport.co.kr"
DART_WEB = "https://dart.fss.or.kr"
OPENDART = "https://opendart.fss.or.kr/api"
GNEWS_RSS = "https://news.google.com/rss/search"
KIND_LIST = "https://kind.krx.co.kr/corpgeneral/corpList.do"

EXCHANGES = {"KS": "KOSPI", "KQ": "KOSDAQ", "KN": "KONEX"}
#: benchmark id -> (Naver fchart symbol, yfinance fallback symbol)
BENCHMARKS = {"KOSPI": ("KOSPI", "^KS11"), "KOSDAQ": ("KOSDAQ", "^KQ11")}

_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")  # KRX short codes can be alphanumeric (e.g. 0126Z0)

#: Extra politeness gaps (seconds) for hosts that quantcio.data.http does not throttle (and DART >= 0.5s).
_EXTRA_MIN_INTERVAL = {
    "fchart.stock.naver.com": 0.1,
    "api.stock.naver.com": 0.3,
    "n.news.naver.com": 0.3,
    "finance.daum.net": 0.2,
    "navercomp.wisereport.co.kr": 0.3,
    "kind.krx.co.kr": 0.5,
    "dart.fss.or.kr": 0.5,
}
_extra_last: dict[str, float] = {}
_extra_lock = threading.Lock()

# filings
_EARNINGS_RELEASE_RE = re.compile(r"영업\(잠정\)실적|잠정\)?실적|매출액또는손익구조|손익구조")
_PERIODIC_RE = re.compile(r"사업보고서|반기보고서|분기보고서")
_MATERIAL_RE = re.compile(
    r"^주요사항보고서|유상증자|무상증자|감자|합병|분할|전환사채|신주인수권|교환사채|횡령|배임|감사의견|"
    r"상장폐지|관리종목|거래정지|회생|파산|부도|영업정지|소송|단일판매|공급계약|자기주식|최대주주변경|불성실공시"
)
_CONTEXT_RE = re.compile(r"실적|전망|기업설명회|IR|장래사업|경영계획|해명|조회공시|배당")
_PERIODIC_SECTIONS = r"사업의 개요|경영진단|요약재무"
#: DART web ``publicType`` filters: periodic, 주요사항보고서, KRX 수시/공정/시장조치. Excludes the
#: insider-ownership flood (005930: 864 filings in 120d, >90% 임원ㆍ주요주주 reports).
DART_WEB_TYPES = ["A001", "A002", "A003", "B001", "I001", "I002", "I003"]
OPENDART_TYPES = ["A", "B", "I"]
_PAGEINFO_RE = re.compile(r"\[(\d+)/(\d+)\]\s*\[총\s*([\d,]+)건\]")
_NODE_RE = re.compile(r"(\w+)\['text'\] = \"([^\"]*)\";(.*?)\1\['dtd'\] = \"([^\"]+)\";", re.S)
_VIEWDOC_RE = re.compile(r'viewDoc\("(\d+)", "(\d+)", "(\d+)", "(\d+)", "(\d+)", "([^"]+)"')

# universe
_SPAC_RE = re.compile(r"스팩|SPAC", re.I)
_REIT_RE = re.compile(r"리츠$|^이리츠|REIT", re.I)  # NOT "메리츠금융지주" / "블리츠웨이"
_ETP_NAME_RE = re.compile(r"^(KODEX|TIGER|KBSTAR|RISE|ARIRANG|HANARO|KOSEF|SOL|ACE|PLUS|TIMEFOLIO)\b|ETN|ETF")
#: Listed infrastructure / real-estate funds (투융자회사) - not operating companies.
LISTED_FUND_CODES = {
    "088980": "맥쿼리인프라",
    "415640": "KB발해인프라",
    "094800": "맵스리얼티1",
}

#: Curated short names used by the Korean press (relevance filter). The company name is always included.
_ALIASES = {
    "005930": ["삼성전자", "삼전"],
    "000660": ["SK하이닉스", "하이닉스"],
    "373220": ["LG에너지솔루션", "LG엔솔"],
    "207940": ["삼성바이오로직스", "삼성바이오"],
    "005380": ["현대차", "현대자동차"],
    "000270": ["기아"],
    "035420": ["NAVER", "네이버"],
    "005490": ["POSCO홀딩스", "포스코홀딩스"],
    "003670": ["포스코퓨처엠"],
    "247540": ["에코프로비엠", "에코프로BM"],
    "055550": ["신한지주", "신한금융"],
    "105560": ["KB금융", "KB금융지주"],
    "012330": ["현대모비스"],
    "028260": ["삼성물산"],
    "066570": ["LG전자"],
    "051910": ["LG화학"],
    "006400": ["삼성SDI"],
}

_OPINION_MAP = {
    "매수": "Buy",
    "buy": "Buy",
    "outperform": "Buy",
    "강력매수": "Strong Buy",
    "적극매수": "Strong Buy",
    "strongbuy": "Strong Buy",
    "tradingbuy": "Trading Buy",
    "단기매수": "Trading Buy",
    "중립": "Hold",
    "보유": "Hold",
    "hold": "Hold",
    "neutral": "Hold",
    "marketperform": "Hold",
    "매도": "Sell",
    "sell": "Sell",
    "underperform": "Sell",
    "비중확대": "Overweight",
    "overweight": "Overweight",
    "비중축소": "Underweight",
    "underweight": "Underweight",
    "시장수익률": "Hold",
    "reduce": "Sell",
    "없음": None,
    "투자의견없음": None,
    "nr": None,
    "n/r": None,
    "notrated": None,
    "": None,
}


class DartError(RuntimeError):
    """OpenDART returned an error status (e.g. 010 unregistered key, 020 rate limit)."""


# ================================================================================ pure helpers


def parse_number(value: Any) -> float | None:
    """First signed decimal in a Naver/WiseReport string ('12.85배', '22,292원', '-341') -> float."""
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return None if (isinstance(value, float) and math.isnan(value)) else float(value)
    m = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(m.group()) if m else None


def parse_krw_korean(value: str | None) -> float | None:
    """'1,674조 9,588억' -> 1.6749588e15 KRW (조 = 1e12, 억 = 1e8, 만 = 1e4)."""
    if not value:
        return None
    s = str(value).replace(",", "").replace(" ", "")
    total = 0.0
    for unit, mult in (("조", 1e12), ("억", 1e8), ("만", 1e4)):
        m = re.search(rf"(\d+(?:\.\d+)?){unit}", s)
        if m:
            total += float(m.group(1)) * mult
    return total or None


def check_code(code: str, *, lang: str = "ko") -> str:
    """Canonical 6-char KRX code or ValueError (US tickers belong to the US provider)."""
    c = (code or "").strip().upper()
    c = re.sub(r"\.(KS|KQ|KN)$", "", c)
    if not _CODE_RE.fullmatch(c):
        raise ValueError(
            _tr(
                lang,
                f"KRX 6자리 종목코드가 아님: {code!r} (미국 종목은 US 데이터 계층 사용)",
                f"Not a 6-character KRX stock code: {code!r} (US tickers use the US data layer)",
            )
        )
    return c


def is_preferred_code(code: str) -> bool:
    """KRX preferred shares (우선주) have a short code that does not end in '0' (005935, 005387, 00680K)."""
    return bool(_CODE_RE.fullmatch(code or "")) and not code.endswith("0")


def common_code(code: str) -> str:
    """Common-share code of the same issuer (005935 -> 005930, 00680K -> 006800); identity for commons."""
    return code[:5] + "0" if is_preferred_code(code) else code


def _kst_date_to_utc(d: date, hour: int = DATE_ONLY_HOUR_KST) -> datetime:
    return datetime(d.year, d.month, d.day, hour, tzinfo=KST).astimezone(UTC)


def _strip_tags(s: str) -> str:
    text = html_lib.unescape(re.sub(r"<[^>]+>", " ", s or ""))
    return re.sub(r"\s+", " ", text).strip()


def html_to_text(markup: str, max_chars: int | None = None) -> str:
    """HTML -> LLM-friendly plain text; table rows are flattened to 'a | b | c' lines."""
    soup = BeautifulSoup(markup or "", "lxml")
    for x in soup(["script", "style", "head"]):
        x.decompose()
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th", "te", "tu"])]
        tr.replace_with(soup.new_string(" | ".join(c for c in cells if c) + "\n"))
    txt = soup.get_text("\n")
    txt = re.sub(r"[ \t\xa0　]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt).strip()
    return txt[:max_chars] if max_chars else txt


def clean_text(text: str) -> str:
    """Collapse runs of spaces / blank lines and drop replacement characters."""
    t = (text or "").replace("�", "")
    t = re.sub(r"[ \t\xa0　]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return re.sub(r"\n{2,}", "\n", t).strip()


def truncate_head_tail(text: str, max_chars: int, head_frac: float = 0.7) -> str:
    """Keep the head and the tail of a long document (result length <= max_chars)."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = "\n…(중략)…\n"
    budget = max_chars - len(marker)
    if budget <= 0:
        return text[:max_chars]
    head = int(budget * head_frac)
    tail = budget - head
    return text[:head].rstrip() + marker + (text[-tail:].lstrip() if tail > 0 else "")


def normalize_opinion(op: str | None) -> str | None:
    """Broker opinion string ('매수', 'OutPerform', 'StrongBuy', '없음') -> canonical label or None."""
    k = re.sub(r"\s+", "", op or "").lower()
    for key in (
        k,
        re.sub(r"\((유지|상향|하향|신규|maintain|upgrade|downgrade|initiate)\)$", "", k),
    ):
        if key in _OPINION_MAP:
            return _OPINION_MAP[key]
    return op.strip() if op else None


def recomm_mean_label(mean: float | None) -> str | None:
    """Naver/FnGuide consensus mean (1 = 매도 .. 5 = 적극매수; higher is more bullish) -> 'Buy (4.00/5)'."""
    if mean is None or not (1.0 <= mean <= 5.0):
        return None
    if mean >= 4.5:
        label = "Strong Buy"
    elif mean >= 3.5:
        label = "Buy"
    elif mean >= 2.5:
        label = "Hold"
    else:
        label = "Sell"
    return f"{label} ({mean:.2f}/5)"


def company_aliases(code: str, name: str) -> list[str]:
    """Name + curated press aliases + '(주)'-stripped / space-free variants, longest first."""
    base = [name, *(_ALIASES.get(code, []))]
    out: list[str] = []
    for a in base:
        for v in (a, re.sub(r"\(주\)|주식회사", "", a).strip(), a.replace(" ", "")):
            if v and len(v) >= 2 and v not in out:
                out.append(v)
    return sorted(out, key=len, reverse=True)


def relevance_score(title: str, summary: str, aliases: Iterable[str]) -> int:
    """2 = company name/alias in the title, 1 = only in the summary, 0 = generic."""
    al = [a.casefold() for a in aliases if a]
    t, s = (title or "").casefold(), (summary or "").casefold()
    if any(a in t for a in al):
        return 2
    if any(a in s for a in al):
        return 1
    return 0


def _norm_title(t: str) -> str:
    return re.sub(r"[\W_]+", "", t or "").lower()[:40]


# ------------------------------------------------------------------------------ prices (pure)


def parse_fchart(text: str, symbol: str, *, lang: str = "ko") -> pd.DataFrame:
    """Naver fchart XML -> PRICE_COLUMNS frame (tz-naive KST dates). Trading-halt rows (O/H/L = 0)
    become flat bars at the close."""
    rows = re.findall(r'<item data="(.*?)"\s*/>', text or "")
    if not rows:
        raise ValueError(_tr(lang, f"네이버 fchart: {symbol!r} 데이터 없음", f"Naver fchart: no data for {symbol!r}"))
    df = pd.read_csv(
        StringIO("\n".join(rows)),
        sep="|",
        header=None,
        dtype={"date": str},
        names=["date", *PRICE_COLUMNS],
        na_values=["", "null"],
    )
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), format="%Y%m%d"), name=None)
    df = df.apply(pd.to_numeric, errors="coerce").astype("float64")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["close"] > 0]
    for col in ("open", "high", "low"):
        bad = ~(df[col] > 0)
        df.loc[bad, col] = df.loc[bad, "close"]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
    return df[PRICE_COLUMNS]


def parse_daum_official_closes(payload: dict) -> pd.Series:
    """Daum day candles -> official KRX close of day t = base price of day t+1 (tradePrice - signed
    changePrice). The latest candle has no successor, so its date is NOT in the result."""
    data = (payload or {}).get("data") or []
    if not data:
        raise ValueError("Daum 차트: 데이터 없음")
    d = pd.DataFrame(data)
    d.index = pd.DatetimeIndex(pd.to_datetime(d["date"]).dt.normalize())
    d = d[~d.index.duplicated(keep="last")].sort_index()
    change = d["changePrice"].astype(float)
    if "change" in d:  # be robust to an unsigned changePrice
        sign = d["change"].map(
            {
                "RISE": 1.0,
                "UPPER_LIMIT": 1.0,
                "FALL": -1.0,
                "LOWER_LIMIT": -1.0,
                "EVEN": 0.0,
            }
        )
        change = np.where(sign.notna(), sign.fillna(0.0) * change.abs(), change)
    base = d["tradePrice"].astype(float) - change
    return pd.Series(base, index=d.index).shift(-1).dropna().rename("official_close")


def session_phase(now_kst: datetime) -> str:
    """'intraday' (< 16:00 KST: today's bar is partial), 'post_krx' (16:00-20:00: KRX final, NXT still
    trading) or 'post_nxt' (>= 20:00: everything final)."""
    t = now_kst.astimezone(KST).time()
    if t < SESSION_FINAL:
        return "intraday"
    if t < NXT_CLOSE:
        return "post_krx"
    return "post_nxt"


def drop_partial_bar(df: pd.DataFrame, now_kst: datetime, confirmed_today: bool = True) -> pd.DataFrame:
    """Drop a bar dated today (KST) while it is still forming.

    - intraday (before 16:00 KST): always dropped;
    - 16:00-20:00: dropped unless today's KRX official close was confirmed (Naver's close is the
      moving NXT-integrated price until 20:00);
    - after 20:00: kept.
    """
    if df.empty:
        return df
    today = pd.Timestamp(now_kst.astimezone(KST).date())
    if df.index[-1] != today:
        return df
    phase = session_phase(now_kst)
    if phase == "intraday" or (phase == "post_krx" and not confirmed_today):
        return df.iloc[:-1]
    return df


def apply_official_close_fix(
    df: pd.DataFrame,
    official: pd.Series | None,
    last_close: tuple[pd.Timestamp, float, str] | None = None,
    since: pd.Timestamp = NXT_SINCE,
    official_src: str = "krx_official(daum)",
    rel_tol: float = OFFICIAL_REL_TOL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace Naver's KRX+NXT integrated closes (bars >= ``since``) with KRX official closes.

    ``official``: official close per date (Daum base price of the next day, or guarded Yahoo closes);
    ``last_close``: (date, price, source) for the latest bar (Daum ``regularTradePrice``).
    A candidate is accepted only inside the bar's integrated [low, high] (the KRX close must lie in it;
    this rejects adjustment-basis mismatches). High/Low are widened to include the new close; Open/High/Low
    of recent bars otherwise stay integrated (documented, slightly inflates ATR).

    Returns (fixed frame, meta) with meta = {"fixed": {src: n}, "verified": n, "unverified": [iso dates],
    "rejected": [iso dates]} - "unverified" lists integrated-window bars whose close could not be confirmed.
    """
    out = df.copy()
    meta: dict[str, Any] = {
        "fixed": {},
        "verified": 0,
        "unverified": [],
        "rejected": [],
    }
    if out.empty:
        return out, meta
    window = np.asarray(out.index >= since)
    verified = pd.Series(False, index=out.index)

    def _inside(dates: pd.DatetimeIndex, px: np.ndarray) -> np.ndarray:
        lo = out.loc[dates, "low"].to_numpy() * (1 - rel_tol)
        hi = out.loc[dates, "high"].to_numpy() * (1 + rel_tol)
        return (px >= lo) & (px <= hi)

    if official is not None and len(official):
        off = official.astype(float).dropna()
        off = off[~off.index.duplicated(keep="last")]
        idx = out.index[window].intersection(off.index)
        if len(idx):
            o = off.reindex(idx).to_numpy()
            ok = _inside(idx, o)
            gap = np.abs(out.loc[idx, "close"].to_numpy() - o)
            fix = ok & (gap >= np.maximum(1.0, rel_tol * o))
            if fix.any():
                out.loc[idx[fix], "close"] = np.round(o[fix])
                meta["fixed"][official_src] = int(fix.sum())
            verified.loc[idx[ok]] = True
            meta["rejected"] += [d.date().isoformat() for d in idx[~ok]]

    if last_close is not None:
        d, px, src = last_close
        d = pd.Timestamp(d).normalize()
        if px and px > 0 and d == out.index[-1] and window[-1]:
            if bool(_inside(pd.DatetimeIndex([d]), np.array([float(px)]))[0]):
                if abs(out.at[d, "close"] - float(px)) >= 0.5:
                    out.at[d, "close"] = float(px)
                    meta["fixed"][src] = meta["fixed"].get(src, 0) + 1
                verified.at[d] = True
            else:
                meta["rejected"].append(d.date().isoformat())

    out["high"] = out[["high", "close"]].max(axis=1)
    out["low"] = out[["low", "close"]].min(axis=1)
    ver = verified.to_numpy()
    meta["verified"] = int((ver & window).sum())
    meta["unverified"] = [d.date().isoformat() for d in out.index[window & ~ver]]
    return out, meta


def yahoo_adjusted_ohlcv(h: pd.DataFrame, *, lang: str = "ko") -> pd.DataFrame:
    """yfinance ``history(auto_adjust=False)`` -> dividend-adjusted PRICE_COLUMNS via Adj Close / Close."""
    need = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in need if c not in h.columns]
    if missing:
        raise ValueError(_tr(lang, f"yfinance 컬럼 누락: {missing}", f"yfinance columns missing: {missing}"))
    close = h["Close"].astype(float)
    ratio = (h["Adj Close"].astype(float) / close).where(close > 0)
    out = pd.DataFrame(
        {
            "open": h["Open"].astype(float) * ratio,
            "high": h["High"].astype(float) * ratio,
            "low": h["Low"].astype(float) * ratio,
            "close": h["Adj Close"].astype(float),
            "volume": h["Volume"].astype(float).fillna(0.0),
        },
        index=h.index,
    )
    out = out[(out["close"] > 0) & out["close"].notna()].astype("float64")
    # halt rows (O/H/L = 0 or NaN) -> flat at the close; Yahoo's stale closes can sit outside [low, high]
    # (005930 2024-10-14, 000660 2023-02-02 ...) -> restore the OHLC invariant like parse_fchart does
    for col in ("open", "high", "low"):
        bad = ~(out[col] > 0)
        out.loc[bad, col] = out.loc[bad, "close"]
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)
    return out


def _naive_daily_index(idx: pd.Index) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(idx)
    if di.tz is not None:
        di = di.tz_localize(None)
    return di.normalize()


# ------------------------------------------------------------------------ fundamentals (pure)


def parse_naver_basic(j: dict, *, lang: str = "ko") -> dict[str, Any]:
    """m.stock basic -> {'name', 'exchange' (KOSPI/KOSDAQ/KONEX), 'end_type'}."""
    ex = (j or {}).get("stockExchangeType") or {}
    exchange = EXCHANGES.get(ex.get("code") or "") or ex.get("name")
    name = (j or {}).get("stockName")
    if not name or not exchange:
        raise ValueError(_tr(lang, "네이버 basic: 종목명/시장 없음", "Naver basic: no name/market"))
    return {"name": name, "exchange": exchange, "end_type": j.get("stockEndType")}


def parse_naver_integration(j: dict, *, lang: str = "ko") -> dict[str, Any]:
    """m.stock integration -> TTM PER/EPS/PBR/BPS (NXT-integrated price), consensus, market cap, industry."""
    if not j or not j.get("totalInfos"):
        raise ValueError(
            _tr(
                lang,
                "네이버 integration: totalInfos 없음 (잘못된 코드?)",
                "Naver integration: no totalInfos (wrong code?)",
            )
        )
    ti = {x.get("code"): x for x in j.get("totalInfos") or []}

    def g(k: str) -> float | None:
        return parse_number((ti.get(k) or {}).get("value"))

    cons = j.get("consensusInfo") or {}
    return {
        "name": j.get("stockName"),
        "prev_close": g("lastClosePrice"),  # KRX official close of the day before the latest bar
        "per": g("per"),
        "eps": g("eps"),
        "pbr": g("pbr"),
        "bps": g("bps"),
        "per_basis": (ti.get("per") or {}).get("valueDesc"),
        "fwd_per": g("cnsPer"),
        "fwd_eps": g("cnsEps"),
        "div_yield_pct": g("dividendYieldRatio"),
        "dps": g("dividend"),
        "market_cap": parse_krw_korean((ti.get("marketValue") or {}).get("value")),
        "high_52w_adj": g("highPriceOf52WeeksAdjusted") or g("highPriceOf52Weeks"),
        "low_52w_adj": g("lowPriceOf52WeeksAdjusted") or g("lowPriceOf52Weeks"),
        "target_mean": parse_number(cons.get("priceTargetMean")),
        "recomm_mean": parse_number(cons.get("recommMean")),
        "consensus_date": cons.get("createDate"),
        "industry_code": j.get("industryCode"),
        "researches": [
            {
                "date": x.get("wdt"),
                "broker": x.get("bnm"),
                "title": x.get("tit"),
                "id": x.get("id"),
            }
            for x in (j.get("researches") or [])
        ],
    }


def parse_naver_finance(j: dict, *, lang: str = "ko") -> pd.DataFrame:
    """m.stock finance/annual|quarter -> rows (매출액, 영업이익, 지배주주순이익, ROE ...) x period columns
    'YYYYMM' ('YYYYMM(E)' for consensus). Money rows are in 억원, ratios in %."""
    fi = (j or {}).get("financeInfo")
    if not fi:
        raise ValueError(_tr(lang, "네이버 재무: financeInfo 없음", "Naver financials: no financeInfo"))
    cols = {t["key"]: t["key"] + ("(E)" if t.get("isConsensus") == "Y" else "") for t in fi["trTitleList"]}
    data = {
        row["title"]: {
            cols[k]: parse_number((v or {}).get("value")) for k, v in (row.get("columns") or {}).items() if k in cols
        }
        for row in fi.get("rowList") or []
    }
    return pd.DataFrame(data, dtype="float64").T.reindex(columns=list(cols.values()))


def _actual_cols(fin: pd.DataFrame) -> list[str]:
    return sorted(c for c in fin.columns if not str(c).endswith("(E)"))


def _cell(fin: pd.DataFrame, row: str, col: str) -> float | None:
    if row not in fin.index or col not in fin.columns:
        return None
    v = fin.at[row, col]
    return None if pd.isna(v) else float(v)


def _yoy(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    return a / b - 1.0


def quarterly_yoy(fin_q: pd.DataFrame, row: str) -> float | None:
    """Latest reported quarter vs the same quarter one year earlier (None if missing / base <= 0)."""
    cols = _actual_cols(fin_q)
    if not cols:
        return None
    last = cols[-1]
    prev = f"{int(last[:4]) - 1}{last[4:6]}"
    return _yoy(_cell(fin_q, row, last), _cell(fin_q, row, prev))


def annual_yoy(fin_a: pd.DataFrame, row: str) -> float | None:
    cols = _actual_cols(fin_a)
    if len(cols) < 2:
        return None
    return _yoy(_cell(fin_a, row, cols[-1]), _cell(fin_a, row, cols[-2]))


def ttm_operating_margin(fin_q: pd.DataFrame) -> float | None:
    cols = _actual_cols(fin_q)[-4:]
    if len(cols) < 4:
        return None
    op = [_cell(fin_q, "영업이익", c) for c in cols]
    rev = [_cell(fin_q, "매출액", c) for c in cols]
    if any(v is None for v in op + rev) or sum(rev) <= 0:  # type: ignore[arg-type]
        return None
    return sum(op) / sum(rev)  # type: ignore[arg-type]


def parse_wisereport_snapshot(page: str, *, lang: str = "ko") -> dict[str, Any]:
    """WiseReport c1010001 page -> PER / 업종PER / PBR / EPS / BPS / dividend / WICS / official close /
    market cap / consensus (opinion 1-5, target, forward EPS/PER, broker count)."""
    soup = BeautifulSoup(page or "", "lxml")

    def txt(el: Any) -> str:
        return re.sub(r"\s+", " ", el.get_text(" ", strip=True))

    head = [txt(x) for x in soup.select("td.cmp-table-cell dt")]
    if not head:
        raise ValueError(
            _tr(
                lang,
                "WiseReport: 기업 헤더 없음 (잘못된 코드 또는 페이지 변경)",
                "WiseReport: no company header (wrong code or page changed)",
            )
        )
    hd: dict[str, str] = {}
    for h in head:
        for k in ["PER", "업종PER", "PBR", "EPS", "BPS", "현금배당수익률", "WICS"]:
            if h.startswith(k + " ") or h.startswith(k + " :"):
                hd[k] = h.split(":", 1)[1].strip() if k == "WICS" else h[len(k) :].strip()
    krx_sector = next(
        (h.split(":", 1)[1].strip() for h in head if h.startswith(("KOSPI :", "KOSDAQ :", "KONEX :"))),
        None,
    )
    q: dict[str, str] = {}
    tb = soup.find("table", id="cTB11")
    for tr in tb.find_all("tr") if tb else []:
        th, td = tr.find("th"), tr.find("td")
        if th and td:
            q[txt(th)] = txt(td)
    cons: dict[str, float | None] = {}
    ct = soup.find("table", id="cTB15")
    if ct and ct.find_all("tr"):
        vals = [txt(td) for td in ct.find_all("tr")[-1].find_all("td")]
        if len(vals) >= 5:
            cons = dict(
                zip(
                    ["opinion", "target", "eps", "per", "n_brokers"],
                    map(parse_number, vals[:5]),
                )
            )
    price_part = q.get("주가/전일대비/수익률", "").split("/")
    hl = q.get("52Weeks 최고/최저", "/").split("/")
    cap_eok = parse_number(q.get("시가총액"))
    return {
        "official_close": parse_number(price_part[0]) if price_part and price_part[0] else None,
        "per": parse_number(hd.get("PER")),
        "sector_per": parse_number(hd.get("업종PER")),
        "pbr": parse_number(hd.get("PBR")),
        "eps": parse_number(hd.get("EPS")),
        "bps": parse_number(hd.get("BPS")),
        "div_yield_pct": parse_number(hd.get("현금배당수익률")),
        "wics": hd.get("WICS"),
        "krx_sector": krx_sector,
        "beta_52w": parse_number(q.get("52주베타")),
        "high_52w_adj": parse_number(hl[0]),
        "low_52w_adj": parse_number(hl[-1]),
        "market_cap": cap_eok * 1e8 if cap_eok else None,  # 억원
        "consensus": cons,
    }


def parse_encparam(page: str, *, lang: str = "ko") -> str:
    m = re.search(r"encparam:\s*'([^']+)'", page or "")
    if not m:
        raise ValueError(
            _tr(
                lang,
                "WiseReport c1040001: encparam 없음 (잘못된 코드 또는 페이지 변경)",
                "WiseReport c1040001: no encparam (wrong code or page changed)",
            )
        )
    return m.group(1)


def parse_wisereport_cf4002(j: dict, *, lang: str = "ko") -> pd.DataFrame:
    """WiseReport cF4002 JSON -> per-period EPS/BPS/PER/PBR/... indexed by period end, + ``is_est``."""
    periods = [re.sub(r"<br\s*/?>.*", "", y) for y in (j or {}).get("YYMM") or []]
    keep = [i for i, p in enumerate(periods) if re.match(r"\d{4}/\d{2}", p)]
    if not keep:
        raise ValueError(_tr(lang, "WiseReport cF4002: 기간 컬럼 없음", "WiseReport cF4002: no period columns"))
    rows: dict[str, list[Any]] = {}
    for row in j.get("DATA") or []:
        nm = row.get("ACC_NM") or ""
        if nm.startswith("보통주.수정주가"):
            nm = "수정주가"
        if (
            nm
            in (
                "EPS",
                "BPS",
                "PER",
                "PBR",
                "PCR",
                "PSR",
                "EV/EBITDA",
                "DPS",
                "현금배당수익률",
                "수정주가",
            )
            and nm not in rows
        ):
            rows[nm] = [row.get(f"DATA{i + 1}") for i in keep]
    df = pd.DataFrame(rows, index=[periods[i] for i in keep])
    df["is_est"] = df.index.str.contains(r"\(E\)")
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index.str[:7], format="%Y/%m") + pd.offsets.MonthEnd(0))
    return df.astype({c: "float64" for c in df.columns if c != "is_est"})


def build_valuation_history(annual: pd.DataFrame, quarterly: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Point-in-time TTM PER / PBR per trading day (no look-ahead).

    Knots: fiscal-year EPS/BPS usable from FY-end + 90 days, rolling 4-quarter EPS from quarter-end + 45 days
    (Korean filing deadlines). Between the last FY knot and the first 4-quarter TTM knot the EPS is the
    (stale) FY value. PER is NaN when TTM EPS <= 0 (loss-making); PBR NaN when BPS <= 0.
    """
    a = annual[~annual["is_est"].astype(bool)]
    knots = pd.DataFrame({"eps_ttm": a["EPS"], "bps": a["BPS"]})
    knots.index = knots.index + pd.Timedelta(90, "D")
    q = quarterly[~quarterly["is_est"].astype(bool)].sort_index()
    qk = pd.DataFrame({"eps_ttm": q["EPS"].rolling(4).sum(), "bps": q["BPS"]}).dropna()
    qk = qk[qk.index.month != 12]  # December quarters are covered by the FY knot
    qk.index = qk.index + pd.Timedelta(45, "D")
    knots = pd.concat([knots, qk]).sort_index()
    knots = knots[~knots.index.duplicated(keep="last")]
    knots.index = pd.DatetimeIndex(knots.index).astype("datetime64[ns]")
    px = close.dropna().astype("float64").sort_index().to_frame("close")
    px.index = pd.DatetimeIndex(px.index).astype("datetime64[ns]")
    v = pd.merge_asof(px, knots, left_index=True, right_index=True)
    v = v.dropna(subset=["eps_ttm"])
    out = pd.DataFrame(index=v.index)
    out["per"] = v["close"] / v["eps_ttm"].where(v["eps_ttm"] > 0)
    out["pbr"] = v["close"] / v["bps"].where(v["bps"] > 0)
    return out.astype("float64")


def compose_fundamentals(
    snap: dict[str, Any] | None,
    wise: dict[str, Any] | None,
    fin_a: pd.DataFrame | None,
    fin_q: pd.DataFrame | None,
    yf_targets: dict[str, Any] | None,
    as_of: date,
    preferred: bool = False,
    *,
    lang: str = "ko",
) -> tuple[Fundamentals, list[str]]:
    """Merge the KR sources into one :class:`Fundamentals` (definitions: see module docstring).

    ``preferred`` (우선주): WiseReport serves the COMMON share's page (its PER/PBR/market cap/yield are
    priced on the common) and every consensus source (Naver ``priceTargetMean``, WiseReport cTB15, Yahoo)
    is the common share's target/opinion. For a preferred share PER/PBR/yield/market cap therefore come
    from Naver only (preferred price, company EPS/BPS) and the analyst target/rating/count are dropped
    (a common-share target against a preferred price would fake a large upside) - recorded in the notes.

    Returns (fundamentals, missing_notes). Raises if neither Naver nor WiseReport data is available.
    """
    if not snap and not wise:
        raise RuntimeError(
            _tr(lang, "네이버·WiseReport 펀더멘털 모두 수집 실패", "Naver and WiseReport fundamentals both unavailable")
        )
    s, w = snap or {}, wise or {}
    notes: list[str] = []
    if preferred:
        # keep only company-level WiseReport items (EPS/BPS/sector PER/WICS); drop common-share pricing
        w = {k: v for k, v in w.items() if k in ("eps", "bps", "sector_per", "wics", "krx_sector")}
        s = {k: v for k, v in s.items() if k not in ("target_mean", "recomm_mean")}
        yf_targets = None
        notes.append(
            _tr(
                lang,
                "우선주: 컨센서스 목표가·투자의견은 보통주 기준이므로 미사용, "
                "PER/PBR·시가총액은 네이버(우선주 가격) 기준",
                "Preferred share: consensus target price · rating refer to the common share and are not used; "
                "PER/PBR · market cap are Naver's (preferred-share price)",
            )
        )
    cons = w.get("consensus") or {}
    sources: list[str] = []
    if snap:
        sources.append("Naver")
    if wise:
        sources.append("WiseReport")

    def first(*vals: float | None) -> float | None:
        return next((v for v in vals if v is not None), None)

    eps, bps = first(s.get("eps"), w.get("eps")), first(s.get("bps"), w.get("bps"))
    roe = eps / bps if (eps is not None and bps and bps > 0) else None
    op_margin = ttm_operating_margin(fin_q) if fin_q is not None else None
    rev_g = quarterly_yoy(fin_q, "매출액") if fin_q is not None else None
    ni_row = "지배주주순이익"
    if fin_q is not None and ni_row not in fin_q.index:
        ni_row = "당기순이익"
    earn_g = quarterly_yoy(fin_q, ni_row) if fin_q is not None else None
    if fin_a is not None:
        if roe is None and _actual_cols(fin_a):
            r = _cell(fin_a, "ROE", _actual_cols(fin_a)[-1])
            roe = r / 100.0 if r is not None else None
        if op_margin is None and _actual_cols(fin_a):
            m = _cell(fin_a, "영업이익률", _actual_cols(fin_a)[-1])
            op_margin = m / 100.0 if m is not None else None
        rev_g = rev_g if rev_g is not None else annual_yoy(fin_a, "매출액")
        a_ni = ni_row if ni_row in fin_a.index else "당기순이익"
        earn_g = earn_g if earn_g is not None else annual_yoy(fin_a, a_ni)
    if fin_q is not None or fin_a is not None:
        sources.append("Naver재무")

    target = first(s.get("target_mean"), cons.get("target"))
    t_high = t_low = None
    count = int(cons["n_brokers"]) if cons.get("n_brokers") else None
    if yf_targets:
        hi, lo = (
            parse_number(yf_targets.get("high")),
            parse_number(yf_targets.get("low")),
        )
        if hi and lo and (target is None or lo <= target <= hi):
            t_high, t_low = hi, lo
            sources.append("yfinance(목표가 범위)")
        if count is None and yf_targets.get("count"):
            count = int(yf_targets["count"])
    div = first(s.get("div_yield_pct"), w.get("div_yield_pct"))
    per_def = "우선주 PER=우선주가/보통주 TTM EPS(네이버)" if preferred else "PER=TTM(KRX 종가)"

    fund = Fundamentals(
        as_of=as_of,
        source="+".join(sources) + f" | {per_def}, 선행PER=컨센서스(당해 FY), 업종PER=WiseReport, 성장률=최근 분기 YoY",
        market_cap=first(s.get("market_cap"), w.get("market_cap")),
        per=first(w.get("per"), s.get("per")),
        forward_per=first(s.get("fwd_per"), cons.get("per")),
        pbr=first(w.get("pbr"), s.get("pbr")),
        eps=eps,
        forward_eps=first(s.get("fwd_eps"), cons.get("eps")),
        bps=bps,
        roe=roe,
        operating_margin=op_margin,
        dividend_yield=div / 100.0 if div is not None else None,
        revenue_growth_yoy=rev_g,
        earnings_growth_yoy=earn_g,
        sector_per=w.get("sector_per"),
        analyst_target_price=target,
        analyst_target_high=t_high,
        analyst_target_low=t_low,
        analyst_count=count,
        analyst_rating=recomm_mean_label(first(s.get("recomm_mean"), cons.get("opinion"))),
    )
    if not wise:
        notes.append(
            _tr(
                lang,
                "WiseReport 수집 실패 → 업종PER 없음, PER/PBR은 네이버(NXT 통합가) 기준",
                "WiseReport unavailable → no sector PER; PER/PBR from Naver (NXT-integrated price)",
            )
        )
    if not snap:
        notes.append(
            _tr(
                lang,
                "네이버 종목 요약 수집 실패 → WiseReport 값만 사용",
                "Naver stock summary unavailable → WiseReport only",
            )
        )
    if fin_q is None and fin_a is None:
        notes.append(
            _tr(
                lang,
                "네이버 재무제표 수집 실패 → 성장률·영업이익률 없음",
                "Naver financial statements unavailable → no growth rates · operating margin",
            )
        )
    elif fin_q is None:
        notes.append(
            _tr(
                lang,
                "네이버 분기 재무 수집 실패 → 성장률·영업이익률은 최근 연간 기준 (최근 분기 미반영)",
                "Naver quarterly financials unavailable → growth rates · operating margin from the latest fiscal "
                "year (latest quarter not reflected)",
            )
        )
    return fund, notes


# ------------------------------------------------------------------------------- text (pure)


def parse_naver_news(clusters: Any, cutoff: datetime | None = None) -> tuple[list[dict[str, Any]], datetime | None]:
    """m.stock news JSON (list of clusters) -> (rows newer than ``cutoff``, oldest timestamp seen).
    ``datetime`` is KST 'YYYYMMDDHHMM'; rows carry UTC ``published_at``."""
    rows: list[dict[str, Any]] = []
    oldest: datetime | None = None
    for cl in clusters or []:
        size = int(cl.get("total") or 1)
        for it in cl.get("items") or []:
            try:
                ts = datetime.strptime(str(it["datetime"]), "%Y%m%d%H%M").replace(tzinfo=KST).astimezone(UTC)
            except (KeyError, ValueError):
                continue
            oldest = ts if oldest is None or ts < oldest else oldest
            if cutoff is not None and ts < cutoff:
                continue
            title = _strip_tags(it.get("titleFull") or it.get("title") or "")
            if not title:
                continue
            oid, aid = it.get("officeId"), it.get("articleId")
            url = it.get("mobileNewsUrl") or (
                f"https://n.news.naver.com/mnews/article/{oid}/{aid}" if oid and aid else ""
            )
            rows.append(
                {
                    "title": title,
                    "summary": _strip_tags(it.get("body") or "").rstrip(". ").strip(),
                    "url": url,
                    "source": it.get("officeName") or "",
                    "published_at": ts,
                    "provider": "naver",
                    "cluster_size": size,
                }
            )
    return rows, oldest


def parse_google_news_rss(content: bytes | str, cutoff: datetime | None = None) -> list[dict[str, Any]]:
    """Google News RSS -> rows (title without ' - 출처', publisher, UTC time). Titles only (no summary)."""
    import feedparser

    feed = feedparser.parse(content)
    rows: list[dict[str, Any]] = []
    for e in feed.entries:
        pp = e.get("published_parsed")
        if not pp:
            continue
        ts = datetime(*pp[:6], tzinfo=UTC)
        if cutoff is not None and ts < cutoff:
            continue
        src = e.get("source") or {}
        stitle = src.get("title") if hasattr(src, "get") else None
        title = e.get("title") or ""
        if stitle and title.endswith(" - " + stitle):
            title = title[: -(len(stitle) + 3)]
        title = _strip_tags(title)
        if not title:
            continue
        rows.append(
            {
                "title": title,
                "summary": "",
                "url": e.get("link") or "",
                "source": stitle or "",
                "published_at": ts,
                "provider": "google",
                "cluster_size": 1,
            }
        )
    return rows


def parse_naver_article(page: str) -> dict[str, Any]:
    """n.news.naver.com article page -> {'title', 'body', 'published_at' (UTC) , 'press'}."""
    soup = BeautifulSoup(page or "", "lxml")
    area = soup.select_one("#dic_area") or soup.select_one("#newsct_article") or soup.select_one("article")
    body = ""
    if area:
        for x in area.select("em.img_desc, span.end_photo_org, div.ab_photo, script, style, .media_end_summary"):
            x.decompose()
        body = re.sub(r"\n\s*\n+", "\n", area.get_text("\n", strip=True))
    ts = soup.select_one("span._ARTICLE_DATE_TIME")
    press = soup.select_one("a.media_end_head_top_logo img")
    title = soup.select_one("#title_area") or soup.select_one("h2.media_end_head_headline")
    published = None
    if ts is not None and ts.get("data-date-time"):
        try:
            published = datetime.strptime(ts["data-date-time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST).astimezone(UTC)
        except ValueError:
            published = None
    return {
        "title": title.get_text(strip=True) if title else None,
        "body": clean_text(body),
        "published_at": published,
        "press": press.get("alt") if press else None,
    }


def select_news(rows: list[dict[str, Any]], aliases: list[str], limit: int) -> tuple[list[dict[str, Any]], bool]:
    """De-duplicate by normalised title (Naver wins), score relevance, keep company-specific articles.

    Returns (rows newest first, used_generic). Generic (relevance 0) market wraps are used only when no
    relevant article exists (``used_generic`` = True). With more candidates than ``limit``, title hits are
    preferred over summary-only hits, then recency.
    """
    ordered = sorted(
        rows,
        key=lambda r: (
            0 if r["provider"] == "naver" else 1,
            -r["published_at"].timestamp(),
        ),
    )
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for r in ordered:
        k = _norm_title(r["title"])
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(dict(r, relevance=relevance_score(r["title"], r["summary"], aliases)))
    pool = [r for r in uniq if r["relevance"] > 0]
    used_generic = False
    if not pool and uniq:
        pool, used_generic = uniq, True
    if len(pool) > limit:
        pool = sorted(pool, key=lambda r: (-r["relevance"], -r["published_at"].timestamp()))[:limit]
    pool.sort(key=lambda r: r["published_at"], reverse=True)
    return pool, used_generic


def parse_dart_search(page: str) -> tuple[list[dict[str, Any]], int]:
    """DART web detailSearch.ax result -> (rows, number of pages from '[p/N] [총 X건]')."""
    m = _PAGEINFO_RE.search(page or "")
    n_pages = int(m.group(2)) if m else 1
    soup = BeautifulSoup(page or "", "lxml")
    rows: list[dict[str, Any]] = []
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        a = tr.select_one("a[href*='rcpNo=']")
        if len(tds) < 5 or not a:
            continue
        rm_ = re.search(r"rcpNo=(\d{14})", a.get("href", ""))
        if not rm_:
            continue
        rcp = rm_.group(1)
        corp = tr.select_one("a[href*='openCorpInfoNew']")
        cm = re.search(r"openCorpInfoNew\('(\d{8})'", corp["href"]) if corp else None
        rm = tds[5].get_text(" ", strip=True) if len(tds) > 5 else ""
        try:
            dt = datetime.strptime(tds[4].get_text(strip=True), "%Y.%m.%d").date()
        except ValueError:
            continue
        rows.append(
            {
                "rcept_dt": dt,
                "rcept_no": rcp,
                "corp_code": cm.group(1) if cm else None,
                "report_nm": re.sub(r"\s+", " ", a.get_text(" ", strip=True)),
                "flr_nm": tds[3].get_text(strip=True),
                "rm": rm,
                "is_krx": is_krx_filing(rcp, rm),
                "url": f"{DART_WEB}/dsaf001/main.do?rcpNo={rcp}",
            }
        )
    return rows, n_pages


def is_krx_filing(rcept_no: str, rm: str = "") -> bool:
    """rcpNo digit 9 = '8' (KRX 유가증권) or '9' (KRX 코스닥), or rm marker 유/코/넥."""
    return (len(rcept_no) > 8 and rcept_no[8] in ("8", "9")) or bool(re.search(r"[유코넥]", rm or ""))


def parse_dart_viewer_nodes(
    main_page: str,
) -> list[tuple[str | None, str, str, str, str, str]]:
    """DART main.do -> [(section title | None, dcmNo, eleId, offset, length, dtd)].

    FSS documents expose a TOC (node blocks, dtd 'dart4.xsd'); KRX single-page documents only a
    ``viewDoc("rcp", "dcm", "0", "0", "0", "HTML")`` call.
    """

    def field(block: str, key: str) -> str:
        m = re.search(rf"\['{key}'\] = \"(\d+)\"", block)
        return m.group(1) if m else "0"

    nodes: list[tuple[str | None, str, str, str, str, str]] = []
    for _var, title, block, dtd in _NODE_RE.findall(main_page or ""):
        dcm = field(block, "dcmNo")
        if dcm == "0":
            continue
        nodes.append(
            (
                title,
                dcm,
                field(block, "eleId"),
                field(block, "offset"),
                field(block, "length"),
                dtd,
            )
        )
    if not nodes:
        m = _VIEWDOC_RE.search(main_page or "")
        if m:
            nodes = [(None, *m.groups()[1:])]
    return nodes


_KO_LEGACY_ENCODINGS = (
    "ms949",
    "euc-kr",
    "euc_kr",
    "cp949",
    "ks_c_5601-1987",
    "x-windows-949",
)


def decode_dart_viewer(content: bytes, encoding: str | None) -> str:
    """KRX single-page viewer documents are served as MS949 (-> cp949); FSS sections are utf-8.

    Without a charset in Content-Type, requests reports ISO-8859-1, which would silently turn Korean into
    mojibake (no replacement chars). Then the document's own ``<meta charset>`` decides, else strict
    utf-8, else cp949.
    """
    enc = (encoding or "").lower()
    if enc in ("", "iso-8859-1", "latin-1", "ascii", "us-ascii"):
        m = re.search(rb"charset\s*=\s*[\"']?([A-Za-z0-9_\-]+)", content[:2048])
        enc = m.group(1).decode("ascii").lower() if m else ""
        if not enc:
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                enc = "cp949"
    if enc in _KO_LEGACY_ENCODINGS:
        enc = "cp949"
    try:
        return content.decode(enc, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def normalise_report_name(report_nm: str) -> str:
    """Strip leading '[기재정정]' / '[첨부추가]' tags and whitespace (dedupe key)."""
    return re.sub(r"\s+", "", re.sub(r"^(\[[^\]]*\]\s*)+", "", report_nm or ""))


def classify_filing(report_nm: str) -> tuple[int, bool, str]:
    """-> (priority 0 = best, is_earnings, form_type).

    0 영업(잠정)실적 / 손익구조변동 (earnings release), 1 주요사항보고서 and material KRX events,
    2 periodic reports (financial statements, also is_earnings), 3 실적/IR/해명 context, 4 other.
    """
    nm = normalise_report_name(report_nm)
    if _EARNINGS_RELEASE_RE.search(nm):
        return (
            0,
            True,
            "영업(잠정)실적(공정공시)" if "잠정" in nm else "매출액또는손익구조변동",
        )
    if nm.startswith("주요사항보고서") or _MATERIAL_RE.search(nm):
        return 1, False, nm[:40]
    m = _PERIODIC_RE.search(nm)
    if m:
        return 2, True, m.group(0)
    if _CONTEXT_RE.search(nm):
        return 3, False, nm[:40]
    return 4, False, nm[:40]


def is_attachment_only_correction(report_nm: str) -> bool:
    """'[첨부정정]' / '[첨부추가]' re-filings change attachments only: their viewer document is just the
    correction notice (e.g. 035720 '[첨부정정] 주요사항보고서(회사합병결정)' = 570 chars, no decision body)."""
    return bool(re.match(r"\s*\[[^\]]*첨부[^\]]*\]", report_nm or ""))


def _dedupe_key(base_name: str, priority: int) -> str:
    """Consolidated and separate earnings releases of the same event share one slot (연결 preferred)."""
    return re.sub(r"^연결재무제표기준", "", base_name) if priority == 0 else base_name


def select_filings(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Rank DART rows by priority then recency; one row per normalised report name. Within a name: any
    filing with a real body beats an attachment-only correction ('[첨부정정]'), then the newest, then the
    consolidated (연결) earnings release over the separate one, then the original over a '[기재정정]'
    filed the same day. Deterministic."""
    enriched = []
    for r in rows:
        pri, earn, form = classify_filing(r["report_nm"])
        base = normalise_report_name(r["report_nm"])
        enriched.append(
            dict(
                r,
                priority=pri,
                is_earnings=earn,
                form_type=form,
                base_name=base,
                dedupe_key=_dedupe_key(base, pri),
                is_correction=r["report_nm"].lstrip().startswith("["),
                attachment_only=is_attachment_only_correction(r["report_nm"]),
            )
        )
    best: dict[str, dict[str, Any]] = {}
    for r in sorted(
        enriched,
        key=lambda r: (
            not r["attachment_only"],
            r["rcept_dt"],
            r["base_name"].startswith("연결"),
            not r["is_correction"],
            r["rcept_no"],
        ),
        reverse=True,
    ):
        best.setdefault(r["dedupe_key"], r)
    ranked = sorted(
        best.values(),
        key=lambda r: (r["priority"], -r["rcept_dt"].toordinal(), r["rcept_no"]),
    )
    return ranked[: max(0, n)]


# ------------------------------------------------------------------ earnings schedule (pure)

#: KRX 안내공시 that can announce an earnings date in advance (report names, whitespace-stripped).
SCHEDULE_NOTICE_RE = re.compile(r"결산실적공시예고|기업설명회\(?IR\)?개최")
#: An IR event IS an earnings release/call only when its purpose/summary says so. NDRs, broker
#: conferences, investor days and governance briefings are not (live 2026-09: 005380, 035420, 035720).
_EARNINGS_EVENT_RE = re.compile(r"실적\s*(?:발표|공시|설명회|컨퍼런스)|earnings\s*(?:release|call|conference)", re.I)
#: ... unless the event itself is an investor-relations roadshow ("2분기 실적발표 후 해외 NDR"): its
#: date is not the release date. ('실적발표 컨퍼런스콜' is not excluded: 컨퍼런스콜 != 컨퍼런스 참가.)
_NON_RELEASE_EVENT_RE = re.compile(
    r"NDR|논딜|로드쇼|roadshow|컨퍼런스\s*참[가석]|conference\s*참|Corp(?:orate)?\s*Day|"
    r"Investor\s*Day|기업\s*탐방|투자자\s*(?:미팅|면담)",
    re.I,
)
_DATE_TOKEN_RE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
#: Days of DART notices scanned for a schedule (notices come ~2-4 weeks before the event).
EARNINGS_NOTICE_LOOKBACK_DAYS = 60
EARNINGS_NOTICE_MAX_DOCS = 8


def _first_date(text: str) -> date | None:
    for m in _DATE_TOKEN_RE.finditer(text or ""):
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            continue
    return None


def _field(text: str, label_re: str) -> str:
    """Value of a numbered KRX form row: '3. 개최목적 | 2026년 2분기 경영실적 발표' -> the text after '|'."""
    m = re.search(rf"^\s*\d+\.\s*(?:{label_re})\s*\|\s*(.+)$", text or "", re.M)
    return m.group(1).strip() if m else ""


def parse_earnings_notice(text: str) -> dict[str, Any]:
    """KRX 안내공시 text (``html_to_text`` of the DART viewer) -> schedule info.

    Returns ``{"status", "date", "kind", "purpose"}``:
    - ``결산실적공시 예고`` -> ``"3. 결산실적 공시예정일 | YYYY-MM-DD"``: status "earnings", kind "preannounce";
    - ``기업설명회(IR) 개최`` (KOSPI "1. 일시 및 장소 | 일시 | date" or KOSDAQ "1. 일시 ... 시작일 | 종료일
      ... / date | date") -> status "earnings" when the purpose (개최/실시목적) or summary (주요 내용) is an
      earnings release/call, else "other" (NDR, conference, investor day);
    - anything unrecognised -> status "unparsed" (the caller reports it, it is never guessed).
    The event date is taken ONLY from item 1 (일시) or the 공시예정일 row, never from 결정일자 / 게재일시.
    """
    t = text or ""
    head = t[:300]
    if re.search(r"결산실적\s*공시\s*예고", head):
        d = _first_date(_field(t, r"결산실적\s*공시\s*예정일"))
        return {
            "status": "earnings" if d else "unparsed",
            "date": d,
            "kind": "preannounce",
            "purpose": "결산실적 공시예정일",
        }
    if re.search(r"기업설명회\s*\(?\s*IR\s*\)?\s*개최", head):
        purpose = _field(t, r"(?:개최|실시)\s*목적")
        summary = _field(t, r"주요\s*(?:설명회)?\s*내용(?:\s*\(요약\))?")
        sec = re.search(r"^\s*1\.\s*일시.*?(?=^\s*2\.\s)", t, re.M | re.S)
        d = _first_date(sec.group(0)) if sec else None
        if d is None or not (purpose or summary):
            return {"status": "unparsed", "date": d, "kind": "ir", "purpose": purpose}
        is_earn = bool(_EARNINGS_EVENT_RE.search(f"{purpose} {summary}")) and not _NON_RELEASE_EVENT_RE.search(purpose)
        return {
            "status": "earnings" if is_earn else "other",
            "date": d,
            "kind": "ir",
            "purpose": purpose or summary,
        }
    return {"status": "unparsed", "date": None, "kind": "", "purpose": ""}


def next_earnings_from_notices(
    events: Iterable[dict[str, Any]], today: date, released: Iterable[date] = ()
) -> tuple[date | None, bool | None]:
    """Earliest earnings event on/after ``today`` -> (date, is_estimate).

    A company-announced date (예고 / earnings-call IR) is confirmed (``is_estimate`` False). Notices that
    disagree (e.g. a '[기재정정]' moved the date) are resolved conservatively: the EARLIEST date, flagged
    as an estimate. A date on which an earnings release was already filed is dropped (it is out).
    """
    done = set(released)
    dates = sorted(
        {
            e["date"]
            for e in events
            if e.get("status") == "earnings"
            and e.get("date") is not None
            and e["date"] >= today
            and e["date"] not in done
        }
    )
    if not dates:
        return None, None
    return dates[0], len(dates) > 1


def parse_naver_research_list(data: Any, since: date) -> list[dict[str, Any]]:
    """m.stock research list -> rows written on/after ``since``, newest first."""
    rows = []
    for d in data or []:
        try:
            wd = datetime.strptime(d["writeDate"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        if wd < since:
            continue
        rows.append(
            {
                "date": wd,
                "broker": d.get("brokerName") or "",
                "title": _strip_tags(d.get("title") or ""),
                "research_id": d.get("researchId"),
                "preview": _strip_tags(d.get("previewContent") or ""),
            }
        )
    rows.sort(key=lambda r: (r["date"], r["research_id"] or 0), reverse=True)
    return rows


def parse_naver_research_detail(j: dict) -> dict[str, Any]:
    """Research detail -> opinion / target / price at write (RAW, usually the previous close) / summary.
    'prevGoalPrice' is mislabelled upstream (equals priceAtWriteDate) and is ignored."""
    rc = (j or {}).get("researchContent") or {}

    def num(v: Any) -> float | None:
        x = parse_number(v)
        return x if x else None

    op = (rc.get("opinion") or "").strip()
    return {
        "opinion_raw": op or None,
        "opinion": normalize_opinion(op),
        "goal_price": num(rc.get("goalPrice")),
        "price_at_write": num(rc.get("priceAtWriteDate")),
        "summary": html_to_text(rc.get("content") or ""),
        "pdf_url": rc.get("attachUrl") or "",
    }


def research_filing(row: dict[str, Any], detail: dict[str, Any], max_chars: int) -> FilingItem:
    """Broker research summary as a FilingItem (form_type '증권사 리포트', source 'Naver Research')."""
    lines = [
        f"[증권사 리포트] {row['broker']} · {row['date'].isoformat()}",
        f"제목: {row['title']}",
    ]
    meta = []
    if detail.get("opinion_raw"):
        norm = detail.get("opinion")
        meta.append(
            f"투자의견: {detail['opinion_raw']}" + (f" ({norm})" if norm and norm != detail["opinion_raw"] else "")
        )
    if detail.get("goal_price"):
        meta.append(f"목표주가: {detail['goal_price']:,.0f}원")
    if detail.get("price_at_write"):
        meta.append(f"작성일 기준 주가(수정 전): {detail['price_at_write']:,.0f}원")
    if meta:
        lines.append(" / ".join(meta))
    body = detail.get("summary") or row.get("preview") or ""
    text = clean_text("\n".join(lines) + "\n요약:\n" + body)
    rid = row.get("research_id")
    return FilingItem(
        id=text_id("naver_research", str(rid)),
        title=f"[{row['broker']}] {row['title']}",
        form_type="증권사 리포트",
        filed_at=_kst_date_to_utc(row["date"]),
        url=detail.get("pdf_url") or f"https://m.stock.naver.com/research/company/{rid}",
        text=truncate_head_tail(text, max_chars),
        is_earnings=False,
        source="Naver Research",
    )


def parse_naver_market_value(j: dict, market: str) -> list[dict[str, Any]]:
    """m.stock marketValue ranking page -> [{'code','name','end_type','market','market_cap'(KRW)}]."""
    out = []
    for s in (j or {}).get("stocks") or []:
        cap = parse_number(s.get("marketValueRaw"))
        if cap is None:
            eok = parse_number(s.get("marketValue"))  # 억원
            cap = eok * 1e8 if eok is not None else None
        out.append(
            {
                "code": s.get("itemCode") or "",
                "name": s.get("stockName") or "",
                "end_type": (s.get("stockEndType") or "stock").lower(),
                "market": market,
                "market_cap": cap,
            }
        )
    return out


def universe_exclusion_reason(row: dict[str, Any]) -> str | None:
    """Why a listing row is not an eligible common stock (None = eligible)."""
    code, name = str(row.get("code") or ""), str(row.get("name") or "")
    if not _CODE_RE.fullmatch(code):
        return "invalid_code"
    if (row.get("end_type") or "stock") != "stock":
        return "etf_etn"
    if not code.endswith("0"):
        return "preferred"
    if _SPAC_RE.search(name):
        return "spac"
    if _REIT_RE.search(name):
        return "reit"
    if code in LISTED_FUND_CODES:
        return "listed_fund"
    if _ETP_NAME_RE.search(name):
        return "etf_etn"
    if row.get("market") not in ("KOSPI", "KOSDAQ"):
        return "market"
    cap = row.get("market_cap")
    if cap is None or not np.isfinite(cap) or cap <= 0:
        return "no_market_cap"
    return None


def select_universe(rows: Iterable[dict[str, Any]], n: int) -> list[str]:
    """Top-n eligible common stocks by market cap (ties by code) - deterministic for a given listing."""
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        if universe_exclusion_reason(r) is None:
            prev = best.get(r["code"])
            if prev is None or r["market_cap"] > prev["market_cap"]:
                best[r["code"]] = r
    ranked = sorted(best.values(), key=lambda r: (-r["market_cap"], r["code"]))
    return [r["code"] for r in ranked[: max(0, n)]]


# --------------------------------------------------------------------------- OpenDART (pure)


def parse_corp_code_zip(content: bytes) -> dict[str, str]:
    """OpenDART corpCode.xml zip -> {stock_code: corp_code} (ElementTree; ~100k entries)."""
    from xml.etree import ElementTree

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        root = ElementTree.fromstring(z.read(z.namelist()[0]))
    out: dict[str, str] = {}
    for e in root.iter("list"):
        sc = (e.findtext("stock_code") or "").strip()
        cc = (e.findtext("corp_code") or "").strip()
        if sc and cc:
            out[sc] = cc
    return out


def document_zip_to_text(content: bytes, max_chars: int = 40000) -> str:
    """OpenDART document.xml zip (dart4.xsd XML, utf-8 or euc-kr members) -> plain text."""
    texts = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in sorted(z.namelist()):
            raw = z.read(name)
            m = re.search(rb'encoding="([^"]+)"', raw[:200])
            enc = (m.group(1).decode() if m else "utf-8").lower()
            enc = "cp949" if enc in ("euc-kr", "ms949", "ksc5601", "cp949") else enc
            try:
                s = raw.decode(enc)
            except (UnicodeDecodeError, LookupError):
                s = raw.decode("cp949", errors="replace")
            # lxml's HTML parser is lenient with DART's malformed XML; drop the declaration it warns about
            texts.append(html_to_text(re.sub(r"^\s*<\?xml[^>]*\?>", "", s)))
    return "\n".join(texts)[:max_chars]


def _dart_status_check(resp: requests.Response, *, lang: str = "ko") -> None:
    """Raise DartError for OpenDART error payloads (JSON status, XML <status>, or the HTML no-key page)."""
    if resp.content[:2] == b"PK":
        return
    ctype = resp.headers.get("content-type", "")
    if "json" in ctype:
        j = resp.json()
        if j.get("status") not in ("000", None):
            raise DartError(f"{j.get('status')}: {j.get('message')}")
    elif "xml" in ctype:
        m = re.search(rb"<status>(\d+)</status>\s*<message>(.*?)</message>", resp.content, re.S)
        if m and m.group(1) != b"000":
            raise DartError(f"{m.group(1).decode()}: {m.group(2).decode('utf-8', 'replace')}")
    elif "html" in ctype:
        raise DartError(
            _tr(
                lang,
                "OpenDART가 HTML을 반환 (crtfc_key 누락 또는 잘못된 경로)",
                "OpenDART returned HTML (missing crtfc_key or wrong path)",
            )
        )


def _polite_wait(url: str) -> None:
    """Per-host minimum gap for hosts the shared PoliteSession does not throttle (DART raised to 0.5s)."""
    host = urlparse(url).netloc
    gap = _EXTRA_MIN_INTERVAL.get(host, 0.0)
    if not gap:
        return
    with _extra_lock:
        now = time.monotonic()
        wait = _extra_last.get(host, 0.0) + gap - now
        _extra_last[host] = max(now, _extra_last.get(host, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


# English company names (English reports only): yfinance ``Ticker(<code>.KS|.KQ).info`` longName / shortName with the
# legal-form suffixes removed ("Samsung Electronics Co., Ltd." -> "Samsung Electronics"), cached 30 days. A response
# without any usable name is cached as a miss for a day; a failed lookup is not cached (retried on the next run).
EN_NAME_NS = "kr_en_name"
EN_NAME_TTL_HOURS = 30 * 24.0
EN_NAME_MISS_TTL_HOURS = 24.0
_EN_LEGAL_SUFFIX = re.compile(
    r"(?:^|[\s,]+)(?:co|ltd|limited|inc|incorporated|corp|corporation|company|plc)\.?\s*$",
    re.IGNORECASE,
)
_HANGUL = re.compile(r"[\u3131-\u318e\uac00-\ud7a3]")


def clean_english_name(raw: Any, code: str = "") -> str | None:
    """Display-ready English company name, or None when ``raw`` is not a usable English name.

    Legal-form suffixes are removed repeatedly (", Ltd.", " Co.,Ltd", " Inc.", " Corporation", " Company" ...), HTML
    entities are decoded and whitespace collapsed. Rejected: empty / non-string values, names containing Hangul,
    names without a Latin letter and bare symbols ("005930.KS", the code itself).
    """
    if not isinstance(raw, str):
        return None
    name = " ".join(html_lib.unescape(raw).split())
    prev = None
    while prev != name:
        prev = name
        name = _EN_LEGAL_SUFFIX.sub("", name).rstrip(" ,.")
    if not name or _HANGUL.search(name) or not re.search(r"[A-Za-z]", name):
        return None
    if re.fullmatch(r"[0-9A-Z]{6}(?:\.K[SQ])?", name.upper()) or (code and name.upper() == code.upper()):
        return None  # a bare symbol echoed back, not a name
    return name


def _yf() -> Any:
    import yfinance as yf

    try:
        yf.config.debug.hide_exceptions = False  # surface Yahoo errors instead of empty frames
    except Exception:  # noqa: BLE001 - older/newer yfinance without the config object
        pass
    return yf


# ================================================================================= provider


class KRProvider(MarketProvider):
    """Korean market data provider. See the module docstring for the source chains.

    Args:
        settings, cache: as for every :class:`MarketProvider`.
        session_factory: creates the HTTP session (default :class:`PoliteSession`); one per thread.
        now_fn: returns the current tz-aware time (tests inject a fixed KST clock).
    """

    market = Market.KR

    def __init__(
        self,
        settings: Settings,
        cache: DiskCache,
        session_factory: Callable[[], requests.Session] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ):
        super().__init__(settings, cache)
        self._session_factory = session_factory or PoliteSession
        self._now_fn = now_fn or (lambda: datetime.now(KST))
        self._local = threading.local()
        self._notes: dict[str, list[str]] = {}
        self._notes_lock = threading.Lock()

    # ------------------------------------------------------------------ infrastructure

    def _now(self) -> datetime:
        return self._now_fn().astimezone(KST)

    def _http(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = self._session_factory()
            self._local.session = s
        return s

    def _get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        _polite_wait(url)
        kw: dict[str, Any] = {"params": params, "headers": headers}
        if timeout:
            kw["timeout"] = timeout
        r = self._http().get(url, **kw)
        r.raise_for_status()
        return r

    def _get_json(self, url: str, **kw: Any) -> Any:
        r = self._get(url, **kw)
        try:
            return r.json()
        except ValueError as exc:
            raise ValueError(
                _tr(
                    self.settings,
                    f"{url}: JSON 응답 아님 ({r.text[:80]!r})",
                    f"{url}: not a JSON response ({r.text[:80]!r})",
                )
            ) from exc

    def _post(self, url: str, *, data: dict, headers: dict | None = None) -> requests.Response:
        _polite_wait(url)
        r = self._http().post(url, data=data, headers=headers, timeout=30)
        r.raise_for_status()
        return r

    def _cached(
        self,
        namespace: str,
        key: str,
        ttl_hours: float | None,
        fetch: Callable[[], T],
    ) -> T:
        hit = self.cache.get(namespace, key, ttl_hours)
        if hit is not None:
            return hit
        value = fetch()
        if value is not None:
            try:
                self.cache.set(namespace, key, value)
            except Exception as exc:  # noqa: BLE001 - a cache write race must not break the fetch
                log.warning("cache write %s/%s failed: %s", namespace, key, exc)
        return value

    def english_name(self, inst: Instrument) -> str | None:
        """English company name of ``inst`` (see EN_NAME_NS) or None. Never raises: any failure is a DEBUG log
        and None, so the caller keeps the Korean name without warning noise."""
        sym = inst.yf_symbol
        if not sym:
            return None
        key = DiskCache.key("en_name", sym)
        try:
            hit = self.cache.get(EN_NAME_NS, key, EN_NAME_TTL_HOURS)
        except Exception as exc:  # noqa: BLE001 - an unreadable cache entry is refetched
            log.debug("KR %s English name cache unreadable: %s", inst.ticker, exc)
            hit = None
        if isinstance(hit, dict):
            if hit.get("name"):
                return str(hit["name"])
            if time.time() - float(hit.get("at", 0.0)) < EN_NAME_MISS_TTL_HOURS * 3600:
                return None
        try:
            info = _yf().Ticker(sym).info or {}
        except Exception as exc:  # noqa: BLE001 - optional display enrichment; the Korean name stays
            log.debug("KR %s English name lookup failed (%s): %s", inst.ticker, sym, exc)
            return None
        name = None
        for field_name in ("longName", "shortName"):
            name = clean_english_name(info.get(field_name), inst.ticker)
            if name:
                break
        if name is None:
            log.debug("KR %s: no English name in the yfinance info of %s", inst.ticker, sym)
        try:
            self.cache.set(EN_NAME_NS, key, {"name": name, "at": time.time()})
        except Exception as exc:  # noqa: BLE001 - a cache write failure only costs a refetch
            log.debug("KR %s English name cache write failed: %s", inst.ticker, exc)
        return name

    def _warn(self, ticker: str, message: str) -> None:
        log.warning("KR %s: %s", ticker, message)
        with self._notes_lock:
            lst = self._notes.setdefault(ticker, [])
            if message not in lst:
                lst.append(message)

    def data_warnings(self, ticker: str) -> list[str]:
        """Korean, report-ready notes on degraded/fallback data for ``ticker`` (non-destructive).

        A real ticker gets its own notes followed by provider-wide ones (``"__fx__"``: the FX source
        switch used to convert it). Internal keys (``"__universe__"``, ``"__fx__"``) return only their
        own notes. Everything here was also logged at WARNING. Frame notes are also in the frame's
        ``attrs["warnings"]`` with identical text (benchmark notes as ``"벤치마크: " + attrs text``), so
        the shared loader's exact-match merge keeps one copy.
        """
        if ticker.startswith("__"):
            with self._notes_lock:
                return list(self._notes.get(ticker, []))
        key = check_code(ticker, lang=lang_of(self.settings))
        with self._notes_lock:
            own = list(self._notes.get(key, []))
            shared = list(self._notes.get("__fx__", []))
        return own + [n for n in shared if n not in own]

    def _frame_out(
        self,
        ticker: str,
        df: pd.DataFrame,
        meta: dict[str, Any],
        note_prefix: str = "",
    ) -> pd.DataFrame:
        out = df.copy()
        out.attrs["source"] = meta.get("source", "")
        out.attrs["warnings"] = list(meta.get("warnings", []))
        for k in ("close_source", "volume_unit", "split_adjusted", "dividend_adjusted"):
            if k in meta:
                out.attrs[k] = meta[k]
        for w in meta.get("warnings", []):
            self._warn(ticker, f"{note_prefix}{w}")
        return out

    def _epoch(self) -> str:
        now = self._now()
        return f"{now.date().isoformat()}:{session_phase(now)}"

    # ------------------------------------------------------------------ resolve

    def resolve(self, ticker: str) -> Instrument:
        code = check_code(normalise_ticker(ticker), lang=lang_of(self.settings))
        inst, notes = self._cached(
            "kr_resolve",
            DiskCache.key("resolve", code),
            24.0,
            lambda: self._resolve_uncached(code),
        )
        for n in notes:
            self._warn(code, n)
        return inst

    def _resolve_uncached(self, code: str) -> tuple[Instrument, list[str]]:
        errors: list[str] = []
        notes: list[str] = []
        info: dict[str, Any] | None = None
        daum: dict[str, Any] | None = None
        try:
            info = parse_naver_basic(self._get_json(f"{NAVER_API}/stock/{code}/basic"), lang=lang_of(self.settings))
        except Exception as exc:  # noqa: BLE001
            errors.append(
                _tr(self.settings, "네이버 basic", "Naver basic") + f": {type(exc).__name__}: {str(exc)[:80]}"
            )
        if info is None:
            try:
                daum = self._daum_quote(code)
                if daum.get("name") and daum.get("market") in (
                    "KOSPI",
                    "KOSDAQ",
                    "KONEX",
                ):
                    info = {
                        "name": daum["name"],
                        "exchange": daum["market"],
                        "end_type": None,
                    }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Daum quote: {type(exc).__name__}: {str(exc)[:80]}")
        if info is None:
            try:
                row = self._kind_listing().set_index("code").loc[code]
                info = {
                    "name": str(row["name"]),
                    "exchange": str(row["market"]),
                    "end_type": None,
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    _tr(self.settings, "KIND 목록", "KIND listing") + f": {type(exc).__name__}: {str(exc)[:80]}"
                )
        if info is None:
            raise ValueError(
                _tr(self.settings, f"KR 종목 {code} 식별 실패: ", f"Could not identify KR stock {code}: ")
                + "; ".join(errors)
            )
        if errors:
            notes.append(
                _tr(
                    self.settings,
                    "종목 식별 1차 소스(네이버) 실패 → 대체 소스 사용: ",
                    "Stock identification: primary source (Naver) failed → using a fallback source: ",
                )
                + "; ".join(errors)
            )

        sector = self._sector(code, daum, notes)
        exchange = info["exchange"]
        if exchange == "KONEX":
            notes.append(
                _tr(self.settings, "코넥스 종목: 벤치마크를 KOSDAQ으로 대체", "KONEX stock: benchmark set to KOSDAQ")
            )
        bench = "KOSDAQ" if exchange in ("KOSDAQ", "KONEX") else "KOSPI"
        yf_symbol = f"{code}.KQ" if exchange == "KOSDAQ" else (f"{code}.KS" if exchange == "KOSPI" else None)
        inst = Instrument(
            ticker=code,
            market=Market.KR,
            name=info["name"],
            exchange=exchange,
            currency="KRW",
            sector=sector,
            yf_symbol=yf_symbol,
            benchmark=bench,
            benchmark_name=bench,
        )
        return inst, notes

    def _sector(self, code: str, daum: dict[str, Any] | None, notes: list[str]) -> str | None:
        """Naver industry group (WICS-style) -> Daum WICS sector -> cached KIND KSIC industry."""
        errors = []
        try:
            ind = parse_naver_integration(self._integration(code), lang=lang_of(self.settings)).get("industry_code")
            if ind:
                name = self._industry_name(str(ind))
                if name:
                    return name
        except Exception as exc:  # noqa: BLE001
            errors.append(_tr(self.settings, "네이버 업종", "Naver industry") + f": {type(exc).__name__}")
        try:
            q = daum if daum is not None else self._daum_quote(code)
            if q.get("wicsSectorName"):
                return str(q["wicsSectorName"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Daum WICS: {type(exc).__name__}")
        listing = self.cache.get("kr_listing", DiskCache.key("kind", self._now().date().isoformat()), 24.0)
        if listing is not None and code in set(listing["code"]):
            ind = listing.set_index("code").at[code, "industry"]
            notes.append(
                _tr(
                    self.settings,
                    "업종: WICS 조회 실패 → KIND KSIC 업종 사용",
                    "Sector: WICS lookup failed → using KIND KSIC industry",
                )
            )
            return str(ind)
        notes.append(
            _tr(self.settings, "업종 정보 수집 실패: ", "Sector information unavailable: ")
            + ", ".join(errors or [_tr(self.settings, "데이터 없음", "no data")])
        )
        return None

    def _integration(self, code: str) -> dict[str, Any]:
        key = DiskCache.key("integration", code, self._now().date().isoformat())
        return self._cached(
            "kr_integration",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._get_json(f"{NAVER_API}/stock/{code}/integration"),
        )

    def _industry_name(self, industry_code: str) -> str | None:
        def fetch() -> str | None:
            j = self._get_json(
                f"{NAVER_API}/stocks/industry/{industry_code}",
                params={"page": 1, "pageSize": 1},
            )
            return ((j or {}).get("groupInfo") or {}).get("name")

        return self._cached("kr_industry", DiskCache.key("industry", industry_code), 24.0 * 7, fetch)

    def _daum_headers(self, code: str) -> dict[str, str]:
        return {"Referer": f"https://finance.daum.net/quotes/A{code}"}

    def _daum_quote(self, code: str) -> dict[str, Any]:
        """Daum quote: ``regularTradePrice`` = KRX official close of ``date``; ``tradePrice`` = integrated."""
        return self._get_json(
            f"{DAUM_API}/quotes/A{code}",
            params={"summary": "false", "changeStatistics": "true"},
            headers=self._daum_headers(code),
        )

    def _kind_listing(self) -> pd.DataFrame:
        """KIND corpList (issuers only; KOSPI/KOSDAQ/KONEX) with KSIC industry, de-duplicated, cached daily."""

        def fetch() -> pd.DataFrame:
            frames = []
            for mkt, label in [
                ("stockMkt", "KOSPI"),
                ("kosdaqMkt", "KOSDAQ"),
                ("konexMkt", "KONEX"),
            ]:
                r = self._get(
                    KIND_LIST,
                    params={
                        "method": "download",
                        "searchType": "13",
                        "marketType": mkt,
                    },
                    timeout=30,
                )
                t = pd.read_html(
                    StringIO(r.content.decode("euc-kr", errors="replace")),
                    converters={"종목코드": str},
                )[0]
                t["market"] = label
                frames.append(t)
            df = pd.concat(frames, ignore_index=True).rename(
                columns={"회사명": "name", "종목코드": "code", "업종": "industry"}
            )
            df["code"] = df["code"].astype(str).str.zfill(6)
            # KIND lists ~43 issuers twice (old + new region name)
            return (
                df[["code", "name", "market", "industry"]].drop_duplicates("code", keep="last").reset_index(drop=True)
            )

        return self._cached(
            "kr_listing",
            DiskCache.key("kind", self._now().date().isoformat()),
            24.0,
            fetch,
        )

    # ------------------------------------------------------------------ prices

    def prices(self, inst: Instrument, start: date) -> pd.DataFrame:
        code = check_code(inst.ticker, lang=lang_of(self.settings))
        key = DiskCache.key("prices", code, start.isoformat(), self._epoch())
        df, meta = self._cached(
            "kr_prices",
            key,
            self.settings.data.price_ttl_hours,
            lambda: self._prices_uncached(inst, start),
        )
        return self._frame_out(code, df, meta)

    def _prices_uncached(self, inst: Instrument, start: date) -> tuple[pd.DataFrame, dict[str, Any]]:
        code = inst.ticker
        now = self._now()
        try:
            df, meta = self._naver_prices(code, inst.yf_symbol, start, now)
        except Exception as exc:  # noqa: BLE001 - fall back to Yahoo, never splice
            naver_err = f"{type(exc).__name__}: {str(exc)[:120]}"
            log.warning("KR %s naver prices failed (%s); trying yfinance", code, naver_err)
            if not inst.yf_symbol:
                raise RuntimeError(
                    _tr(
                        self.settings,
                        f"{code}: 네이버 시세 실패({naver_err}), yfinance 심볼 없음",
                        f"{code}: Naver prices failed ({naver_err}), no yfinance symbol",
                    )
                ) from exc
            try:
                df = self._yahoo_prices(inst.yf_symbol, start, now)
            except Exception as exc2:  # noqa: BLE001
                y_err = f"{type(exc2).__name__}: {str(exc2)[:120]}"
                raise RuntimeError(
                    _tr(
                        self.settings,
                        f"{code}: 시세 수집 실패 - 네이버({naver_err}), yfinance({y_err})",
                        f"{code}: price fetch failed - Naver ({naver_err}), yfinance ({y_err})",
                    )
                ) from exc2
            meta = {
                "source": f"yfinance {inst.yf_symbol} (배당조정)",
                "close_source": {"yahoo_adj": int(len(df))},
                "split_adjusted": True,
                "dividend_adjusted": True,
                "warnings": [
                    _tr(
                        self.settings,
                        f"네이버 시세 실패({naver_err}) → yfinance {inst.yf_symbol} 배당조정 시계열로 대체"
                        " (소스 혼합 없음, 최근 봉은 KRX 종가)",
                        f"Naver prices failed ({naver_err}) → replaced by the yfinance {inst.yf_symbol} "
                        "dividend-adjusted series (no source mixing; latest bar is the KRX close)",
                    )
                ],
            }
        if df.empty:
            raise RuntimeError(
                _tr(self.settings, f"{code}: {start} 이후 시세 없음", f"{code}: no prices since {start}")
            )
        self._staleness_check(df, now, meta)
        return df, meta

    def _fchart(self, symbol: str, start: date, now: datetime) -> tuple[pd.DataFrame, list[str]]:
        """Full fchart history needed for ``start`` (capped at 3000 bars) + truncation warning."""
        bdays = int(np.busday_count(start.isoformat(), (now.date() + timedelta(days=1)).isoformat()))
        count = int(min(FCHART_MAX_BARS, max(60, bdays + 10)))
        r = self._get(
            NAVER_FCHART,
            params={
                "timeframe": "day",
                "count": count,
                "requestType": 0,
                "symbol": symbol,
            },
        )
        df = parse_fchart(r.text, symbol, lang=lang_of(self.settings))
        warns = []
        if len(df) >= FCHART_MAX_BARS and df.index[0] > pd.Timestamp(start):
            warns.append(
                _tr(
                    self.settings,
                    f"네이버 fchart 3000봉 한도: 요청 시작일 {start} → 실제 시작 {df.index[0].date()}",
                    f"Naver fchart 3000-bar limit: requested start {start} → actual start {df.index[0].date()}",
                )
            )
        return df, warns

    def _naver_prices(
        self, code: str, yf_symbol: str | None, start: date, now: datetime
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        raw, warns = self._fchart(code, start, now)
        today = pd.Timestamp(now.date())
        if session_phase(now) == "intraday":
            raw = drop_partial_bar(raw, now)
        if raw.empty:
            raise ValueError(
                _tr(self.settings, f"네이버 fchart {code}: 완성된 봉 없음", f"Naver fchart {code}: no completed bars")
            )

        n_window = int((raw.index >= NXT_SINCE).sum())
        official: pd.Series | None = None
        official_src = "krx_official(daum)"
        last_close: tuple[pd.Timestamp, float, str] | None = None
        if n_window:
            lookback = min(1999, max(30, n_window + 5))
            try:
                payload = self._get_json(
                    f"{DAUM_API}/charts/A{code}/days",
                    params={"limit": lookback + 1, "adjusted": "true"},
                    headers=self._daum_headers(code),
                )
                official = parse_daum_official_closes(payload)
            except Exception as exc:  # noqa: BLE001
                warns.append(
                    _tr(
                        self.settings,
                        f"Daum 공식종가 조회 실패({type(exc).__name__}) → yfinance 종가로 보정 시도",
                        f"Daum official close lookup failed ({type(exc).__name__}) → trying a yfinance close "
                        "correction",
                    )
                )
                first_nxt_bar = raw.index[raw.index >= NXT_SINCE][0]  # calendar start of the whole window
                official = self._yahoo_closes(yf_symbol, first_nxt_bar - pd.Timedelta(7, "D"))
                official_src = "krx_official(yahoo)"
            last = raw.index[-1]
            if official is None or last not in official.index:
                # the latest completed bar has no successor candle -> Daum quote regularTradePrice
                try:
                    q = self._daum_quote(code)
                    px = parse_number(q.get("regularTradePrice"))
                    qd = q.get("date") or q.get("tradeDate")
                    if px and qd:
                        last_close = (
                            pd.Timestamp(str(qd)).normalize(),
                            px,
                            "krx_official(daum_quote)",
                        )
                except Exception as exc:  # noqa: BLE001 - an unconfirmed last close is reported below
                    log.warning("KR %s daum quote failed: %s", code, exc)
                if last_close is None or last_close[0] != last:
                    yc = self._yahoo_closes(yf_symbol, last - pd.Timedelta(10, "D"))
                    if yc is not None and last in yc.index:
                        last_close = (last, float(yc.loc[last]), "krx_official(yahoo)")

        fixed, fmeta = apply_official_close_fix(raw, official, last_close, since=NXT_SINCE, official_src=official_src)
        if fixed.index[-1] == today:
            confirmed = fixed.index[-1].date().isoformat() not in fmeta["unverified"]
            before = len(fixed)
            fixed = drop_partial_bar(fixed, now, confirmed_today=confirmed)
            if len(fixed) < before:
                # 16:00-20:00 and neither the Daum quote nor Yahoo confirmed today's KRX close: degraded
                # (one session stale until 20:00), so it is reported, not just logged
                warns.append(
                    _tr(
                        self.settings,
                        f"당일({today.date()}) KRX 공식종가 미확인(Daum·Yahoo 실패) → 당일 봉 제외, "
                        f"{fixed.index[-1].date()} 종가 기준 (20:00 이후 재조회 필요)",
                        f"Today's ({today.date()}) KRX official close unconfirmed (Daum · Yahoo failed) → today's bar "
                        f"excluded, based on the {fixed.index[-1].date()} close (re-fetch after 20:00)",
                    )
                )
                fmeta["unverified"] = [d for d in fmeta["unverified"] if d != today.date().isoformat()]
                fmeta["rejected"] = [d for d in fmeta["rejected"] if d != today.date().isoformat()]
        out = fixed.loc[pd.Timestamp(start) :]
        if fmeta["rejected"]:
            warns.append(
                _tr(
                    self.settings,
                    f"공식종가 후보가 당일 고저 범위 밖 → 미적용 {len(fmeta['rejected'])}일 "
                    f"({', '.join(fmeta['rejected'][-3:])}; 수정주가 기준 불일치 가능)",
                    f"Official close candidate outside the day's high-low range → not applied on "
                    f"{len(fmeta['rejected'])} day(s) ({', '.join(fmeta['rejected'][-3:])}; "
                    "adjusted-price basis may differ)",
                )
            )
        unver = [d for d in fmeta["unverified"] if pd.Timestamp(d) >= pd.Timestamp(start)]
        if unver:
            warns.append(
                _tr(
                    self.settings,
                    f"KRX 공식종가 미확인 {len(unver)}일 → NXT 통합(20:00) 종가 사용: {', '.join(unver[-5:])}",
                    f"KRX official close unconfirmed on {len(unver)} day(s) → using the NXT-integrated (20:00) "
                    f"close: {', '.join(unver[-5:])}",
                )
            )
        n_fixed = sum(fmeta["fixed"].values())
        close_source = {"naver": int(len(out)) - n_fixed, **fmeta["fixed"]}
        meta = {
            "source": "Naver fchart + KRX 공식종가 보정",
            "close_source": close_source,
            # Naver: split/rights adjusted, NOT dividend adjusted (the documented primary KR source).
            # attrs["adjusted"] is deliberately NOT set: the shared loader reads adjusted=False as a
            # degraded fallback and would warn on every KR ticker.
            "split_adjusted": True,
            "dividend_adjusted": False,
            "warnings": warns,
        }
        return out, meta

    def _yahoo_closes(self, yf_symbol: str | None, start: pd.Timestamp) -> pd.Series | None:
        """Raw (split-adjusted, not dividend-adjusted) Yahoo closes for the official-close cross-check.
        Only used through the [low, high] guard of :func:`apply_official_close_fix` (Yahoo has stale closes)."""
        if not yf_symbol:
            return None
        try:
            h = _yf().Ticker(yf_symbol).history(start=pd.Timestamp(start).date().isoformat(), auto_adjust=False)
            if h is None or h.empty or "Close" not in h:
                return None
            s = h["Close"].astype(float)
            s.index = _naive_daily_index(s.index)
            return s[~s.index.duplicated(keep="last")]
        except Exception as exc:  # noqa: BLE001 - cross-check only; absence shows up as "unverified"
            log.warning("KR %s yahoo close cross-check failed: %s", yf_symbol, exc)
            return None

    def _yahoo_prices(self, symbol: str, start: date, now: datetime) -> pd.DataFrame:
        h = _yf().Ticker(symbol).history(start=start.isoformat(), auto_adjust=False)
        if h is None or h.empty:
            raise ValueError(_tr(self.settings, f"yfinance {symbol}: 빈 시계열", f"yfinance {symbol}: empty series"))
        h.index = _naive_daily_index(h.index)
        h = h[~h.index.duplicated(keep="last")].sort_index()
        df = yahoo_adjusted_ohlcv(h, lang=lang_of(self.settings))
        df = drop_partial_bar(df, now, confirmed_today=True)
        return df.loc[pd.Timestamp(start) :]

    def _staleness_check(self, df: pd.DataFrame, now: datetime, meta: dict[str, Any]) -> None:
        age = (now.date() - df.index[-1].date()).days
        if age > 10:
            meta.setdefault("warnings", []).append(
                _tr(
                    self.settings,
                    f"최신 시세가 {df.index[-1].date()} ({age}일 전) — 데이터 지연 가능",
                    f"Latest price is from {df.index[-1].date()} ({age} days ago) — data may be delayed",
                )
            )

    # ------------------------------------------------------------------ benchmark

    def benchmark(self, inst: Instrument, start: date) -> pd.DataFrame:
        bench = inst.benchmark if inst.benchmark in BENCHMARKS else "KOSPI"
        key = DiskCache.key("benchmark", bench, start.isoformat(), self._epoch())
        df, meta = self._cached(
            "kr_benchmark",
            key,
            self.settings.data.price_ttl_hours,
            lambda: self._benchmark_uncached(bench, start),
        )
        if inst.benchmark not in BENCHMARKS:
            meta = dict(
                meta,
                warnings=[
                    *meta.get("warnings", []),
                    _tr(
                        self.settings,
                        f"알 수 없는 벤치마크 {inst.benchmark!r} → KOSPI 사용",
                        f"Unknown benchmark {inst.benchmark!r} → using KOSPI",
                    ),
                ],
            )
        # the loader prefixes benchmark-frame notes with "벤치마크: " -> record the same text
        return self._frame_out(inst.ticker, df, meta, note_prefix=_tr(self.settings, "벤치마크: ", "Benchmark: "))

    def _benchmark_uncached(self, bench: str, start: date) -> tuple[pd.DataFrame, dict[str, Any]]:
        naver_sym, yf_sym = BENCHMARKS[bench]
        now = self._now()
        try:
            raw, warns = self._fchart(naver_sym, start, now)
            df = drop_partial_bar(raw, now, confirmed_today=True).loc[pd.Timestamp(start) :]
            if df.empty:
                raise ValueError(
                    _tr(self.settings, f"{bench}: {start} 이후 데이터 없음", f"{bench}: no data since {start}")
                )
            meta: dict[str, Any] = {
                "source": f"Naver fchart {naver_sym}",
                "volume_unit": "1,000주",
                "warnings": warns,
            }
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {str(exc)[:120]}"
            log.warning(
                "KR benchmark %s naver failed (%s); trying yfinance %s",
                bench,
                err,
                yf_sym,
            )
            try:
                h = _yf().Ticker(yf_sym).history(start=start.isoformat(), auto_adjust=False)
                if h is None or h.empty:
                    raise ValueError(
                        _tr(self.settings, f"yfinance {yf_sym}: 빈 시계열", f"yfinance {yf_sym}: empty series")
                    )
                h.index = _naive_daily_index(h.index)
                h = h[~h.index.duplicated(keep="last")].sort_index()
                df = pd.DataFrame(
                    {c: h[c.title()].astype(float) for c in PRICE_COLUMNS},
                    index=h.index,
                )
                df = drop_partial_bar(df[df["close"] > 0], now, confirmed_today=True)
                if df.empty:
                    raise ValueError(
                        _tr(self.settings, f"yfinance {yf_sym}: 유효 종가 없음", f"yfinance {yf_sym}: no valid closes")
                    )
            except Exception as exc2:  # noqa: BLE001
                y_err = f"{type(exc2).__name__}: {str(exc2)[:120]}"
                raise RuntimeError(
                    _tr(
                        self.settings,
                        f"{bench} 지수 수집 실패 - 네이버({err}), yfinance({y_err})",
                        f"{bench} index fetch failed - Naver ({err}), yfinance ({y_err})",
                    )
                ) from exc2
            meta = {
                "source": f"yfinance {yf_sym}",
                "volume_unit": "주",
                "warnings": [
                    _tr(
                        self.settings,
                        f"네이버 {bench} 지수 실패({err}) → yfinance {yf_sym} 사용 (일부 결측일 존재)",
                        f"Naver {bench} index failed ({err}) → using yfinance {yf_sym} (some days missing)",
                    )
                ],
            }
        self._staleness_check(df, now, meta)
        return df, meta

    # ------------------------------------------------------------------ fundamentals

    def fundamentals(self, inst: Instrument) -> Fundamentals:
        code = check_code(inst.ticker, lang=lang_of(self.settings))
        # v2: Fundamentals gained next_earnings_*; a v1 pickle would lack those attributes
        key = DiskCache.key("fundamentals", 2, code, self._now().date().isoformat())
        fund, notes = self._cached(
            "kr_fundamentals",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._fundamentals_uncached(inst),
        )
        for n in notes:
            self._warn(code, n)
        return fund

    def _fundamentals_uncached(self, inst: Instrument) -> tuple[Fundamentals, list[str]]:
        code = inst.ticker
        errors: list[str] = []
        lang = lang_of(self.settings)

        def attempt(label: str, fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except Exception as exc:  # noqa: BLE001 - each source is optional; recorded below
                log.warning("KR %s fundamentals %s failed: %s", code, label, exc)
                errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:80]}")
                return None

        snap = attempt("naver", lambda: parse_naver_integration(self._integration(code), lang=lang))
        wise = attempt(
            "wisereport",
            lambda: parse_wisereport_snapshot(  # issuer page (common share)
                self._get(
                    f"{WISE}/v2/company/c1010001.aspx",
                    params={"cmp_cd": common_code(code)},
                ).text,
                lang=lang,
            ),
        )
        fin_q = attempt(
            "naver_quarter",
            lambda: parse_naver_finance(self._get_json(f"{NAVER_API}/stock/{code}/finance/quarter"), lang=lang),
        )
        fin_a = attempt(
            "naver_annual",
            lambda: parse_naver_finance(self._get_json(f"{NAVER_API}/stock/{code}/finance/annual"), lang=lang),
        )
        yf_t = None
        preferred = is_preferred_code(code)
        if inst.yf_symbol and not preferred:
            try:
                info = _yf().Ticker(inst.yf_symbol).info or {}
                yf_t = {
                    "high": info.get("targetHighPrice"),
                    "low": info.get("targetLowPrice"),
                    "count": info.get("numberOfAnalystOpinions"),
                }
            except Exception as exc:  # noqa: BLE001 - optional enrichment; absence is visible in `source`
                log.info("KR %s yfinance target range unavailable: %s", code, exc)
        try:
            fund, notes = compose_fundamentals(
                snap, wise, fin_a, fin_q, yf_t, self._now().date(), preferred=preferred, lang=lang
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc}: " + "; ".join(errors)) from exc
        try:
            nxt, est, sched_notes = self._earnings_schedule(code)
            fund.next_earnings_date, fund.next_earnings_is_estimate = nxt, est
            notes = [*notes, *sched_notes]
        except Exception as exc:  # noqa: BLE001 - optional field, but an unknown schedule is reported
            log.warning("KR %s earnings schedule lookup failed: %s", code, exc)
            notes = [
                *notes,
                _tr(
                    lang,
                    f"실적발표 일정 조회 실패(DART: {type(exc).__name__}) — 다음 실적발표일 미확인 "
                    "(실적 임박 진입 제한 미적용)",
                    f"Earnings schedule lookup failed (DART: {type(exc).__name__}) — next earnings date unknown "
                    "(pre-earnings entry limit not applied)",
                ),
            ]
        return fund, notes

    def _earnings_schedule(self, code: str) -> tuple[date | None, bool | None, list[str]]:
        """Next earnings date (KST) from the issuer's recent KRX 안내공시 -> (date, is_estimate, notes).

        Sources: '결산실적공시 예고' (the announced 잠정실적 disclosure date) and '기업설명회(IR) 개최' notices
        whose purpose is an earnings release/call, filed in the last ``EARNINGS_NOTICE_LOOKBACK_DAYS``
        (OpenDART list with a key, else the keyless DART web search; documents via the viewer, cached).
        Nothing scheduled -> (None, None): the date is never extrapolated from past quarters. Samsung-style
        issuers pre-announce only the confirmed-results call, not the early-month 잠정실적. Raises when
        the disclosure list itself cannot be fetched (the caller records a note).
        """
        issuer = common_code(code)  # notices are filed by the issuer (preferred shares share them)
        today = self._now().date()
        start = today - timedelta(days=EARNINGS_NOTICE_LOOKBACK_DAYS)
        rows: list[dict[str, Any]] | None = None
        if self.settings.data.dart_api_key:
            try:
                rows = self._opendart_rows(issuer, start, today)
            except Exception as exc:  # noqa: BLE001 - same list via the keyless web search
                log.warning(
                    "KR %s OpenDART list for the earnings schedule failed (%s); using DART web",
                    code,
                    exc,
                )
        if rows is None:
            rows = self._dart_web_rows(issuer, start, today, max_pages=2)
        rows = [r for r in rows if start <= r["rcept_dt"] <= today]
        released = [r["rcept_dt"] for r in rows if classify_filing(r["report_nm"])[0] == 0]
        cands = sorted(
            (
                r
                for r in rows
                if SCHEDULE_NOTICE_RE.search(normalise_report_name(r["report_nm"]))
                and not is_attachment_only_correction(r["report_nm"])
            ),
            key=lambda r: (r["rcept_dt"], r["rcept_no"]),
            reverse=True,
        )[:EARNINGS_NOTICE_MAX_DOCS]
        events: list[dict[str, Any]] = []
        failed = unparsed = 0
        for r in cands:
            try:
                text = self._filing_text(dict(r, priority=3), "DART")  # KRX doc: web viewer
            except Exception as exc:  # noqa: BLE001 - counted and reported below
                failed += 1
                log.warning("KR %s notice %s text failed: %s", code, r["rcept_no"], exc)
                continue
            ev = parse_earnings_notice(text)
            if ev["status"] == "unparsed":
                unparsed += 1
                log.warning(
                    "KR %s notice %s (%s): layout not recognised",
                    code,
                    r["rcept_no"],
                    r["report_nm"],
                )
                continue
            events.append(dict(ev, rcept_no=r["rcept_no"]))
        nxt, est = next_earnings_from_notices(events, today, released)
        notes: list[str] = []
        if nxt is None and failed + unparsed:
            # an unreadable notice may be THE earnings notice: "nothing scheduled" is not established
            notes.append(
                _tr(
                    self.settings,
                    f"실적발표 일정 공시 {len(cands)}건 중 {failed + unparsed}건 확인 불가(본문 수집 실패 "
                    f"{failed}건, 형식 미인식 {unparsed}건) — 다음 실적발표일 미확인",
                    f"{failed + unparsed} of {len(cands)} earnings-schedule notices unreadable (text fetch failed: "
                    f"{failed}, layout not recognised: {unparsed}) — next earnings date unknown",
                )
            )
        elif failed + unparsed:
            log.warning(
                "KR %s: %d/%d schedule notices unreadable; date %s from the readable ones",
                code,
                failed + unparsed,
                len(cands),
                nxt,
            )
        if est:
            log.warning(
                "KR %s earnings notices disagree %s -> earliest %s, flagged estimate",
                code,
                sorted({(e["date"], e["rcept_no"]) for e in events if e["status"] == "earnings"}),
                nxt,
            )
        return nxt, est, notes

    # ------------------------------------------------------------------ valuation history

    def valuation_history(self, inst: Instrument, start: date) -> pd.DataFrame | None:
        """Daily point-in-time TTM PER/PBR (~4.5y keyless). Raises on fetch/parse failure (the loader turns
        that into a warning); returns None only when the company has no usable EPS/BPS knots.
        Preferred shares use the issuer's (common code) EPS/BPS knots on their own prices, the same
        definition as Naver's preferred-share PER."""
        code = check_code(inst.ticker, lang=lang_of(self.settings))
        px = self.prices(inst, start)
        issuer = common_code(code)
        key = DiskCache.key("valuation", issuer, start.isoformat(), self._now().date().isoformat())
        ann, qtr = self._cached("kr_valuation", key, 24.0, lambda: self._wisereport_cf4002(issuer))
        v = build_valuation_history(ann, qtr, px["close"])
        v = v.loc[pd.Timestamp(start) :]
        if v.empty or v.dropna(how="all").empty:
            self._warn(
                code,
                _tr(
                    self.settings,
                    "밸류에이션 히스토리 없음 (WiseReport EPS/BPS 이력 부족)",
                    "No valuation history (insufficient WiseReport EPS/BPS history)",
                ),
            )
            return None
        v.attrs["source"] = "WiseReport cF4002 (FY+90일, 분기+45일 시점 기준 TTM)"
        v.attrs["warnings"] = []
        if str(px.attrs.get("source", "")).startswith("yfinance"):
            msg = _tr(
                self.settings,
                "밸류에이션 히스토리가 yfinance 배당조정가 기준 (과거 PER/PBR 소폭 과소)",
                "Valuation history based on yfinance dividend-adjusted prices (past PER/PBR slightly understated)",
            )
            v.attrs["warnings"].append(msg)
            self._warn(code, msg)
        return v

    def _wisereport_cf4002(self, code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        page = self._get(f"{WISE}/v2/company/c1040001.aspx", params={"cmp_cd": code}).text
        enc = parse_encparam(page, lang=lang_of(self.settings))
        ref = {"Referer": f"{WISE}/v2/company/c1040001.aspx?cmp_cd={code}"}
        out = []
        for quarterly in (False, True):
            j = self._get_json(
                f"{WISE}/v2/company/cF4002.aspx",
                headers=ref,
                params={
                    "cmp_cd": code,
                    "frq": "1" if quarterly else "0",
                    "rpt": 5,
                    "finGubun": "MAIN",
                    "frqTyp": 1 if quarterly else 0,
                    "cn": "",
                    "encparam": enc,
                },
            )
            out.append(parse_wisereport_cf4002(j, lang=lang_of(self.settings)))
        return out[0], out[1]

    # ------------------------------------------------------------------ news

    def news(self, inst: Instrument, since: datetime, limit: int) -> list[NewsItem]:
        code = check_code(inst.ticker, lang=lang_of(self.settings))
        since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
        key = DiskCache.key("news", code, since_utc.strftime("%Y-%m-%d"), limit)
        items, notes = self._cached(
            "kr_news",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._news_uncached(inst, since_utc, limit),
        )
        for n in notes:
            self._warn(code, n)
        return items

    def _news_uncached(
        self,
        inst: Instrument,
        since: datetime,
        limit: int,
        max_pages: int = 6,
        n_bodies: int = 5,
    ) -> tuple[list[NewsItem], list[str]]:
        code, name = inst.ticker, inst.name
        notes: list[str] = []
        errors: list[str] = []
        rows: list[dict[str, Any]] = []
        try:
            for page in range(1, max_pages + 1):
                clusters = self._get_json(
                    f"{NAVER_API}/news/stock/{code}",
                    params={"pageSize": 50, "page": page},
                    headers={"Referer": f"https://m.stock.naver.com/domestic/stock/{code}/news"},
                )
                if not clusters:
                    break
                part, oldest = parse_naver_news(clusters, since)
                rows += part
                if oldest is not None and oldest < since:
                    break
        except Exception as exc:  # noqa: BLE001
            errors.append(_tr(self.settings, "네이버 뉴스", "Naver News") + f": {type(exc).__name__}: {str(exc)[:80]}")
        days = int(min(30, max(1, math.ceil((self._now() - since).total_seconds() / 86400))))
        try:
            url = (
                GNEWS_RSS
                + "?"
                + urllib.parse.urlencode(
                    {
                        "q": f'"{name}" when:{days}d',
                        "hl": "ko",
                        "gl": "KR",
                        "ceid": "KR:ko",
                    }
                )
            )
            rows += parse_google_news_rss(self._get(url).content, since)
        except Exception as exc:  # noqa: BLE001
            errors.append(_tr(self.settings, "구글 뉴스", "Google News") + f": {type(exc).__name__}: {str(exc)[:80]}")
        if errors and not rows:
            raise RuntimeError(_tr(self.settings, "뉴스 수집 실패 - ", "News fetch failed - ") + "; ".join(errors))
        if errors:
            notes.append(_tr(self.settings, "뉴스 소스 일부 실패: ", "Some news sources failed: ") + "; ".join(errors))

        aliases = company_aliases(code, name)
        selected, used_generic = select_news(rows, aliases, limit)
        if used_generic:
            notes.append(
                _tr(
                    self.settings,
                    f"'{name}' 관련 기사 없음 → 일반 시황 기사 {len(selected)}건으로 대체 (관련도 낮음)",
                    f"No articles about '{name}' → replaced by {len(selected)} general market articles (low relevance)",
                )
            )

        # full body for the most relevant Naver articles (summary <= 600 chars)
        cands = sorted(
            (r for r in selected if "n.news.naver.com" in r["url"]),
            key=lambda r: (r["relevance"], r["cluster_size"], r["published_at"]),
            reverse=True,
        )[:n_bodies]
        body_fail = 0
        for r in cands:
            try:
                art = parse_naver_article(self._get(r["url"], headers={"Referer": "https://m.stock.naver.com/"}).text)
                if art["body"]:
                    r["summary"] = art["body"][:600]
            except Exception as exc:  # noqa: BLE001
                body_fail += 1
                log.warning("KR %s article body %s failed: %s", code, r["url"], exc)
        if cands and body_fail == len(cands):
            notes.append(
                _tr(
                    self.settings,
                    f"기사 본문 수집 전부 실패({body_fail}건) → 스니펫만 사용",
                    f"All article body fetches failed ({body_fail}) → snippets only",
                )
            )

        items = [
            NewsItem(
                id=text_id(r["url"] or (r["title"] + r["published_at"].isoformat())),
                title=r["title"],
                summary=r["summary"][:600],
                url=r["url"],
                source=r["source"],
                published_at=r["published_at"],
                lang="ko",
            )
            for r in selected
        ]
        return items, notes

    # ------------------------------------------------------------------ filings

    def filings(self, inst: Instrument, since: datetime, limit: int) -> list[FilingItem]:
        code = check_code(inst.ticker, lang=lang_of(self.settings))
        since_utc = since.astimezone(UTC) if since.tzinfo else since.replace(tzinfo=UTC)
        key = DiskCache.key(
            "filings",
            code,
            since_utc.strftime("%Y-%m-%d"),
            limit,
            bool(self.settings.data.dart_api_key),
            self.settings.data.max_filing_chars,
        )
        items, notes = self._cached(
            "kr_filings",
            key,
            self.settings.data.text_ttl_hours,
            lambda: self._filings_uncached(inst, since_utc, limit),
        )
        for n in notes:
            self._warn(code, n)
        return items

    def _filings_uncached(self, inst: Instrument, since: datetime, limit: int) -> tuple[list[FilingItem], list[str]]:
        code = inst.ticker
        notes: list[str] = []
        errors: list[str] = []
        start = since.astimezone(KST).date()
        end = self._now().date()
        max_chars = self.settings.data.max_filing_chars

        rows: list[dict[str, Any]] | None = None
        src = "DART"
        if self.settings.data.dart_api_key:
            try:
                rows, src = self._opendart_rows(code, start, end), "DART OpenAPI"
            except Exception as exc:  # noqa: BLE001
                notes.append(
                    _tr(
                        self.settings,
                        f"OpenDART 실패({type(exc).__name__}: {str(exc)[:80]}) → DART 웹 검색으로 대체",
                        f"OpenDART failed ({type(exc).__name__}: {str(exc)[:80]}) → falling back to DART web search",
                    )
                )
        if rows is None:
            try:
                rows = self._dart_web_rows(code, start, end, notes=notes)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"DART: {type(exc).__name__}: {str(exc)[:80]}")
        research: list[dict[str, Any]] = []
        try:
            data = self._get_json(
                f"{NAVER_API}/research/stock/{code}",
                params={"pageSize": 20, "page": 1},
                headers={"Referer": "https://m.stock.naver.com/"},
            )
            research = parse_naver_research_list(data, start)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                _tr(self.settings, "네이버 리서치", "Naver Research") + f": {type(exc).__name__}: {str(exc)[:80]}"
            )
        if rows is None and errors and not research:
            raise RuntimeError(
                _tr(self.settings, "공시·리포트 수집 실패 - ", "Filings · broker reports fetch failed - ")
                + "; ".join(errors)
            )
        if errors:
            notes.append(
                _tr(self.settings, "공시 소스 일부 실패: ", "Some filing sources failed: ") + "; ".join(errors)
            )

        ranked = select_filings(rows or [], limit)
        n_res_slots = min(len(research), max(1, limit // 3)) if limit >= 2 else 0
        n_dart = min(len(ranked), limit - n_res_slots)
        n_res = min(len(research), limit - n_dart)

        items: list[FilingItem] = []
        text_fail = 0
        for r in ranked[:n_dart]:
            text = ""
            try:
                text = self._filing_text(r, src)
            except Exception as exc:  # noqa: BLE001
                text_fail += 1
                log.warning("KR %s filing text %s failed: %s", code, r["rcept_no"], exc)
            items.append(
                FilingItem(
                    id=text_id("dart", r["rcept_no"]),
                    title=r["report_nm"],
                    form_type=r["form_type"],
                    filed_at=_kst_date_to_utc(r["rcept_dt"]),
                    url=r["url"],
                    text=truncate_head_tail(clean_text(text), max_chars),
                    is_earnings=bool(r["is_earnings"]),
                    source=src,
                )
            )
        res_fail = 0
        for row in research[:n_res]:
            try:
                detail = parse_naver_research_detail(
                    self._get_json(
                        f"{NAVER_API}/research/company/{row['research_id']}",
                        headers={"Referer": "https://m.stock.naver.com/"},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("KR %s research detail %s failed: %s", code, row["research_id"], exc)
                detail = {"summary": row.get("preview", "")}
                res_fail += 1
            items.append(research_filing(row, detail, max_chars))
        if text_fail:
            notes.append(
                _tr(
                    self.settings,
                    f"공시 본문 수집 실패 {text_fail}건 → 제목만 제공",
                    f"Filing text fetch failed for {text_fail} → titles only",
                )
            )
        if res_fail:
            notes.append(
                _tr(
                    self.settings,
                    f"증권사 리포트 상세 수집 실패 {res_fail}건 → 미리보기만 제공",
                    f"Broker report detail fetch failed for {res_fail} → previews only",
                )
            )
        items.sort(key=lambda f: (f.filed_at, f.is_earnings), reverse=True)
        return items[:limit], notes

    def _filing_text(self, row: dict[str, Any], src: str) -> str:
        rcp = row["rcept_no"]
        fetch_chars = 2 * self.settings.data.max_filing_chars
        section_re = _PERIODIC_SECTIONS if row.get("priority") == 2 else None

        def fetch() -> str:
            if src == "DART OpenAPI" and not is_krx_filing(rcp, row.get("rm", "")):
                try:
                    return self._opendart_document_text(rcp, fetch_chars)
                except Exception as exc:  # noqa: BLE001 - KRX/large docs: fall back to the web viewer
                    log.info(
                        "OpenDART document %s failed (%s); using DART web viewer",
                        rcp,
                        exc,
                    )
            return self._dart_web_document_text(rcp, fetch_chars, section_re)

        return self._cached(
            "kr_dart_doc",
            DiskCache.key("doc", rcp, fetch_chars, section_re),
            24.0 * 30,
            fetch,
        )

    def _dart_web_rows(
        self,
        code: str,
        start: date,
        end: date,
        max_pages: int = 5,
        notes: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page, n_pages = 1, 1
        while page <= min(n_pages, max_pages):
            # no "finalReport=recent": it hides originals superseded by a correction, and an attachment-only
            # correction ('[첨부정정]') carries no decision body - select_filings picks the right version.
            form = {
                "currentPage": str(page),
                "maxResults": "100",
                "maxLinks": "10",
                "sort": "date",
                "series": "desc",
                "startDate": start.strftime("%Y%m%d"),
                "endDate": end.strftime("%Y%m%d"),
                "option": "corp",
                "textCrpNm": code,
                "textCrpCik": "",
                "publicType": DART_WEB_TYPES,
            }
            r = self._post(
                f"{DART_WEB}/dsab007/detailSearch.ax",
                data=form,
                headers={
                    "Referer": f"{DART_WEB}/dsab007/main.do",
                    "X-Requested-With": "XMLHttpRequest",
                },
            )
            part, n_pages = parse_dart_search(r.text)
            if page == 1 and not part and "조회 결과가 없습니다" not in r.text:
                raise ValueError(
                    _tr(
                        self.settings,
                        "DART 검색 결과 파싱 실패 (페이지 구조 변경?)",
                        "DART search result parsing failed (page layout changed?)",
                    )
                )
            rows += part
            page += 1
        if n_pages > max_pages:
            log.warning("KR %s DART search truncated at %d/%d pages", code, max_pages, n_pages)
            if notes is not None:
                oldest = min((r["rcept_dt"] for r in rows), default=None)
                notes.append(
                    _tr(
                        self.settings,
                        f"DART 공시 목록 {max_pages}/{n_pages}페이지까지만 조회 → {oldest} 이전 공시 누락 가능",
                        f"DART filing list read only up to page {max_pages}/{n_pages} → filings before {oldest} "
                        "may be missing",
                    )
                )
        uniq = {r["rcept_no"]: r for r in rows}
        return list(uniq.values())

    def _dart_web_document_text(
        self,
        rcept_no: str,
        max_chars: int,
        section_re: str | None = None,
        max_sections: int = 12,
    ) -> str:
        main = self._get(f"{DART_WEB}/dsaf001/main.do", params={"rcpNo": rcept_no}).text
        nodes = parse_dart_viewer_nodes(main)
        if nodes and section_re:
            nodes = [n for n in nodes if n[0] and re.search(section_re, n[0])] or nodes
        if not nodes:
            raise ValueError(
                _tr(
                    self.settings, f"DART 뷰어 {rcept_no}: 문서 노드 없음", f"DART viewer {rcept_no}: no document nodes"
                )
            )
        parts: list[str] = []
        for title, dcm, ele, off, ln, dtd in nodes[:max_sections]:
            r = self._get(
                f"{DART_WEB}/report/viewer.do",
                params={
                    "rcpNo": rcept_no,
                    "dcmNo": dcm,
                    "eleId": ele,
                    "offset": off,
                    "length": ln,
                    "dtd": dtd,
                },
                headers={"Referer": f"{DART_WEB}/dsaf001/main.do?rcpNo={rcept_no}"},
            )
            txt = html_to_text(decode_dart_viewer(r.content, r.encoding))
            parts.append(f"## {title}\n{txt}" if title and len(nodes) > 1 else txt)
            if sum(map(len, parts)) > max_chars:
                break
        return "\n".join(parts)[:max_chars]

    # OpenDART (DART_API_KEY) - response shapes per the official spec; not live-verified (no key).
    def _opendart(self, path: str, **params: Any) -> requests.Response:
        r = self._get(
            f"{OPENDART}/{path}",
            params={"crtfc_key": self.settings.data.dart_api_key, **params},
            timeout=30,
        )
        _dart_status_check(r, lang=lang_of(self.settings))
        return r

    def _opendart_corp_code(self, code: str) -> str:
        mapping = self._cached(
            "kr_dart_corpcode",
            DiskCache.key("corpcode"),
            24.0 * 7,
            lambda: parse_corp_code_zip(self._opendart("corpCode.xml").content),
        )
        if code not in mapping:
            raise DartError(_tr(self.settings, f"corp_code 없음: {code}", f"no corp_code: {code}"))
        return mapping[code]

    def _opendart_rows(self, code: str, start: date, end: date) -> list[dict[str, Any]]:
        corp = self._opendart_corp_code(code)
        rows: dict[str, dict[str, Any]] = {}
        for ty in OPENDART_TYPES:
            page = 1
            while True:
                try:
                    j = self._opendart(
                        "list.json",
                        corp_code=corp,
                        bgn_de=start.strftime("%Y%m%d"),
                        end_de=end.strftime("%Y%m%d"),
                        last_reprt_at="N",
                        pblntf_ty=ty,
                        sort="date",
                        sort_mth="desc",
                        page_no=page,
                        page_count=100,
                    ).json()
                except DartError as exc:
                    if str(exc).startswith("013"):  # no data
                        break
                    raise
                for x in j.get("list") or []:
                    rcp = str(x.get("rcept_no") or "")
                    try:
                        dt = datetime.strptime(str(x.get("rcept_dt")), "%Y%m%d").date()
                    except ValueError:
                        continue
                    rows[rcp] = {
                        "rcept_dt": dt,
                        "rcept_no": rcp,
                        "corp_code": corp,
                        "report_nm": re.sub(r"\s+", " ", str(x.get("report_nm") or "")).strip(),
                        "flr_nm": x.get("flr_nm") or "",
                        "rm": x.get("rm") or "",
                        "is_krx": is_krx_filing(rcp, x.get("rm") or ""),
                        "url": f"{DART_WEB}/dsaf001/main.do?rcpNo={rcp}",
                    }
                if page >= int(j.get("total_page") or 1):
                    break
                page += 1
        return list(rows.values())

    def _opendart_document_text(self, rcept_no: str, max_chars: int) -> str:
        text = document_zip_to_text(self._opendart("document.xml", rcept_no=rcept_no).content, max_chars)
        if not text.strip():
            raise DartError(
                _tr(self.settings, f"document.xml {rcept_no}: 빈 문서", f"document.xml {rcept_no}: empty document")
            )
        return text

    # ------------------------------------------------------------------ fx

    def fx_rate(self, from_ccy: str, to_ccy: str) -> float:
        """USD<->KRW under the ONE definition shared with ``USProvider`` (``quantcio.data.us``): the
        last completed Yahoo KRW=X bar aligned to the last completed US session (never the live bar),
        through the same cache entry; the Naver/Hana Bank fallback is flagged under ``"__fx__"``.
        Unsupported pairs raise ``UnsupportedCurrencyPairError`` (a ValueError)."""
        from quantcio.data import us as _us

        return _us.usdkrw_fx(from_ccy, to_ccy, self._usdkrw, lang=lang_of(self.settings))

    def _usdkrw(self) -> float:
        from quantcio.data import us as _us

        payload = _us.cached_usdkrw(
            self.cache,
            self.settings.data.price_ttl_hours,
            lambda: self._fetch_krw_series_yf(),
            lambda: self._fetch_usdkrw_naver(),
            now=pd.Timestamp(self._now()),
            lang=lang_of(self.settings),
        )
        note = _us.usdkrw_fallback_note(payload, lang=lang_of(self.settings))
        if note:
            self._warn("__fx__", note)
        return float(payload["rate"])

    def _fetch_krw_series_yf(self) -> tuple[pd.Series, str]:
        from quantcio.data import us as _us

        return _us.fetch_krw_series_yf(_yf(), lang=lang_of(self.settings))

    def _fetch_usdkrw_naver(self) -> pd.Series:
        from quantcio.data import us as _us

        return _us.parse_naver_fx(self._get_json(_us.NAVER_FX_PRICES, params=dict(_us.NAVER_FX_PARAMS)))

    # ------------------------------------------------------------------ universe

    def universe(self, n: int) -> list[str]:
        key = DiskCache.key("universe", n, self._now().date().isoformat())
        codes, notes = self._cached("kr_universe", key, 24.0, lambda: self._universe_uncached(n))
        for msg in notes:
            self._warn("__universe__", msg)
        return list(codes)

    def _universe_uncached(self, n: int) -> tuple[list[str], list[str]]:
        notes: list[str] = []
        try:
            rows: list[dict[str, Any]] = []
            for mk in ("KOSPI", "KOSDAQ"):
                eligible = 0
                for page in range(1, 11):
                    j = self._get_json(
                        f"{NAVER_API}/stocks/marketValue/{mk}",
                        params={"page": page, "pageSize": 100},
                    )
                    part = parse_naver_market_value(j, mk)
                    rows += part
                    eligible += sum(1 for r in part if universe_exclusion_reason(r) is None)
                    if not part or eligible >= n or page * 100 >= int(j.get("totalCount") or 0):
                        break
            codes = select_universe(rows, n)
            if len(codes) < n:
                raise ValueError(
                    _tr(self.settings, f"적격 종목 {len(codes)}개 < {n}", f"eligible stocks {len(codes)} < {n}")
                )
            return codes, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(
                _tr(
                    self.settings,
                    f"네이버 시가총액 순위 실패({type(exc).__name__}: {str(exc)[:80]}) → "
                    "FinanceDataReader KRX 목록 사용",
                    f"Naver market-cap ranking failed ({type(exc).__name__}: {str(exc)[:80]}) → using the "
                    "FinanceDataReader KRX listing",
                )
            )
        import FinanceDataReader as fdr  # noqa: N813 - library's documented alias

        listing = fdr.StockListing("KRX")
        market_of = {"STK": "KOSPI", "KSQ": "KOSDAQ"}
        rows = [
            {
                "code": str(r.Code).zfill(6),
                "name": str(r.Name),
                "end_type": "stock",
                "market": market_of.get(str(r.MarketId)),
                "market_cap": float(r.Marcap),
            }
            for r in listing.itertuples()
            if str(r.MarketId) in market_of and pd.notna(r.Marcap)
        ]
        codes = select_universe(rows, n)
        if not codes:
            raise RuntimeError(
                _tr(
                    self.settings,
                    "KR 유니버스 구성 실패 (네이버·FinanceDataReader 모두 실패)",
                    "KR universe construction failed (Naver · FinanceDataReader both failed)",
                )
            )
        if len(codes) < n:
            notes.append(
                _tr(
                    self.settings,
                    f"KR 유니버스 {len(codes)}/{n}개만 구성",
                    f"KR universe: only {len(codes)}/{n} stocks",
                )
            )
        return codes, notes
