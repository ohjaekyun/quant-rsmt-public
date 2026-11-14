"""Corrected marcap PIT-universe helpers (verifier). Changes vs the claimed code are marked FIX."""
from __future__ import annotations
import os, re, time
import numpy as np
import pandas as pd
import requests

MARCAP_URL = "https://raw.githubusercontent.com/FinanceData/marcap/master/data/marcap-{year}.parquet"
COLS = ["Date", "Code", "Name", "Market", "MarketId", "Dept", "Open", "High", "Low", "Close",
        "Changes", "ChangesRatio", "Volume", "Amount", "Marcap", "Stocks", "Rank"]
_SPAC_RE = re.compile(r"스팩|기업인수목적|SPAC", re.I)
# FIX: the original regex let ship funds (아시아N호, 동북아N호, 바다로N호, 하이골드N호, 코리아0N호), REIT/infra funds
# (맵스리얼티, KB발해인프라, 코크렙N호, 트러스제7호) and investment companies (베트남개발1, 한국패러랠, 하나니켈N호) through.
# '\d+호$' also catches SPACs without '스팩' in the name (케이프이에스제4호, 골든브릿지이안5호). No false positives 2013-2026.
_NONSTOCK_RE = re.compile(r"리츠$|^이리츠|REIT|부동산투자회사|인프라투융자|맥쿼리인프라|발해인프라|맵스리얼티|선박투자|유전"
                          r"|\d+호$|베트남개발|한국패러랠|하나니켈", re.I)
_ADMIN_DEPT = ("관리종목", "투자주의환기")

def download_marcap(years, cache_dir: str, refresh_current: bool = True, pause: float = 0.3) -> list[str]:
    """Yearly parquet files (17-25 MB each). FIX: atomic write, so an interrupted download is not cached forever."""
    os.makedirs(cache_dir, exist_ok=True)
    paths, this_year = [], pd.Timestamp.today().year
    for y in years:
        path = os.path.join(cache_dir, f"marcap-{y}.parquet")
        if not os.path.exists(path) or (refresh_current and y == this_year):
            r = requests.get(MARCAP_URL.format(year=y), timeout=300)
            r.raise_for_status()
            if not r.content.startswith(b"PAR1"):
                raise ValueError(f"marcap-{y}: not a parquet file ({r.content[:60]!r})")
            with open(path + ".tmp", "wb") as fh:
                fh.write(r.content)
            os.replace(path + ".tmp", path)
            time.sleep(pause)
        paths.append(path)
    return paths

def load_marcap(paths) -> pd.DataFrame:
    df = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    df["Date"] = df["Date"].astype("datetime64[ns]")
    df["Code"] = df["Code"].astype(str)
    for c in ("Name", "Market", "MarketId", "Dept"):
        df[c] = df[c].astype("category")
    return df.sort_values(["Code", "Date"], kind="stable").reset_index(drop=True)

def add_adjusted(df: pd.DataFrame, gap_days: int = 20) -> pd.DataFrame:
    """ret = KRX base-price return Close/(Close-Changes)-1 (price only). Adds:
    spell / sess_in_spell (FIX: session count within the listing spell, so min_sessions works on one-day slices),
    reset_after_halt, ambiguous_reset (FIX: base reset after a halt with NO share-count change since the last traded
    day - e.g. SPAC-merger or re-appraisal resets; the holder's return is the raw Close/prevClose), ret_alt (= ret,
    but raw return on ambiguous_reset rows), adj_factor / AdjClose (FIX: cumulated within (Code, spell) only)."""
    df = df.copy()
    g = df.groupby("Code", sort=False)
    prev_close, prev_vol, prev_date = g["Close"].shift(1), g["Volume"].shift(1), g["Date"].shift(1)
    base = df["Close"] - df["Changes"]
    df["ret"] = (df["Close"] / base - 1.0).where((base > 0) & prev_close.notna())
    f = (base / prev_close).where((base > 0) & (prev_close > 0))
    f = f.where((f - 1).abs() > 1e-6, 1.0).fillna(1.0)
    df["new_spell"] = prev_date.isna() | ((df["Date"] - prev_date).dt.days > gap_days)
    f = f.where(~df["new_spell"], 1.0)
    df.loc[df["new_spell"], "ret"] = np.nan
    df["reset_after_halt"] = (f != 1.0) & (prev_vol == 0)
    shares_last_traded = df["Stocks"].where(df["Volume"] > 0).groupby(df["Code"]).shift(1).groupby(df["Code"]).ffill()
    df["ambiguous_reset"] = df["reset_after_halt"] & (df["Stocks"] == shares_last_traded)
    raw = (df["Close"] / prev_close - 1.0).where(prev_close > 0)
    df["ret_alt"] = df["ret"].where(~df["ambiguous_reset"], raw)
    df["spell"] = df.groupby("Code", sort=False)["new_spell"].cumsum().astype("int16")
    df["sess_in_spell"] = df.groupby(["Code", "spell"], sort=False).cumcount().astype("int32")
    df["_f"] = f
    rev = df.iloc[::-1].groupby(["Code", "spell"], sort=False)["_f"].cumprod().iloc[::-1]
    df["adj_factor"] = (rev / df["_f"]).astype(float)
    df["AdjClose"] = df["Close"] * df["adj_factor"]   # NOT point-in-time as a level (uses later splits): returns only
    return df.drop(columns="_f")

def eligible_mask(df: pd.DataFrame, markets=("STK", "KSQ"), common_only=True, drop_spac_reit=True,
                  drop_admin=True, min_sessions: int = 0, min_price: float = 0.0,
                  exclude_codes: set | None = None, drop_foreign: bool = False) -> pd.Series:
    """PIT eligibility per row. exclude_codes: e.g. non-주권 symbols from the FDR delisting table
    (set(f.Symbol[f.SecuGroup != '주권'])). drop_foreign: drop 9xxxxx foreign shares / KDRs."""
    m = df["MarketId"].isin(markets)
    if common_only:
        m &= df["Code"].str.endswith("0")
    name = df["Name"].astype(str)
    if drop_spac_reit:
        m &= ~name.str.contains(_SPAC_RE) & ~name.str.contains(_NONSTOCK_RE)
        m &= ~df["Dept"].astype(str).str.startswith("SPAC")
    if exclude_codes:
        m &= ~df["Code"].isin(exclude_codes)
    if drop_foreign:
        m &= ~df["Code"].str.startswith("9")
    if drop_admin:  # KOSDAQ 소속부 only; KOSPI 관리종목 are NOT in marcap
        m &= ~df["Dept"].astype(str).str.startswith(_ADMIN_DEPT)
    if min_sessions:
        if "sess_in_spell" not in df:  # FIX: cumcount on a one-day slice is always 0 -> silently empty universe
            raise ValueError("min_sessions needs the sess_in_spell column from add_adjusted()")
        m &= df["sess_in_spell"] >= min_sessions
    if min_price:
        m &= df["Close"] >= min_price
    return m

def rolling_mean(df: pd.DataFrame, col: str, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing mean within (Code, spell). Needs 'spell' from add_adjusted()."""
    return (df.groupby(["Code", "spell"], sort=False)[col]
              .transform(lambda s: s.rolling(window, min_periods=min_periods or max(1, window // 2)).mean()))

def last_session(df: pd.DataFrame, date) -> pd.Timestamp:
    dates = df["Date"].drop_duplicates().sort_values()
    d = dates[dates <= pd.Timestamp(date)]
    if d.empty:
        raise ValueError(f"no KRX session on or before {date}")
    return d.iloc[-1]

def universe_on(df: pd.DataFrame, date, n: int, rank_col: str = "Marcap", **elig) -> pd.DataFrame:
    """Top-n eligible stocks at the close of the last session <= date (FIX: calendar month-ends such as
    2019-12-31 or holidays returned an empty frame). Trade from the next session."""
    t = last_session(df, date)
    d = df[df["Date"] == t]
    d = d[eligible_mask(d, **elig) & (d["Volume"] > 0)]
    if d.empty:
        raise ValueError(f"empty universe on {t.date()}")
    return d.nlargest(n, rank_col)
