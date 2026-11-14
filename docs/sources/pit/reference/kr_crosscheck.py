"""Cross-check helpers (keyless): Daum listed-shares history, spell-aware Naver fchart."""
import re, time
from io import StringIO
import pandas as pd
import requests

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/128.0 Safari/537.36")

def daum_shares_history(code: str, per_page: int = 1000, pause: float = 0.25) -> pd.DataFrame:
    """Daily listedSharesCount (+ close/volume/value) back to 1995 for LISTED codes and codes delisted
    since ~2024-09; older delisted codes -> HTTP 400. Referer header mandatory. Close on/after
    2026-09-14 is the NXT-integrated price, not the KRX official close."""
    rows, page = [], 1
    while True:
        r = requests.get(f"https://finance.daum.net/api/quote/A{code}/days",
                         params={"symbolCode": f"A{code}", "page": page, "perPage": per_page, "pagination": "true"},
                         headers={"User-Agent": UA, "Referer": f"https://finance.daum.net/quotes/A{code}"}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Daum quote/days {code}: HTTP {r.status_code}")
        j = r.json()
        rows += j["data"]
        if page >= j["totalPages"]:
            break
        page += 1
        time.sleep(pause)
    d = pd.DataFrame(rows)
    d.index = pd.to_datetime(d["date"].str[:10])
    return d.sort_index()[["tradePrice", "accTradeVolume", "accTradePrice", "listedSharesCount"]]

def naver_fchart_spells(code: str, gap_days: int = 20) -> pd.DataFrame:
    """Naver fchart daily bars (<= 3000, split/bonus/rights-adjusted but NOT always: verify vs KRX base price)
    with a 'spell' column; Naver STITCHES relisting spells (e.g. 101970 2015->2025), never compute returns across them."""
    r = requests.get("https://fchart.stock.naver.com/sise.nhn",
                     params=dict(timeframe="day", count=3000, requestType=0, symbol=code),
                     headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    rows = re.findall(r'<item data="(.*?)"\s*/>', r.text)
    if not rows:
        raise ValueError(f"fchart: no data for {code}")
    df = pd.read_csv(StringIO("\n".join(rows)), sep="|", header=None, dtype={"d": str},
                     names=["d", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df.pop("d"), format="%Y%m%d")
    df["spell"] = (df.index.to_series().diff().dt.days > gap_days).cumsum().values
    df["ret"] = df.groupby("spell")["close"].pct_change()
    return df
