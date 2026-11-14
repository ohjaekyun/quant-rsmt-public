"""KRX collection queue (docs/MODELING.md 53.1 N amendment + 53.4 loading), started 2026-10-13. Research only.

ONE process, one request at a time: >= KRX_PAUSE (2 s) + jitter between requests, a 30 s rest every 200 requests,
back-off on HTML / empty answers, a stop (with the count saved) on 'Access ... Restricted' or an expired session.
Every step is resumable (one file per date / code) - rerun the same command after a stop.

    .venv/bin/python scripts/v19_krx.py all          # or etf_weekly, etf_pdf_q, index, short_bal, short_vol, etf_pdf_m

Steps (priority order):
  etf_weekly  all-ETF screen on the last session of every week (v19_data.fetch_krx_etf)
  etf_pdf_q   holdings (PDF) of the 30 largest domestic equity ETFs (no leveraged / inverse) at every quarter's
              last weekly date -> stock-level ETF ownership (53.1 N, amended before the KR run)
  index       KOSPI 200 / KOSDAQ 150 constituents on the last weekly date of every month
  short_bal   short-selling balances of all stocks (KOSPI, KOSDAQ) on every weekly date since 2016-06-30
  short_vol   daily short-selling volume of the universe codes since 2013 (2-year request chunks)
  etf_pdf_m   the same PDFs on the other month-ends (lowest priority)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v19_data as v  # noqa: E402

say = v13.say
PDF_TOP = 30
REST_EVERY, REST_S = 200, 30.0
SHORT_FROM = pd.Timestamp("2016-06-30")  # short-balance reporting starts
SHORT_VOL_FROM = pd.Timestamp("2013-01-01")
INDEX = {"kospi200": ("1", "028"), "kosdaq150": ("2", "203")}


class KrxBlockedError(RuntimeError):
    pass


class Krx:
    """Paced KRX reader (pykrx session from v13_data._krx_stock). A non-JSON answer is checked at once: an expired
    session ('LOGOUT') stops the run (or, when KRX_ID / KRX_PW log in by script, is renewed quietly); anything else
    backs off and retries."""

    def __init__(self):
        with contextlib.redirect_stdout(io.StringIO()):  # pykrx prints the login ID when it logs in
            v13._krx_stock()
        from pykrx.website.krx.krxio import KrxWebIo

        self._base = KrxWebIo
        self.n = 0
        self.relogins = 0

    def _relogin(self) -> bool:
        """Script login with KRX_ID / KRX_PW (works only for accounts with a site password); output suppressed."""
        from pykrx.website.comm import auth

        uid, pw = kd.key("KRX_ID"), kd.key("KRX_PW")
        if not (uid and pw) or self.relogins >= 20:
            return False
        s = auth.KRXSession()
        with contextlib.redirect_stdout(io.StringIO()):
            ok = s.refresh(uid, pw)
        if ok:
            auth.set_auth_session(s)
            self.relogins += 1
            say(f"KRX 세션 갱신(스크립트 로그인 {self.relogins}회)")
        return ok

    def read(self, bld: str, **params) -> dict:
        base = self._base

        class _Io(base):
            @property
            def bld(self):
                return bld

        for attempt in range(5):
            if self.n and self.n % REST_EVERY == 0:
                time.sleep(REST_S)
            time.sleep(v13.KRX_PAUSE + random.uniform(0.3, 1.2))
            self.n += 1
            try:
                return _Io().read(**params)
            except ValueError as exc:  # not JSON: expired session ('LOGOUT'), throttling page or a server hiccup
                if "Restricted" in str(exc) or "restricted" in str(exc):
                    raise KrxBlockedError("KRX가 접속을 일시 제한함 — 몇 시간 뒤 다시 실행") from exc
                if not self.alive():
                    if self._relogin():
                        continue
                    raise KrxBlockedError(
                        "KRX 세션 만료(LOGOUT) — 브라우저에서 다시 로그인해 .env의 KRX_JSESSIONID를 바꾼 뒤 "
                        "같은 명령을 다시 실행"
                    ) from exc
                time.sleep(20.0 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - network errors: back off and retry
                say(f"KRX {bld[-11:]}: {type(exc).__name__}, retry {attempt + 1}")
                time.sleep(20.0 * (attempt + 1))
        raise KrxBlockedError(f"KRX {bld[-11:]}: 5번 실패 — 잠시 뒤 다시 실행")

    def alive(self) -> bool:
        time.sleep(v13.KRX_PAUSE)
        self.n += 1
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                from pykrx import stock

                return v13._krx_alive(stock)
            except Exception:  # noqa: BLE001 - an error answer means the session is not usable
                return False


def isin_kr(code: str) -> str:
    """ISIN of a KRX short code (KR7 + code + '00' + Luhn check digit over the letter-expanded string)."""
    base = f"KR7{code}00"
    digits = "".join(str(int(ch, 36)) for ch in base)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return base + str((10 - total % 10) % 10)


def _weekly_dates(settings) -> pd.DatetimeIndex:
    d = v.root(settings) / "krx_etf"
    return pd.DatetimeIndex(sorted(pd.Timestamp(p.stem) for p in d.glob("*.parquet")))


def _last_per(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period(freq)).max().to_numpy())


def _stop_if_dead(k: Krx, what: str, done: int) -> None:
    if not k.alive():
        raise KrxBlockedError(
            f"KRX 세션 만료({what} {done}건 새로 저장) — KRX_JSESSIONID를 바꾼 뒤 같은 명령을 다시 실행"
        )


def etf_pdf(settings, k: Krx, months: str) -> dict:
    """PDFs of the PDF_TOP largest domestic equity ETFs (sector + market, no leveraged / inverse) per date."""
    t = v.kr_etf_weekly(settings)
    if t is None:
        return {"skipped": "no weekly ETF data"}
    dates = _last_per(_weekly_dates(settings), "Q")
    if months == "other":
        dates = _last_per(_weekly_dates(settings), "M").difference(dates)
    out = v.root(settings) / "krx_etf_pdf"
    out.mkdir(exist_ok=True)
    done, empty_run = 0, 0
    for day in dates:
        p = out / f"{day:%Y%m%d}.parquet"
        if p.exists():
            continue
        snap = t[(t["date"] == day) & t["kind"].isin(["sector", "market"])].nlargest(PDF_TOP, "aum")
        rows = []
        # ETFs on the same underlying index hold that index's weights: one PDF per index (the largest ETF's),
        # reused with each ETF's own net assets (2026-10-13, after 17 quarter-ends had been fetched in full)
        for _idx, grp in snap.groupby("IDX_IND_NM", sort=False, dropna=False):
            lead = grp.iloc[0]
            r = k.read("dbms/MDC/STAT/standard/MDCSTAT05001", trdDd=f"{day:%Y%m%d}", isuCd=isin_kr(lead["code"]))
            out_rows = r.get("output") or []
            for code, aum in zip(grp["code"], grp["aum"], strict=True):
                for x in out_rows:
                    isin = str(x.get("COMPST_ISU_CD2") or "")
                    rows.append(
                        {
                            "date": day,
                            "etf": code,
                            "etf_aum": aum,
                            "isin": isin,
                            "code": isin[3:9] if isin.startswith("KR7") else str(x.get("COMPST_ISU_CD") or ""),
                            "weight": v13._num_krx(x.get("COMPST_RTO")),
                            "value_cu": v13._num_krx(x.get("VALU_AMT")) if code == lead["code"] else np.nan,
                            "pdf_of": lead["code"],
                        }
                    )
        if not rows:
            empty_run += 1
            if empty_run >= 2:
                _stop_if_dead(k, "ETF PDF", done)
            continue
        empty_run = 0
        pd.DataFrame(rows).to_parquet(p, index=False)
        done += 1
        if done % 5 == 0:
            say(f"ETF PDF {done} dates ({k.n} requests)")
    return {"dates_new": done, "dates": len(list(out.glob("*.parquet")))}


def index_members(settings, k: Krx) -> dict:
    out = v.root(settings) / "krx_index"
    out.mkdir(exist_ok=True)
    done = 0
    for day in _last_per(_weekly_dates(settings), "M"):
        p = out / f"{day:%Y%m%d}.parquet"
        if p.exists():
            continue
        rows = []
        for name, (i1, i2) in INDEX.items():
            r = k.read("dbms/MDC/STAT/standard/MDCSTAT00601", trdDd=f"{day:%Y%m%d}", indIdx=i1, indIdx2=i2)
            rows += [
                {"date": day, "index": name, "code": x.get("ISU_SRT_CD"), "mcap": v13._num_krx(x.get("MKTCAP"))}
                for x in r.get("output") or []
            ]
        if not rows:
            _stop_if_dead(k, "지수 구성", done)
            continue
        pd.DataFrame(rows).to_parquet(p, index=False)
        done += 1
    return {"months_new": done, "months": len(list(out.glob("*.parquet")))}


def short_balances(settings, k: Krx) -> dict:
    out = v.root(settings) / "krx_short_bal"
    out.mkdir(exist_ok=True)
    done, empty_run = 0, 0
    for day in _weekly_dates(settings):
        p = out / f"{day:%Y%m%d}.parquet"
        if day < SHORT_FROM or p.exists():
            continue
        rows = []
        for mkt in (1, 2):
            r = k.read("dbms/MDC/STAT/srt/MDCSTAT30501", trdDd=f"{day:%Y%m%d}", mktTpCd=mkt)
            for x in r.get("OutBlock_1") or r.get("output") or []:
                rows.append({"date": day, "mkt": mkt, **{c: x.get(c) for c in x}})
        if not rows:
            empty_run += 1
            if empty_run >= 3:
                _stop_if_dead(k, "공매도 잔고", done)
            (out / f"{day:%Y%m%d}.empty").write_text("")
            continue
        empty_run = 0
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c not in ("date", "mkt", "ISU_CD", "ISU_ABBRV", "ISU_SRT_CD", "RPT_DUTY_OCCR_DD"):
                df[c] = df[c].map(v13._num_krx)
        df.to_parquet(p, index=False)
        done += 1
        if done % 50 == 0:
            say(f"공매도 잔고 {done} weeks ({k.n} requests)")
    return {"weeks_new": done, "weeks": len(list(out.glob("*.parquet")))}


def short_volume(settings, cache, k: Krx) -> dict:
    out = v.root(settings) / "krx_short_vol"
    out.mkdir(exist_ok=True)
    codes = sorted(v16.universe("KR", settings, cache)["entity"].astype(str))
    end = pd.Timestamp.today().normalize() - pd.Timedelta(1, "D")
    done = 0
    for i, code in enumerate(codes, 1):
        p = out / f"{code}.parquet"
        if p.exists():
            continue
        rows, lo = [], SHORT_VOL_FROM
        while lo <= end:
            hi = min(lo + pd.Timedelta(729, "D"), end)
            r = k.read(
                "dbms/MDC/STAT/srt/MDCSTAT30102", strtDd=f"{lo:%Y%m%d}", endDd=f"{hi:%Y%m%d}", isuCd=isin_kr(code)
            )
            rows += r.get("OutBlock_1") or r.get("output") or []
            lo = hi + pd.Timedelta(1, "D")
        df = pd.DataFrame(rows)
        if len(df):
            for c in df.columns:
                if c != "TRD_DD":
                    df[c] = df[c].map(v13._num_krx)
            df["date"] = pd.to_datetime(df.pop("TRD_DD"), format="%Y/%m/%d")
        df.to_parquet(p, index=False)
        done += 1
        if done % 25 == 0:
            say(f"공매도 거래 {done}/{len(codes)} codes ({k.n} requests)")
    return {"codes_new": done, "codes": len(list(out.glob("*.parquet")))}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    steps = ["etf_weekly", "etf_pdf_q", "index", "short_bal", "short_vol", "etf_pdf_m"]
    ap.add_argument("what", choices=["all", *steps])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    if not v13.krx_ready():
        raise SystemExit("KRX 로그인 정보 없음(.env)")
    k = Krx()
    if not k.alive() and not k._relogin():
        raise SystemExit("KRX 세션이 동작하지 않음(LOGOUT) — 브라우저에서 다시 로그인해 KRX_JSESSIONID를 바꿔 주세요")
    run = {
        "etf_weekly": lambda: v.fetch_krx_etf(s),
        "etf_pdf_q": lambda: etf_pdf(s, k, "quarter"),
        "index": lambda: index_members(s, k),
        "short_bal": lambda: short_balances(s, k),
        "short_vol": lambda: short_volume(s, cache, k),
        "etf_pdf_m": lambda: etf_pdf(s, k, "other"),
    }
    for name in steps:
        if a.what not in ("all", name):
            continue
        try:
            say(f"{name}: {run[name]()} ({k.n} requests so far)")
        except KrxBlockedError as exc:
            say(f"{name}: STOPPED - {exc}")
            return


if __name__ == "__main__":
    main()
