"""Korean customs exports / imports by product (docs/MODELING.md 76, pre-declared 2026-10-24). Research only.

    .venv/bin/python scripts/v32_trade_data.py fetch

관세청 품목별 국가별 수출입실적 (apis.data.go.kr/1220000/nitemtrade, DATA_GO_KR_KEY with the 활용신청): one request per
HS-4 code and calendar year (the API caps a request at 12 months and returns every HS-6 x country x month row at once).
Rows are summed per HS-4 and month; the 총계 row is used only as a check. Total customs exports / imports come from
ECOS 901Y118 (수출입 총괄). A month is published around the 15th of the next month -> available on the 16th.
Re-runs refetch only the last two calendar years and any missing (HS, year). Files under <cache>/alt/keyed/v32/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say
URL = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"
HS_SECTOR = {  # fixed before any result (76)
    "반도체": ["8542", "8541"],
    "IT·전자": ["8517", "8471", "8528", "9013"],
    "2차전지·전기장비": ["8507", "8504"],
    "자동차": ["8703", "8708"],
    "산업재": ["8901", "8429", "8710", "9301"],
    "화학": ["3901", "3902", "2902"],
    "에너지": ["2710"],
    "철강·소재": ["7208", "7210", "7306"],
    "제약·바이오": ["3002", "3004"],
    "경기소비재": ["3304"],
    "필수소비재": ["1902", "2106", "2402"],
}
HS_CODES = [h for hs in HS_SECTOR.values() for h in hs]
FIELDS = {"expDlr": "exp_usd", "expWgt": "exp_kg", "impDlr": "imp_usd", "impWgt": "imp_kg"}
FIRST_YEAR = 2000


def root(settings) -> Path:
    return kd.root(settings, "v32")


def available(date: pd.Series) -> pd.Series:
    """The 16th of the month after the reference month."""
    return pd.to_datetime(date) + pd.offsets.MonthEnd(0) + pd.Timedelta(16, "D")


def _year(k: str, hs: str, year: int, last: pd.Timestamp) -> tuple[pd.DataFrame, dict | None]:
    end = min(pd.Timestamp(year=year, month=12, day=1), last)
    params = {"serviceKey": k, "strtYymm": f"{year}01", "endYymm": f"{end:%Y%m}", "hsSgn": hs}
    for attempt in range(4):
        try:
            r = requests.get(URL, params=params, headers=kd.UA, timeout=180)
            if r.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"HTTP failure for HS {hs} {year}")
    tree = ElementTree.fromstring(r.content)
    code = tree.findtext("header/resultCode")
    if code != "00":
        raise RuntimeError(f"HS {hs} {year}: {code} {tree.findtext('header/resultMsg') or r.text[:200]}")
    rows, total = [], None
    for it in tree.iter("item"):
        rec = {v: pd.to_numeric(it.findtext(f), errors="coerce") for f, v in FIELDS.items()}
        period = it.findtext("year") or ""
        if period == "총계":
            total = rec
            continue
        rows.append({"period": period, **rec})
    t = pd.DataFrame(rows, columns=["period", *FIELDS.values()])
    if t.empty:
        return pd.DataFrame(columns=["hs", "date", *FIELDS.values()]), total
    t = t.groupby("period", as_index=False)[list(FIELDS.values())].sum()
    t["date"] = pd.to_datetime(t["period"], format="%Y.%m") + pd.offsets.MonthEnd(0)
    return t.assign(hs=hs)[["hs", "date", *FIELDS.values()]], total


def _save(parts: list[pd.DataFrame], path: Path) -> pd.DataFrame:
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    out = out.drop_duplicates(["hs", "date"], keep="last").sort_values(["hs", "date"])
    out["available"] = available(out["date"])
    out.to_parquet(path, index=False)
    return out


def fetch_customs(settings) -> dict:
    k = kd.key("DATA_GO_KR_KEY")
    if not k:
        return {"skipped": "DATA_GO_KR_KEY 없음"}
    path = root(settings) / "kr_trade.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["hs", "date", *FIELDS.values()])
    today = pd.Timestamp.today().normalize()
    last = (today - pd.offsets.MonthBegin(1)).replace(day=1)  # the latest month that can exist
    have = {(h, int(y)) for h, y in zip(old["hs"], pd.to_datetime(old["date"]).dt.year, strict=False)}
    redo = {last.year, last.year - 1}
    keep = old[~pd.to_datetime(old["date"]).dt.year.isin(redo)] if len(old) else old
    parts, checks, n = [keep], [], 0
    for hs in HS_CODES:
        for y in range(FIRST_YEAR, last.year + 1):
            if y not in redo and (hs, y) in have:
                continue
            t, total = _year(k, hs, y, last)
            n += 1
            if total is not None and len(t):
                diff = abs(t["exp_usd"].sum() - total["exp_usd"]) / max(total["exp_usd"], 1)
                if diff > 1e-6:
                    checks.append(f"{hs} {y} 합계 차이 {diff:.2%}")
            parts.append(t)
            time.sleep(0.3)
        out = _save(parts, path)  # after every code, so an interrupted run keeps what it has
        say(f"customs HS {hs}: done ({n} requests so far)")
    return {
        "requests": n,
        "rows": len(out),
        "from": str(out["date"].min().date()),
        "to": str(out["date"].max().date()),
        "total_mismatch": checks[:5],
    }


def fetch_total(settings) -> dict:
    k = kd.key("ECOS_API_KEY")
    if not k:
        return {"skipped": "ECOS_API_KEY 없음"}
    rows = []
    for item, name in (("T002", "exp_usd"), ("T004", "imp_usd")):
        url = f"{kd.ECOS}/StatisticSearch/{k}/json/kr/1/1000/901Y118/M/200001/{pd.Timestamp.today():%Y%m}/{item}"
        js = requests.get(url, timeout=120).json()
        for x in js.get("StatisticSearch", {}).get("row", []):
            rows.append({"series": name, "period": x["TIME"], "value": pd.to_numeric(x["DATA_VALUE"], errors="coerce")})
        time.sleep(0.25)
    t = pd.DataFrame(rows).dropna(subset=["value"])
    t["date"] = pd.to_datetime(t["period"], format="%Y%m") + pd.offsets.MonthEnd(0)
    t["available"] = available(t["date"])
    t.to_parquet(root(settings) / "kr_trade_total.parquet", index=False)
    return {"rows": len(t), "to": str(t["date"].max().date())}


def main() -> None:
    from quantcio.config import load_settings

    s = load_settings()
    secrets = [kd.key("DATA_GO_KR_KEY"), kd.key("ECOS_API_KEY")]
    for name, fn in (("total", fetch_total), ("customs", fetch_customs)):
        try:
            say(f"{name}: {fn(s)}")
        except Exception as exc:  # noqa: BLE001 - the other source still loads
            say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *secrets)}")


if __name__ == "__main__":
    main()
