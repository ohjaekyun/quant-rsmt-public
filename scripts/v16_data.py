"""Data for the v16 research runs (docs/MODELING.md 39, pre-declared 2026-10-10): block R (research papers, OpenAlex),
X (taxes and government support), J (legal sanctions) and B (credit ratings and corporate bonds), KR and US.
Research only - nothing here is served.

    .venv/bin/python scripts/v16_data.py fetch --what all   # or kr_fin, dart_bonds, ecos, fred, epa, xbrl, ...

Sources: KR = DART bulk financial statements (already cached by src/quantcio/data/pit/kr_fund.py), DART filing lists
(cached by the v3 alt-data loader), OpenDART 'detScritsIsuAcmslt' (채무증권 발행실적), ECOS daily yields; US = SEC XBRL
companyfacts, USAspending.gov, EPA ECHO enforcement case files, FRED; both = OpenAlex. Point-in-time rules: 39.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import us_v4_data as v4  # noqa: E402
import v13_data as v13  # noqa: E402
from us_v8_data import norm_name  # noqa: E402

say = v13.say
UA = {"User-Agent": "Mozilla/5.0 (research; quantcio)"}
OPENALEX = "https://api.openalex.org"
USASP = "https://api.usaspending.gov/api/v2"
DART_API = "https://opendart.fss.or.kr/api"
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
EPA_ZIP = "https://echo.epa.gov/files/echodownloads/case_downloads.zip"
MAX_AGE = 500  # days a financial-statement value stays usable
DAY = np.timedelta64(1, "D")

DOCS: dict[str, tuple[str, str]] = {
    # block R
    "p_pub_n": ("지난 연도 논문 수 log(1 + n) (OpenAlex)", "papers last calendar year, log(1 + n)"),
    "p_pub_growth": ("논문 수 증가 log((1 + n) ÷ (1 + 전년))", "paper growth"),
    "p_pub_int": ("논문 수 ÷ 시가총액", "papers / market cap"),
    # block X
    "x_etr": ("유효세율(법인세비용 ÷ 세전이익)", "effective tax rate"),
    "x_cash_etr": ("현금 세율(납부 법인세 ÷ 세전이익)", "cash effective tax rate"),
    "x_etr_chg": ("유효세율 전년 대비 변화", "ETR change"),
    "x_tax_chg": ("법인세비용 변화 ÷ 전년 자산", "tax expense change / lagged assets"),
    "x_tax_gap": ("(법인세비용 − 납부액) ÷ 자산", "(tax expense - taxes paid) / assets"),
    "x_utb": ("미인식 세무 혜택 ÷ 자산", "unrecognized tax benefits / assets"),
    "g_grant_cf": ("정부보조금 수취 ÷ 자산", "government grants received / assets"),
    "g_grant_bs": ("재무상태표 정부보조금 ÷ 자산", "government grants on the balance sheet / assets"),
    "g_fed_contract": ("연방 정부 계약(최근 4분기) ÷ 매출", "federal contracts (4 quarters) / revenue"),
    "g_fed_contract_chg": ("연방 계약 증가 log", "federal contract growth (log)"),
    "g_fed_assist": ("연방 보조금·대출·직접 지급(최근 4분기) ÷ 매출", "federal assistance (4 quarters) / revenue"),
    # block J
    "j_suit_12m": ("소송 제기 공시 수(12개월, log)", "lawsuit filings (12m)"),
    "j_ruling_12m": ("소송 판결·결정 공시 수(12개월, log)", "lawsuit rulings (12m)"),
    "j_fine_24m": ("벌금 등 부과 공시(24개월, log)", "fines (24m)"),
    "j_embezzle_24m": ("횡령·배임 공시(24개월, log)", "embezzlement / breach of trust (24m)"),
    "j_acct_24m": ("회계처리기준 위반 조치(24개월, log)", "accounting violation sanctions (24m)"),
    "j_unfaithful_24m": ("불성실공시법인 지정·예고(24개월, log)", "unfaithful disclosure designations (24m)"),
    "j_suspend_24m": ("영업정지 공시(24개월, log)", "business suspensions (24m)"),
    "j_recent_90d": ("법적 제재 공시가 최근 90일 안에 있음", "any legal filing in the last 90 days"),
    "j_lit_prov": ("소송충당부채 ÷ 자산", "litigation provisions / assets"),
    "j_accrual": ("우발손실 충당액 ÷ 자산", "loss-contingency accrual / assets"),
    "j_accrual_chg": ("우발손실 충당액 변화 ÷ 자산", "loss-contingency accrual change / assets"),
    "j_settle": ("소송 합의·우발손실 비용 ÷ 매출", "litigation settlement / loss expense / revenue"),
    "j_epa_n_24m": ("EPA 환경 집행 합의·명령 수(24개월, log)", "EPA enforcement conclusions (24m)"),
    "j_epa_pen_24m": ("EPA 벌금(24개월) ÷ 시가총액", "EPA penalties (24m) / market cap"),
    # block B
    "b_rating": ("최근 회사채 국내 신용등급(AAA = 1)", "latest bond rating (AAA = 1)"),
    "b_rating_chg": ("회사채 등급 변화(+ = 상승)", "bond rating change (+ = upgrade)"),
    "b_spread": ("최근 회사채 금리 − 같은 만기 국고채", "latest bond coupon - KTB of the same tenor"),
    "b_spread_chg": ("가산금리 변화(직전 발행 대비)", "spread change vs the previous issue"),
    "b_issue_12m": ("회사채 발행액(12개월) ÷ 시가총액", "bond issuance (12m) / market cap"),
    "b_cp_12m": ("기업어음·단기사채 발행액(12개월) ÷ 시가총액", "CP / short-term bonds (12m) / market cap"),
    "b_net_debt": ("순차입 조달(조달 − 상환) ÷ 자산", "net debt issuance / assets"),
    "b_int_assets": ("이자 지급액 ÷ 자산", "interest paid / assets"),
    "b_int_cost": ("이자비용 ÷ 평균 부채", "interest expense / average debt"),
    "b_int_spread": ("차입 비용 − 10년 국채 금리", "cost of debt - 10y Treasury"),
    "b_due_1y": ("1년 안 만기 원금 ÷ 부채", "principal due within 12 months / debt"),
    "b_int_cov": ("이자보상배율 log(영업이익 ÷ 이자비용)", "interest coverage (log)"),
    "r_kr_cs": ("회사채 AA− − 국고채 3년 가산금리 3개월 변화", "KR AA- credit spread, 3-month change"),
    "r_kr_cs_bbb": ("BBB− − AA− 가산금리 3년 z", "KR BBB- minus AA- spread, 3-year z"),
    "r_us_baa": ("BAA − 10년 국채 3개월 변화", "US BAA - 10y, 3-month change"),
    "r_us_def": ("BAA − AAA 3년 z", "US BAA - AAA, 3-year z"),
}
BLOCK_COLS = {
    "R": ["p_pub_n", "p_pub_growth", "p_pub_int"],
    "X": [c for c in DOCS if c[:2] in ("x_", "g_")],
    "J": [c for c in DOCS if c.startswith("j_")],
    "B": [c for c in DOCS if c.startswith("b_")] + ["r_kr_cs", "r_kr_cs_bbb", "r_us_baa", "r_us_def"],
}


def root(settings) -> Path:
    return kd.root(settings, "v16")


# ================================================================================================ names & universe

_KO_DROP = re.compile(r"\(주\)|㈜|주식회사|\(유\)|유한회사")


def norm_ko(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", _KO_DROP.sub("", str(s or ""))).upper()


def norm_en(s: str) -> str:
    return norm_name(re.sub(r"\([^)]*\)", " ", str(s or "")))


def name_keys(names) -> set[tuple[str, str]]:
    """('ko', x) / ('en', x) keys of a set of names (Hangul names -> ko, others -> en)."""
    out = set()
    for n in names:
        if not n:
            continue
        if re.search(r"[가-힣]", str(n)):
            k = norm_ko(n)
            if len(k) >= 2:
                out.add(("ko", k))
        else:
            k = norm_en(n)
            if len(k) >= 2:
                out.add(("en", k))
    return out


def universe(market: str, settings, cache) -> pd.DataFrame:
    """Ever-members of the market's PIT universe: entity, ticker, name (latest)."""
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market(market), settings)
    m = load_pit_universe(Market(market), settings, cache, n, start).membership
    return m.sort_values("date").groupby("entity", as_index=False).agg(ticker=("ticker", "last"), name=("name", "last"))


def kr_corp_codes(settings) -> dict[str, str]:
    with open(Path(settings.cache_path) / "pit" / "kr" / "opendart" / "corp_codes.pkl", "rb") as fh:
        return pickle.load(fh)[1]


def kr_company(settings, corp: str) -> dict:
    p = Path(settings.cache_path) / "alt" / "kr_dart_company" / f"{corp}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def kr_names(settings, ents: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    cc = kr_corp_codes(settings)
    out = {}
    for e, nm in ents[["entity", "name"]].itertuples(index=False):
        co = kr_company(settings, cc.get(str(e), ""))
        out[str(e)] = name_keys([nm, co.get("corp_name"), co.get("corp_name_eng"), co.get("stock_name")])
    return out


def us_chains(settings, cache, ents: pd.DataFrame) -> dict[str, set[int]]:
    from us_v8_data import cik_chains

    return cik_chains(SimpleNamespace(frame=ents[["entity"]]), settings, cache)


def us_names(
    settings, cache, ents: pd.DataFrame, chains: dict[str, set[int]], former: bool = True
) -> dict[str, set[tuple[str, str]]]:
    """Normalised English names: the universe name + SEC names of every CIK in the chain (with former names)."""
    from quantcio.data.pit.us_cik import Submissions
    from quantcio.data.pit.us_net import PitNet

    subs = Submissions(PitNet(settings))
    out = {}
    for e, nm in ents[["entity", "name"]].itertuples(index=False):
        names = [nm]
        for c in chains.get(str(e), set()):
            names += subs.names(c) if former else subs.names(c)[:1]
        out[str(e)] = {k for k in name_keys(names) if k[0] == "en"}
    return out


# ================================================================================================ fetch: DART


def _dart_get(sess: requests.Session, k: str, api: str, params: dict) -> dict:
    for attempt in range(40):
        try:
            js = sess.get(f"{DART_API}/{api}.json", params={"crtfc_key": k, **params}, timeout=60).json()
        except (requests.RequestException, ValueError) as exc:
            if attempt >= 5:
                raise RuntimeError(kd._redact(f"DART {api}: {exc}", k)) from None
            time.sleep(3.0 * (attempt + 1))
            continue
        if js.get("status") == "800":
            say("DART 시스템 점검(800) — 10분 뒤 다시 시도")
            time.sleep(600)
            continue
        if js.get("status") == "020":
            raise RuntimeError("DART 일일 한도(020) 도달 — 내일 다시 실행")
        return js
    raise RuntimeError("DART 점검이 끝나지 않음")


def dart_bonds_fetch(settings, codes: list[str], first_year: int = 2015) -> dict:
    """채무증권 발행실적 of every annual report (first_year .. last year) + the latest half-year report, per code."""
    k = kd.key("DART_API_KEY")
    if not k:
        return {"skipped": "DART_API_KEY 없음"}
    d = root(settings) / "dart_bonds"
    d.mkdir(exist_ok=True)
    cc = kr_corp_codes(settings)
    sess = requests.Session()
    today = pd.Timestamp.today()
    reports = [(y, "11011") for y in range(first_year, today.year)] + [(today.year, "11012")]
    calls = 0
    for i, code in enumerate(sorted(codes), 1):
        corp = cc.get(code)
        if not corp:
            continue
        p = d / f"{code}.json"
        have = json.loads(p.read_text()) if p.exists() else {}
        for y, rc in reports:
            key = f"{y}_{rc}"
            if key in have and (have[key] or y < today.year - 1):
                continue
            js = _dart_get(sess, k, "detScritsIsuAcmslt", {"corp_code": corp, "bsns_year": str(y), "reprt_code": rc})
            have[key] = js.get("list", []) if js.get("status") == "000" else []
            calls += 1
            time.sleep(0.1)
        p.write_text(json.dumps(have, ensure_ascii=False))
        if i % 50 == 0:
            say(f"DART 채무증권 {i}/{len(codes)} codes, {calls} calls")
    return {"codes": len(codes), "calls": calls}


def kr_company_fetch(settings, codes: list[str]) -> dict:
    """company.json for universe codes whose cache entry is missing (names for the OpenAlex match)."""
    k = kd.key("DART_API_KEY")
    cc = kr_corp_codes(settings)
    d = Path(settings.cache_path) / "alt" / "kr_dart_company"
    d.mkdir(parents=True, exist_ok=True)
    sess, n = requests.Session(), 0
    for code in codes:
        corp = cc.get(code)
        if corp and not (d / f"{corp}.json").exists() and k:
            js = _dart_get(sess, k, "company", {"corp_code": corp})
            if js.get("status") == "000":
                (d / f"{corp}.json").write_text(json.dumps(js, ensure_ascii=False))
                n += 1
            time.sleep(0.1)
    return {"fetched": n}


# ------------------------------------------------------------------------------------------------ KR bulk statements

KR_SINGLE = {  # item -> (statement, local concept names); the first line wins
    "tax": ("PL", {"IncomeTaxExpenseContinuingOperations"}),
    "pretax": ("PL", {"ProfitLossBeforeTax"}),
    "assets": ("BS", {"Assets"}),
    "liabilities": ("BS", {"Liabilities"}),
}
KR_SUM = {  # item -> (statement, local concept names); absolute values summed
    "tax_paid": ("CF", {"IncomeTaxesPaidRefundClassifiedAsOperatingActivities", "IncomeTaxesPaidRefund"}),
    "grant_cf": (
        "CF",
        {
            "ProceedsFromGovernmentGrantsClassifiedAsInvestingActivities",
            "ProceedsFromGovernmentGrantsClassifiedAsFinancingActivities",
            "ProceedsFromGovernmentGrantsClassifiedAsOperatingActivities",
        },
    ),
    "interest_paid": (
        "CF",
        {
            "InterestPaidClassifiedAsOperatingActivities",
            "InterestPaidClassifiedAsFinancingActivities",
            "InterestPaidClassifiedAsInvestingActivities",
        },
    ),
}
_PREFIX = re.compile(r"^(ifrs-full_|ifrs_|dart_)")
_DEBT_IN = re.compile(r"^Proceeds(From|Of)\w*(Borrowing|Bond|Debenture|Loan|Notes)")
_DEBT_OUT = re.compile(r"^Repayments?(Of|For)\w*(Borrowing|Bond|Debenture|Loan|Notes)")
_GRANT_LABEL = re.compile(r"(정부|국고)보조금")
_LIT_LABEL = re.compile(r"소송충당")


def _bulk_zip_rows(path: str, stmt: str) -> dict[tuple, dict[str, float]]:
    """(stock_code, period_end, consolidated) -> {item: value} for one FY bulk zip."""
    import csv

    out: dict[tuple, dict[str, float]] = {}
    lookup = {n: (it, "first") for it, (st, ns) in KR_SINGLE.items() if st == stmt for n in ns}
    lookup.update({n: (it, "sum") for it, (st, ns) in KR_SUM.items() if st == stmt for n in ns})
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = info.filename if info.flag_bits & 0x800 else info.filename.encode("cp437").decode("cp949", "replace")
            cons = "_연결_" in name
            rdr = csv.reader(
                io.StringIO(z.read(info).decode("cp949", errors="replace")), delimiter="\t", quoting=csv.QUOTE_NONE
            )
            header = [c.strip() for c in next(rdr, [])]
            vk = next((k for k, c in enumerate(header) if k >= 12 and c.startswith("당기")), None)
            if vk is None:
                continue
            for row in rdr:
                if len(row) <= vk or "사업보고서" not in row[8]:
                    continue
                code, label = row[10].strip(), re.sub(r"\s+", "", row[11])
                local = _PREFIX.sub("", code)
                item, mode = lookup.get(local, (None, None))
                if item is None and stmt == "CF" and "Government" not in local:
                    if _DEBT_IN.match(local):
                        item, mode = "debt_in", "sum"
                    elif _DEBT_OUT.match(local):
                        item, mode = "debt_out", "sum"
                if item is None and stmt == "BS":
                    if _GRANT_LABEL.search(label):
                        item, mode = "grant_bs", "sum"
                    elif _LIT_LABEL.search(label):
                        item, mode = "lit_prov", "sum"
                if item is None:
                    continue
                s = row[vk].strip().replace(",", "")
                try:
                    val = float(s)
                except ValueError:
                    continue
                key = (row[1].strip().strip("[]"), row[7].strip(), cons)
                rec = out.setdefault(key, {})
                if mode == "first":
                    rec.setdefault(item, val)
                else:
                    rec[item] = rec.get(item, 0.0) + abs(val)
    return out


def kr_fin_fetch(settings) -> dict:
    """Parse the cached FY bulk zips (BS/PL/CF) -> kr_fin.parquet: one row per (code, period_end) (consolidated when
    the company filed consolidated statements)."""
    zdir = Path(settings.cache_path) / "pit" / "kr" / "dart_bulk" / "zips"
    recs: dict[tuple, dict[str, float]] = {}
    years = sorted({int(Path(p).name[:4]) for p in glob.glob(str(zdir / "*_4Q_*.zip"))})
    for y in years:
        for stmt in ("BS", "PL", "CF"):
            files = sorted(glob.glob(str(zdir / f"{y}_4Q_{stmt}_*.zip")))
            if not files:
                continue
            for key, rec in _bulk_zip_rows(files[-1], stmt).items():
                recs.setdefault(key, {}).update(rec)
        say(f"KR bulk FY{y}: {len(recs)} company-basis rows so far")
    df = pd.DataFrame([{"code": k[0], "period_end": k[1], "cons": k[2], **v} for k, v in recs.items()])
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df = df.dropna(subset=["period_end"])
    has_cons = df[df["cons"]].groupby(["code", "period_end"]).size()
    keep = df["cons"] | ~df.set_index(["code", "period_end"]).index.isin(has_cons.index)
    df = df[keep.to_numpy()].drop(columns="cons").sort_values(["code", "period_end"]).reset_index(drop=True)
    df.to_parquet(root(settings) / "kr_fin.parquet", index=False)
    return {"rows": int(len(df)), "codes": int(df["code"].nunique()), "years": years}


def kr_avail(settings, code: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """period_end (month end) -> the day after the LATEST statement version of the annual report (what the bulk
    file carries), from the cached OpenDART periodic list."""
    from quantcio.data.pit import kr_dart as kdart

    p = Path(settings.cache_path) / "pit" / "kr" / "opendart" / "list" / f"{code}.pkl"
    if not p.exists():
        return {}
    with open(p, "rb") as fh:
        rows = pickle.load(fh)[1]
    t = kdart.publication_table(rows)
    t = t[t["kind"] == "사업보고서"]
    return {
        pd.Timestamp(pe): pd.Timestamp(d) + pd.Timedelta(1, "D") for pe, d in zip(t["period_end"], t["last_rcept_dt"])
    }


def kr_fin_table(settings) -> dict[str, pd.DataFrame]:
    """code -> annual rows with prev-year values and avail (the day after the latest version; unknown: + 120 days)."""
    p = root(settings) / "kr_fin.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    df["pe_m"] = df["period_end"] + pd.offsets.MonthEnd(0)
    out = {}
    for code, g in df.groupby("code"):
        g = g.sort_values("pe_m").drop_duplicates("pe_m", keep="last").reset_index(drop=True)
        av = kr_avail(settings, code)
        g["avail"] = [av.get(pe, pe + pd.Timedelta(120, "D")) for pe in g["pe_m"]]
        g["avail_known"] = [pe in av for pe in g["pe_m"]]
        prev = g.set_index("pe_m")
        for c in ("tax", "assets", "pretax"):
            g[f"{c}_prev"] = [
                prev[c].get(pe - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0), np.nan) if c in prev else np.nan
                for pe in g["pe_m"]
            ]
        out[str(code)] = g
    return out


# ------------------------------------------------------------------------------------------------ ECOS yields

ECOS_DAILY = {
    "ktb1": "010190000",
    "ktb3": "010200000",
    "ktb5": "010200001",
    "ktb10": "010210000",
    "ktb20": "010220000",
    "aa3": "010300000",
    "bbb3": "010320000",
}


def ecos_fetch(settings) -> dict:
    k = kd.key("ECOS_API_KEY")
    if not k:
        return {"skipped": "ECOS_API_KEY 없음"}
    end = pd.Timestamp.today().strftime("%Y%m%d")
    frames = {}
    for name, item in ECOS_DAILY.items():
        rows, s = [], 1
        while True:
            url = f"{kd.ECOS}/StatisticSearch/{k}/json/kr/{s}/{s + 9999}/817Y002/D/20050101/{end}/{item}"
            js = requests.get(url, timeout=180).json()
            part = (js.get("StatisticSearch") or {}).get("row") or []
            if not part and not rows:
                raise RuntimeError(kd._redact(f"ECOS 817Y002 {item}: {str(js)[:200]}", k))
            rows += part
            total = int((js.get("StatisticSearch") or {}).get("list_total_count") or 0)
            s += 10000
            if not part or s > total:
                break
        frames[name] = pd.Series(
            pd.to_numeric([r["DATA_VALUE"] for r in rows], errors="coerce"),
            index=pd.to_datetime([r["TIME"] for r in rows], format="%Y%m%d"),
        )
        time.sleep(0.3)
    df = pd.DataFrame(frames).sort_index()
    df.to_parquet(root(settings) / "ecos_yields.parquet")
    return {"rows": int(len(df)), "from": str(df.index.min().date())}


def ecos_yields(settings) -> pd.DataFrame | None:
    p = root(settings) / "ecos_yields.parquet"
    return pd.read_parquet(p) if p.exists() else None


# ------------------------------------------------------------------------------------------------ KR bonds table

_RATING = {
    "AAA": 1, "AA+": 2, "AA": 3, "AA0": 3, "AA-": 4, "A+": 5, "A": 6, "A0": 6, "A-": 7,
    "BBB+": 8, "BBB": 9, "BBB0": 9, "BBB-": 10, "BB+": 11, "BB": 12, "BB0": 12, "BB-": 13,
    "B+": 14, "B": 15, "B0": 15, "B-": 16, "CCC": 17, "CC": 18, "C": 19, "D": 20,
}  # fmt: skip
_DOMESTIC = re.compile(r"한기평|한신평|NICE|나이스|한국기업평가|한국신용평가|서울신용평가|서신평|이크레더블")
_FOREIGN = re.compile(r"S&P|Moody|Fitch|무디스|피치", re.I)
_RAT_RE = re.compile(r"^\s*(AAA|AA|A|BBB|BB|B|CCC|CC|C|D)\s*([+\-0])?(?![0-9A-Za-z])")


def parse_rating(text: str) -> float:
    t = str(text or "")
    if _FOREIGN.search(t) and not _DOMESTIC.search(t):
        return np.nan
    m = _RAT_RE.match(t.replace("−", "-"))
    if not m:
        return np.nan
    return float(_RATING.get(m.group(1) + (m.group(2) or ""), np.nan))


def _num(x) -> float:
    s = str(x or "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _ymd(x) -> pd.Timestamp | None:
    s = re.sub(r"[^0-9]", "", str(x or ""))
    if len(s) < 8:
        return None
    try:
        return pd.Timestamp(s[:8])
    except ValueError:
        return None


def ktb_at(y: pd.DataFrame, day: pd.Timestamp, tenor: float) -> float:
    """KTB yield (%) of ``tenor`` years on the last ECOS day <= day (linear between 1/3/5/10/20 years)."""
    i = y.index.searchsorted(day, "right") - 1
    if i < 0:
        return np.nan
    row = y.iloc[max(0, i - 5) : i + 1].ffill().iloc[-1]
    pts = [
        (1, row.get("ktb1")),
        (3, row.get("ktb3")),
        (5, row.get("ktb5")),
        (10, row.get("ktb10")),
        (20, row.get("ktb20")),
    ]
    pts = [(t, v) for t, v in pts if v is not None and np.isfinite(v)]
    if not pts:
        return np.nan
    xs, vs = zip(*pts, strict=True)
    return float(np.interp(min(max(tenor, xs[0]), xs[-1]), xs, vs))


_LATIN_KO = dict(
    zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "에이 비 씨 디 이 에프 지 에이치 아이 제이 케이 엘 엠 엔 오 피 큐 알 에스 티 유 브이 더블유 엑스 와이 "
        "지".split(),
        strict=True,
    )
)


def norm_issuer(s: str) -> str:
    """Korean company name with Latin letters spelt in Hangul ('CJ제일제당' = '씨제이제일제당')."""
    return "".join(_LATIN_KO.get(ch, ch) for ch in norm_ko(s))


def own_issuer(settings, code: str):
    """Predicate: the issuing company of a debt row is ``code`` itself (consolidated reports also list the
    subsidiaries' issues). Name match only: issues made under a former company name are dropped."""
    co = kr_company(settings, kr_corp_codes(settings).get(code, ""))
    keys = {norm_issuer(co.get("corp_name")), norm_issuer(co.get("stock_name"))} - {""}

    def ok(name) -> bool:
        n = norm_issuer(name)
        return bool(n) and any(n == k or n.startswith(k) or k.startswith(n) for k in keys)

    return ok


def kr_bond_table(settings, code: str, yields: pd.DataFrame | None) -> pd.DataFrame:
    """Unique own issues of ``code``: issue date, kind (bond / cp / other), face value, coupon, rating, spread."""
    p = root(settings) / "dart_bonds" / f"{code}.json"
    if not p.exists():
        return pd.DataFrame()
    own = own_issuer(settings, code)
    recs = {}
    for rows in json.loads(p.read_text()).values():
        for r in rows or []:
            kind = str(r.get("scrits_knd_nm") or "")
            d = _ymd(r.get("isu_de"))
            if d is None or kind in ("", "-") or not own(r.get("isu_cmpny")):
                continue
            face = _num(r.get("facvalu_totamt"))
            key = (d, kind, face, str(r.get("intrt")), str(r.get("mtd")))
            if key in recs:
                continue
            if re.search(r"기업어음|단기사채", kind):
                k = "cp"
            elif "사채" in kind and not re.search(r"전환|신주인수권|교환|이익참가", kind):
                k = "bond"
            else:
                k = "other"
            mtd = _ymd(r.get("mtd"))
            recs[key] = {
                "date": d,
                "kind": k,
                "face": face,
                "coupon": _num(r.get("intrt")),
                "rating": parse_rating(r.get("evl_grad_instt")) if k == "bond" else np.nan,
                "tenor": (mtd - d).days / 365.25 if mtd is not None and mtd > d else np.nan,
            }
    t = pd.DataFrame(list(recs.values()))
    if t.empty:
        return t
    t["spread"] = np.nan
    if yields is not None:
        ok = (t["kind"] == "bond") & t["rating"].notna() & t["coupon"].between(0.01, 30) & t["tenor"].notna()
        t.loc[ok, "spread"] = [
            c - ktb_at(yields, d, n) for c, d, n in t.loc[ok, ["coupon", "date", "tenor"]].to_numpy()
        ]
    return t.sort_values("date").reset_index(drop=True)


# ================================================================================================ fetch: OpenAlex


def _oa_get(path: str, params: dict) -> tuple[dict, int | None]:
    for attempt in range(6):
        try:
            r = requests.get(OPENALEX + path, params=params, timeout=90, headers=UA)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            raise RuntimeError("OpenAlex 하루 무료 한도 소진 — 내일 다시 실행(받은 부분은 저장됨)")
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        rem = r.headers.get("x-ratelimit-remaining")
        return r.json(), (int(rem) if rem and rem.isdigit() else None)
    raise RuntimeError(f"OpenAlex {path}: 응답 없음")


def openalex_institutions(settings, cc: str) -> list[dict]:
    p = root(settings) / f"openalex_inst_{cc}.json"
    if p.exists():
        return json.loads(p.read_text())
    rows, cursor = [], "*"
    while cursor:
        js, _ = _oa_get(
            "/institutions",
            {
                "filter": f"type:company,country_code:{cc}",
                "per_page": 200,
                "cursor": cursor,
                "select": "id,display_name,display_name_alternatives,lineage,works_count",
            },
        )
        rows += js.get("results") or []
        cursor = (js.get("meta") or {}).get("next_cursor")
        if not js.get("results"):
            break
        time.sleep(0.2)
    p.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def openalex_match(insts: list[dict], names: dict[str, set[tuple[str, str]]]) -> dict[str, list[str]]:
    """entity -> top-most matched institution ids (exact normalised name match only)."""
    by_key: dict[tuple[str, str], set[str]] = {}
    lineage = {}
    for r in insts:
        iid = r["id"].rsplit("/", 1)[-1]
        lineage[iid] = {x.rsplit("/", 1)[-1] for x in r.get("lineage") or []}
        for k in name_keys([r.get("display_name"), *(r.get("display_name_alternatives") or [])]):
            by_key.setdefault(k, set()).add(iid)
    out = {}
    for e, keys in names.items():
        hit = set().union(*[by_key.get(k, set()) for k in keys]) if keys else set()
        top = sorted(i for i in hit if not ((lineage.get(i, set()) - {i}) & hit))
        if top:
            out[e] = top
    return out


def openalex_fetch(settings, market_names: dict[str, dict[str, set[tuple[str, str]]]]) -> dict:
    p = root(settings) / "openalex_counts.json"
    have = json.loads(p.read_text()) if p.exists() else {}
    stats = {}
    for market, names in market_names.items():
        insts = openalex_institutions(settings, market.lower())
        match = openalex_match(insts, names)
        stats[market] = {"institutions": len(insts), "entities": len(names), "matched": len(match)}
        have.setdefault("match", {})[market] = match
        for e, ids in match.items():
            key = f"{market}:{e}"
            if key in have.get("counts", {}):
                continue
            try:
                js, rem = _oa_get(
                    "/works",
                    {"filter": f"authorships.institutions.lineage:{'|'.join(ids)}", "group_by": "publication_year"},
                )
            except RuntimeError:
                p.write_text(json.dumps(have, ensure_ascii=False))
                raise
            have.setdefault("counts", {})[key] = {g["key"]: int(g["count"]) for g in js.get("group_by") or []}
            if len(have["counts"]) % 50 == 0:
                p.write_text(json.dumps(have, ensure_ascii=False))
                say(f"OpenAlex counts {len(have['counts'])} (남은 무료 호출 {rem})")
            time.sleep(0.15)
        p.write_text(json.dumps(have, ensure_ascii=False))
    return stats


def openalex_counts(settings) -> dict:
    p = root(settings) / "openalex_counts.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ================================================================================================ fetch: US XBRL

X_CONCEPTS: dict[str, list[str]] = {
    "assets": ["Assets"],
    "revenue": list(v4.REVENUE),
    "op_income": ["OperatingIncomeLoss"],
    "tax": ["IncomeTaxExpenseBenefit"],
    "pretax": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "tax_paid": ["IncomeTaxesPaidNet", "IncomeTaxesPaid"],
    "utb": ["UnrecognizedTaxBenefits"],
    "accrual": [
        "LossContingencyAccrualAtCarryingValue",
        "LitigationReserve",
        "LossContingencyAccrualCarryingValueCurrent",
    ],
    "settle": ["LitigationSettlementExpense", "LossContingencyLossInPeriod", "LossContingencyAccrualProvision"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "ltd": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "DebtLongtermAndShorttermCombinedAmount",
    ],
    "std": ["ShortTermBorrowings", "CommercialPaper"],
    "due_1y": ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"],
    "debt_in": [
        "ProceedsFromIssuanceOfLongTermDebt",
        "ProceedsFromIssuanceOfSeniorLongTermDebt",
        "ProceedsFromIssuanceOfDebt",
        "ProceedsFromIssuanceOfUnsecuredDebt",
        "ProceedsFromDebtNetOfIssuanceCosts",
        "ProceedsFromIssuanceOfLongTermDebtAndCapitalSecuritiesNet",
    ],
    "debt_out": [
        "RepaymentsOfLongTermDebt",
        "RepaymentsOfDebt",
        "RepaymentsOfSeniorDebt",
        "RepaymentsOfUnsecuredDebt",
        "RepaymentsOfLongTermDebtAndCapitalSecurities",
    ],
}
X_INSTANT = {"assets", "utb", "accrual", "ltd", "std", "due_1y"}


def xbrl_fetch(settings, ciks) -> dict:
    from quantcio.data.pit.us_net import PitNet

    net = PitNet(settings)
    want = {c for cs in X_CONCEPTS.values() for c in cs}
    ciks = sorted({int(c) for c in ciks})
    n_new = 0
    for i, cik in enumerate(ciks, 1):
        rel = f"facts_v16/{cik}.json.gz"
        if net.read_json("sec", rel) is not None:
            continue
        r = net.http_get(COMPANYFACTS.format(cik=cik), "sec", ok=(200, 404))
        facts = {}
        if r.status_code == 200:
            g = (r.json().get("facts") or {}).get("us-gaap") or {}
            for c in want & set(g):
                usd = ((g[c] or {}).get("units") or {}).get("USD") or []
                facts[c] = [{k: f.get(k) for k in ("start", "end", "val", "form", "filed")} for f in usd]
        net.write_json("sec", rel, {"cik": cik, "facts": facts})
        n_new += 1
        if i % 100 == 0:
            say(f"XBRL (v16) {i}/{len(ciks)}")
    return {"ciks": len(ciks), "fetched": n_new}


def xbrl_facts(settings, ciks) -> dict[int, dict]:
    from quantcio.data.pit.us_net import PitNet

    net = PitNet(settings)
    return {int(c): (net.read_json("sec", f"facts_v16/{int(c)}.json.gz") or {}) for c in ciks}


def _first_filed(facts: dict, item: str) -> pd.DataFrame:
    """(end, filed, value): first-filed annual (10-K) value per fiscal period over the item's concepts (priority)."""
    rows = []
    for prio, c in enumerate(X_CONCEPTS[item]):
        for f in facts.get(c, []) or []:
            if str(f.get("form") or "") not in v4.ANNUAL_FORMS:
                continue
            if item not in X_INSTANT and not (330 <= v4._days(f.get("start"), f.get("end")) <= 400):
                continue
            if f.get("val") is None or not f.get("end") or not f.get("filed"):
                continue
            rows.append((pd.Timestamp(f["end"]), pd.Timestamp(f["filed"]), prio, float(f["val"])))
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "value"])
    df = pd.DataFrame(rows, columns=["end", "filed", "prio", "value"]).sort_values(["end", "filed", "prio"])
    return df.drop_duplicates("end", keep="first")[["end", "filed", "value"]].reset_index(drop=True)


def us_annual(facts: dict) -> pd.DataFrame:
    """One row per fiscal year end (anchored on annual total assets); a component counts only when it was first
    filed within 45 days of the anchor (the same 10-K season) - later first filings would move the row's date."""
    tabs = {k: _first_filed(facts, k) for k in X_CONCEPTS}
    recs = []
    for r in tabs["assets"].itertuples(index=False):
        rec = {"end": r.end, "filed": r.filed, "assets": r.value}
        for k, t in tabs.items():
            if k == "assets":
                continue
            m = v4._match(t, r.end)
            ok = m is not None and m["filed"] <= r.filed + pd.Timedelta(45, "D")
            rec[k] = float(m["value"]) if ok else np.nan
            if ok:
                rec["filed"] = max(rec["filed"], m["filed"])
        recs.append(rec)
    t = pd.DataFrame(recs).sort_values("end").reset_index(drop=True) if recs else pd.DataFrame()
    if t.empty:
        return t
    prev = t.set_index("end")
    for c in ("assets", "tax", "pretax", "accrual", "ltd", "std"):
        t[f"{c}_prev"] = [
            (lambda m: float(m["value"]) if m is not None else np.nan)(
                v4._match(prev[[c]].rename(columns={c: "value"}).reset_index(), e - pd.Timedelta(365, "D"), tol=45)
            )
            for e in t["end"]
        ]
    return t


# ================================================================================================ fetch: USAspending


def _usa_post(path: str, body: dict) -> dict:
    for attempt in range(8):
        try:
            r = requests.post(USASP + path, json=body, timeout=180, headers=UA)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"USAspending {path}: 응답 없음")


def _usa_entity(keys: set[tuple[str, str]]) -> tuple[dict, int]:
    en = {k for t, k in keys if t == "en" and len(k) >= 3}
    found: dict[str, dict] = {}
    calls = 0
    for nm in sorted(en):
        js = _usa_post("/recipient/", {"keyword": nm, "award_type": "all", "limit": 100})
        calls += 1
        for r in js.get("results") or []:
            if norm_en(r.get("name")) in en:
                found[r["id"]] = {"name": r.get("name"), "level": r.get("recipient_level")}
    ids = []
    for lvl in ("P", "R", "C"):
        ids = [rid for rid, v in found.items() if v["level"] == lvl]
        if ids:
            break
    quarters: dict[str, dict[str, float]] = {}
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for rid in ids:
        body = {
            "group": "quarter",
            "filters": {"recipient_id": rid, "time_period": [{"start_date": "2007-10-01", "end_date": today}]},
        }
        js = _usa_post("/search/spending_over_time/", body)
        calls += 1
        for q in js.get("results") or []:
            tp = q.get("time_period") or {}
            key = f"{tp.get('fiscal_year')}Q{tp.get('quarter')}"
            acc = quarters.setdefault(key, {"contract": 0.0, "assist": 0.0})
            acc["contract"] += float(q.get("Contract_Obligations") or 0) + float(q.get("Idv_Obligations") or 0)
            acc["assist"] += sum(
                float(q.get(f) or 0)
                for f in ("Grant_Obligations", "Loan_Obligations", "Direct_Obligations", "Other_Obligations")
            )
    return {"recipients": {rid: found[rid] for rid in ids}, "quarters": quarters}, calls


def usaspending_fetch(settings, names: dict[str, set[tuple[str, str]]], workers: int = 4) -> dict:
    """Per entity: recipients whose normalised name equals one of the company's CURRENT names (parent level P
    preferred, else R, else C) and their quarterly obligations by award type (FY2008+). Former names are not used:
    USAspending parent records carry the current name, and old generic names matched unrelated recipients."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    p = root(settings) / "usaspending.json"
    have = json.loads(p.read_text()) if p.exists() else {}
    todo = {e: k for e, k in sorted(names.items()) if e not in have}
    n_calls = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(_usa_entity, k): e for e, k in todo.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            have[futs[fu]], c = fu.result()
            n_calls += c
            if i % 50 == 0:
                p.write_text(json.dumps(have))
                n_m = sum(1 for v in have.values() if v["recipients"])
                say(f"USAspending {i}/{len(todo)} entities, {n_calls} calls, matched {n_m}")
    p.write_text(json.dumps(have))
    return {"entities": len(have), "matched": sum(1 for v in have.values() if v["recipients"]), "calls": n_calls}


def usaspending_table(settings) -> dict[str, pd.DataFrame]:
    """entity -> (avail, contract, assist) per fiscal quarter; avail = quarter end + 90 days."""
    p = root(settings) / "usaspending.json"
    if not p.exists():
        return {}
    out = {}
    for e, v in json.loads(p.read_text()).items():
        if not v.get("recipients"):
            continue
        recs = []
        for key, q in v["quarters"].items():
            fy, qq = int(key.split("Q")[0]), int(key.split("Q")[1])
            end = pd.Timestamp(fy - 1, 12, 31) + pd.offsets.QuarterEnd(qq - 1)  # FY q1 = Oct-Dec of fy - 1
            recs.append({"end": end, "contract": q["contract"], "assist": q["assist"]})
        t = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["end", "contract", "assist"])
        out[e] = t.sort_values("end").reset_index(drop=True)
    return out


# ================================================================================================ fetch: EPA, FRED

FRED_IDS = {"dgs10": "DGS10", "dbaa": "DBAA", "daaa": "DAAA", "baa10y": "BAA10Y"}


def fred_fetch(settings) -> dict:
    k = kd.key("FRED_API_KEY")
    if not k:
        return {"skipped": "FRED_API_KEY 없음"}
    stats = {}
    for name, sid in FRED_IDS.items():
        r = requests.get(
            kd.FRED,
            params={"series_id": sid, "api_key": k, "file_type": "json", "observation_start": "2005-01-01"},
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"FRED {sid}: HTTP {r.status_code}", k))
        obs = pd.DataFrame(r.json().get("observations", []))
        s = pd.Series(pd.to_numeric(obs["value"], errors="coerce").to_numpy(), index=pd.to_datetime(obs["date"]))
        s.dropna().rename("value").to_frame().to_parquet(root(settings) / f"fred_{name}.parquet")
        stats[name] = int(s.notna().sum())
        time.sleep(0.6)
    return stats


def fred(settings, name: str) -> pd.Series | None:
    p = root(settings) / f"fred_{name}.parquet"
    return pd.read_parquet(p)["value"] if p.exists() else None


def epa_fetch(settings) -> dict:
    d = Path(settings.cache_path) / "alt" / "keyed" / "legal_us"
    d.mkdir(parents=True, exist_ok=True)
    z = d / "case_downloads.zip"
    if not z.exists():
        r = requests.get(EPA_ZIP, timeout=900, headers=UA)
        r.raise_for_status()
        z.write_bytes(r.content)
    with zipfile.ZipFile(z) as zf:
        dfd = pd.read_csv(zf.open("CASE_DEFENDANTS.csv"), dtype=str, usecols=["CASE_NUMBER", "DEFENDANT_NAME"])
        con = pd.read_csv(
            zf.open("CASE_ENFORCEMENT_CONCLUSIONS.csv"),
            dtype=str,
            usecols=[
                "CASE_NUMBER",
                "ENF_CONCLUSION_ID",
                "ENF_CONCLUSION_NAME",
                "SETTLEMENT_LODGED_DATE",
                "SETTLEMENT_ENTERED_DATE",
                "FED_PENALTY_ASSESSED_AMT",
                "STATE_LOCAL_PENALTY_AMT",
            ],
        )
    lodged = pd.to_datetime(con["SETTLEMENT_LODGED_DATE"], format="%m/%d/%Y", errors="coerce")
    entered = pd.to_datetime(con["SETTLEMENT_ENTERED_DATE"], format="%m/%d/%Y", errors="coerce")
    con["date"] = lodged.fillna(entered)
    con["penalty"] = pd.to_numeric(con["FED_PENALTY_ASSESSED_AMT"], errors="coerce").fillna(0) + pd.to_numeric(
        con["STATE_LOCAL_PENALTY_AMT"], errors="coerce"
    ).fillna(0)
    con = con.dropna(subset=["date"])
    con = con[con["date"] >= pd.Timestamp("2005-01-01")]
    names = pd.concat(
        [
            dfd.rename(columns={"DEFENDANT_NAME": "name"}),
            con[["CASE_NUMBER", "ENF_CONCLUSION_NAME"]].rename(columns={"ENF_CONCLUSION_NAME": "name"}),
        ]
    ).dropna()
    names["norm"] = names["name"].map(norm_en)
    names = names[names["norm"].str.len() >= 3].drop_duplicates(["CASE_NUMBER", "norm"])
    ev = con[["CASE_NUMBER", "ENF_CONCLUSION_ID", "date", "penalty"]].merge(
        names[["CASE_NUMBER", "norm"]], on="CASE_NUMBER"
    )
    ev.to_parquet(root(settings) / "epa_events.parquet", index=False)
    return {"conclusions": int(len(con)), "event_names": int(len(ev))}


def epa_match(settings, names: dict[str, set[tuple[str, str]]]) -> dict[str, pd.DataFrame]:
    """entity -> unique EPA conclusions (date, penalty) whose defendant equals the company's name or starts with a
    company name of two or more words (39.3)."""
    p = root(settings) / "epa_events.parquet"
    if not p.exists():
        return {}
    ev = pd.read_parquet(p)
    by_norm = {n: g for n, g in ev.groupby("norm")}
    by_first: dict[str, list[str]] = {}
    for n in by_norm:
        by_first.setdefault(n.split()[0], []).append(n)
    out = {}
    for e, keys in names.items():
        hit = set()
        for _, n in keys:
            if n in by_norm:
                hit.add(n)
            if len(n.split()) >= 2:
                hit.update(x for x in by_first.get(n.split()[0], []) if x.startswith(n + " "))
        if hit:
            m = pd.concat([by_norm[h] for h in hit]).drop_duplicates(["CASE_NUMBER", "ENF_CONCLUSION_ID"])
            out[e] = m[["date", "penalty"]].sort_values("date").reset_index(drop=True)
    return out


# ================================================================================================ feature helpers


def _windowed(ev_dates: np.ndarray, ev_vals: np.ndarray, dates: np.ndarray, days: int) -> tuple[np.ndarray, np.ndarray]:
    """(count, sum of values) of events with ev_date < d (known from the next day) and ev_date >= d - days."""
    if len(ev_dates) == 0:
        return np.zeros(len(dates)), np.zeros(len(dates))
    ev = ev_dates.astype("datetime64[ns]")
    o = np.argsort(ev)
    ev, vals = ev[o], ev_vals[o]
    cum = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(ev, dates, "left")
    lo = np.searchsorted(ev, dates - np.timedelta64(days, "D"), "left")
    return (hi - lo).astype(float), cum[hi] - cum[lo]


def _asof(table: pd.DataFrame, cols: list[str], avail: str, dates: np.ndarray, max_age: int = MAX_AGE) -> np.ndarray:
    t = table.dropna(subset=[avail]).sort_values(avail)
    t = t[~t[avail].duplicated(keep="last")]
    return v13._asof_rows(t.set_index(avail)[cols], dates, max_age)


def _mcap(frame: pd.DataFrame) -> pd.Series:
    return np.exp(frame["log_mcap"].astype(float))


def _daily_regime(s: pd.Series, dates: pd.DatetimeIndex, max_age: int = 10) -> np.ndarray:
    """Value known on each date = the series up to the previous calendar day (market data, next-day use)."""
    s = s.dropna().sort_index()
    shifted = pd.Series(s.to_numpy(), index=s.index + pd.Timedelta(1, "D"))
    return v13._asof_rows(shifted.to_frame(), dates.to_numpy(dtype="datetime64[ns]"), max_age)[:, 0]


def _z_daily(s: pd.Series, window: int = 756, minp: int = 252) -> pd.Series:
    m = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std()
    return (s - m) / sd


# ================================================================================================ block R


def block_r(frame: pd.DataFrame, market: str, settings) -> tuple[pd.DataFrame, dict]:
    oc = openalex_counts(settings)
    counts = oc.get("counts", {})
    matched = set((oc.get("match") or {}).get(market, {}))
    out = pd.DataFrame(np.nan, index=frame.index, columns=BLOCK_COLS["R"])
    dates = pd.DatetimeIndex(frame["date"])
    # last complete year usable from March 1 of the next year
    ylast = np.where(dates.month >= 3, dates.year - 1, dates.year - 2)
    mcap = _mcap(frame).to_numpy()
    ents = frame["entity"].astype(str).to_numpy()
    n_m = 0
    for e in np.unique(ents):
        if e not in matched:
            continue
        c = counts.get(f"{market}:{e}")
        if c is None:
            continue
        n_m += 1
        ii = np.where(ents == e)[0]
        n0 = np.array([c.get(str(y), 0) for y in ylast[ii]], dtype=float)
        n1 = np.array([c.get(str(y - 1), 0) for y in ylast[ii]], dtype=float)
        out.iloc[ii, 0] = np.log1p(n0)
        out.iloc[ii, 1] = np.log1p(n0) - np.log1p(n1)
        out.iloc[ii, 2] = n0 / mcap[ii]
    return out, {"openalex_matched_entities": n_m, "openalex_universe_matched": len(matched)}


# ================================================================================================ block X / J / B: KR


def _kr_fin_features(g: pd.DataFrame) -> pd.DataFrame:
    a = g["assets"].where(g["assets"] > 0)
    pre = g["pretax"].where(g["pretax"] > 0)
    etr = (g["tax"] / pre).clip(-0.5, 1)
    etr_prev = (g["tax_prev"] / g["pretax_prev"].where(g["pretax_prev"] > 0)).clip(-0.5, 1)
    col = lambda c: g[c] if c in g else pd.Series(np.nan, index=g.index)  # noqa: E731
    return pd.DataFrame(
        {
            "x_etr": etr,
            "x_cash_etr": (col("tax_paid") / pre).clip(-0.5, 1),
            "x_etr_chg": etr - etr_prev,
            "x_tax_chg": (g["tax"] - g["tax_prev"]) / g["assets_prev"].where(g["assets_prev"] > 0),
            "x_tax_gap": (g["tax"] - col("tax_paid")) / a,
            "g_grant_cf": col("grant_cf").fillna(0) / a,
            "g_grant_bs": col("grant_bs").fillna(0) / a,
            "j_lit_prov": col("lit_prov").fillna(0) / a,
            "b_net_debt": (col("debt_in").fillna(0) - col("debt_out").fillna(0)) / a,
            "b_int_assets": col("interest_paid") / a,
        }
    )


KR_LEGAL = {
    "j_suit_12m": (re.compile(r"소송등의제기|증권관련집단소송"), 365),
    "j_ruling_12m": (re.compile(r"소송등의판결"), 365),
    "j_fine_24m": (re.compile(r"벌금등의부과"), 730),
    "j_embezzle_24m": (re.compile(r"횡령ㆍ?·?배임"), 730),
    "j_acct_24m": (re.compile(r"회계처리기준위반"), 730),
    "j_unfaithful_24m": (re.compile(r"불성실공시법인지정"), 730),
    "j_suspend_24m": (re.compile(r"영업정지"), 730),
}


def kr_legal_events(settings, corp: str) -> pd.DataFrame:
    """(date, kind) of original (non-correction) legal filings in the cached DART lists (2013+)."""
    recs = []
    for f in glob.glob(str(Path(settings.cache_path) / "alt" / "kr_dart" / "list" / corp / "*.json")):
        for r in json.loads(Path(f).read_text()).get("rows", []):
            nm = str(r.get("report_nm") or "").strip()
            if nm.startswith("["):
                continue
            nm = re.sub(r"\s+", "", nm)
            for k, (pat, _d) in KR_LEGAL.items():
                if pat.search(nm):
                    recs.append({"date": pd.Timestamp(r["rcept_dt"]), "kind": k, "rcept_no": r.get("rcept_no")})
                    break
    t = pd.DataFrame(recs, columns=["date", "kind", "rcept_no"])
    return t.drop_duplicates("rcept_no") if len(t) else t


def block_kr(block: str, frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    cols = {
        "X": ["x_etr", "x_cash_etr", "x_etr_chg", "x_tax_chg", "x_tax_gap", "g_grant_cf", "g_grant_bs"],
        "J": [*KR_LEGAL, "j_recent_90d", "j_lit_prov"],
        "B": [
            "b_rating",
            "b_rating_chg",
            "b_spread",
            "b_spread_chg",
            "b_issue_12m",
            "b_cp_12m",
            "b_net_debt",
            "b_int_assets",
            "r_kr_cs",
            "r_kr_cs_bbb",
        ],
    }[block]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    info: dict = {}
    fin_cols = [c for c in cols if c in ("x_etr", "x_cash_etr", "x_etr_chg", "x_tax_chg", "x_tax_gap", "g_grant_cf",
                                         "g_grant_bs", "j_lit_prov", "b_net_debt", "b_int_assets")]  # fmt: skip
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    mcap = _mcap(frame).to_numpy()
    if fin_cols:
        fin = kr_fin_table(settings)
        n, known = 0, []
        for e in np.unique(ents):
            g = fin.get(e)
            if g is None or g.empty:
                continue
            n += 1
            known.append(float(g["avail_known"].mean()))
            feats = _kr_fin_features(g)
            feats["avail"] = g["avail"].to_numpy()
            ii = np.where(ents == e)[0]
            out.iloc[ii, [out.columns.get_loc(c) for c in fin_cols]] = _asof(feats, fin_cols, "avail", dates[ii])
        info["kr_fin_entities"] = n
        info["kr_fin_avail_known_share"] = round(float(np.mean(known)), 3) if known else None
    if block == "J":
        cc = kr_corp_codes(settings)
        n_ev = 0
        for e in np.unique(ents):
            corp = cc.get(e)
            if not corp:
                continue
            ev = kr_legal_events(settings, corp)
            ii = np.where(ents == e)[0]
            d = dates[ii]
            recent = np.zeros(len(ii))
            for k, (_p, days) in KR_LEGAL.items():
                sub = ev[ev["kind"] == k]
                cnt, _ = _windowed(sub["date"].to_numpy(), np.ones(len(sub)), d, days)
                out.iloc[ii, out.columns.get_loc(k)] = np.log1p(cnt)
            if len(ev):
                n_ev += len(ev)
                recent, _ = _windowed(ev["date"].to_numpy(), np.ones(len(ev)), d, 90)
            out.iloc[ii, out.columns.get_loc("j_recent_90d")] = (recent > 0).astype(float)
        info["kr_legal_filings"] = n_ev
    if block == "B":
        y = ecos_yields(settings)
        n_b, n_r = 0, 0
        for e in np.unique(ents):
            t = kr_bond_table(settings, e, y)
            ii = np.where(ents == e)[0]
            d = dates[ii]
            if t.empty:
                out.iloc[ii, out.columns.get_loc("b_issue_12m")] = 0.0
                out.iloc[ii, out.columns.get_loc("b_cp_12m")] = 0.0
                continue
            n_b += 1
            for kind, c in (("bond", "b_issue_12m"), ("cp", "b_cp_12m")):
                sub = t[t["kind"] == kind]
                _, s = _windowed(sub["date"].to_numpy(), sub["face"].fillna(0).to_numpy(), d, 365)
                out.iloc[ii, out.columns.get_loc(c)] = s / mcap[ii]
            rb = t[(t["kind"] == "bond") & t["rating"].notna()]
            if rb.empty:
                continue
            n_r += 1
            per_day = rb.groupby("date").agg(rating=("rating", "median"), spread=("spread", "mean")).sort_index()
            pdays = per_day.index.to_numpy(dtype="datetime64[ns]")
            k = np.searchsorted(pdays, d, "left") - 1  # issues strictly before d (known from the next day)
            for j, ix in enumerate(ii):
                if k[j] < 0:
                    continue
                last = per_day.iloc[k[j]]
                age = (d[j] - pdays[k[j]]) / DAY
                if age <= 3 * 365:
                    out.iat[ix, out.columns.get_loc("b_rating")] = last["rating"]
                    if k[j] >= 1 and (d[j] - pdays[k[j] - 1]) / DAY <= 3 * 365:
                        prev = per_day.iloc[k[j] - 1]
                        out.iat[ix, out.columns.get_loc("b_rating_chg")] = prev["rating"] - last["rating"]
                        if np.isfinite(last["spread"]) and np.isfinite(prev["spread"]):
                            out.iat[ix, out.columns.get_loc("b_spread_chg")] = last["spread"] - prev["spread"]
                if age <= 2 * 365 and np.isfinite(last["spread"]):
                    out.iat[ix, out.columns.get_loc("b_spread")] = last["spread"]
        info["kr_bond_entities"] = n_b
        info["kr_rated_entities"] = n_r
        if y is not None:
            dd = pd.DatetimeIndex(frame["date"])
            cs = (y["aa3"] - y["ktb3"]).dropna()
            out["r_kr_cs"] = _daily_regime(cs - cs.shift(63), dd)
            out["r_kr_cs_bbb"] = _daily_regime(_z_daily((y["bbb3"] - y["aa3"]).dropna()), dd)
    return out, info


# ================================================================================================ block X / J / B: US


def block_us(block: str, frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    from us_v8_data import cik_chains

    from quantcio.data.pit.us_cik import entity_cik

    cols = {
        "X": ["x_etr", "x_cash_etr", "x_etr_chg", "x_tax_chg", "x_tax_gap", "x_utb", "g_fed_contract",
              "g_fed_contract_chg", "g_fed_assist"],
        "J": ["j_accrual", "j_accrual_chg", "j_settle", "j_epa_n_24m", "j_epa_pen_24m"],
        "B": ["b_int_cost", "b_int_spread", "b_net_debt", "b_due_1y", "b_int_cov", "r_us_baa", "r_us_def"],
    }[block]  # fmt: skip
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    info: dict = {}
    chains = cik_chains(ds, settings, cache)
    facts = xbrl_facts(settings, {c for cs in chains.values() for c in cs})
    dgs10 = fred(settings, "dgs10")
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    mcap = _mcap(frame).to_numpy()
    annual: dict[str, pd.DataFrame] = {}
    for e in np.unique(ents):
        merged: dict[str, list] = {}
        for c in chains.get(e, {entity_cik(e)}):
            for k, v in ((facts.get(c) or {}).get("facts") or {}).items():
                merged.setdefault(k, []).extend(v)
        t = us_annual(merged) if merged else pd.DataFrame()
        if not t.empty:
            t["avail"] = t["filed"] + pd.Timedelta(1, "D")
            annual[e] = t
    info["xbrl_entities"] = len(annual)
    fin_cols = [c for c in cols if not c.startswith(("g_", "j_epa", "r_"))]
    revenue: dict[str, pd.DataFrame] = {}
    for e, t in annual.items():
        a = t["assets"].where(t["assets"] > 0)
        pre = t["pretax"].where(t["pretax"] > 0)
        etr = (t["tax"] / pre).clip(-0.5, 1)
        etr_prev = (t["tax_prev"] / t["pretax_prev"].where(t["pretax_prev"] > 0)).clip(-0.5, 1)
        debt = t["ltd"] + t["std"].fillna(0)
        debt_prev = t["ltd_prev"] + t["std_prev"].fillna(0)
        avg_debt = pd.concat([debt, debt_prev], axis=1).mean(axis=1).where(lambda x: x > 0)
        cost = (t["interest"] / avg_debt).clip(0, 0.3)
        y10 = (
            pd.Series(_daily_regime(dgs10, pd.DatetimeIndex(t["end"]), 10), index=t.index) / 100.0
            if dgs10 is not None
            else pd.Series(np.nan, index=t.index)
        )
        op = t["op_income"]
        cov = np.log((op / t["interest"].where(t["interest"] > 0)).where(op > 0)).clip(-3, 5)
        cov = cov.where(~((op <= 0) & (t["interest"] > 0)), -3.0)
        both_nan = t["debt_in"].isna() & t["debt_out"].isna()
        f = pd.DataFrame(
            {
                "x_etr": etr,
                "x_cash_etr": (t["tax_paid"] / pre).clip(-0.5, 1),
                "x_etr_chg": etr - etr_prev,
                "x_tax_chg": (t["tax"] - t["tax_prev"]) / t["assets_prev"].where(t["assets_prev"] > 0),
                "x_tax_gap": (t["tax"] - t["tax_paid"]) / a,
                "x_utb": t["utb"] / a,
                "j_accrual": t["accrual"] / a,
                "j_accrual_chg": (t["accrual"] - t["accrual_prev"]) / a,
                "j_settle": t["settle"] / t["revenue"].where(t["revenue"] > 0),
                "b_int_cost": cost,
                "b_int_spread": cost - y10,
                "b_net_debt": ((t["debt_in"].fillna(0) - t["debt_out"].fillna(0)) / a).where(~both_nan),
                "b_due_1y": (t["due_1y"] / debt.where(debt > 0)).clip(0, 1),
                "b_int_cov": cov,
                "revenue": t["revenue"],
                "avail": t["avail"],
            }
        )
        ii = np.where(ents == e)[0]
        if fin_cols:
            out.iloc[ii, [out.columns.get_loc(c) for c in fin_cols]] = _asof(f, fin_cols, "avail", dates[ii])
        revenue[e] = f[["revenue", "avail"]]
    if block == "X":
        usa = usaspending_table(settings)
        n_u = 0
        for e in np.unique(ents):
            ii = np.where(ents == e)[0]
            q = usa.get(e)
            if q is None or q.empty:
                continue
            n_u += 1
            d = dates[ii]
            avail = (q["end"] + pd.Timedelta(90, "D")).to_numpy(dtype="datetime64[ns]")
            con = q["contract"].to_numpy(float)
            ass = q["assist"].to_numpy(float)
            c4, c4p, a4 = (np.full(len(ii), np.nan) for _ in range(3))
            for j, dj in enumerate(d):
                k = np.searchsorted(avail, dj, "right")  # quarters known on dj
                if k == 0 or (dj - avail[k - 1]) / DAY > 200:
                    continue
                c4[j] = max(con[max(0, k - 4) : k].sum(), 0.0)
                a4[j] = max(ass[max(0, k - 4) : k].sum(), 0.0)
                c4p[j] = max(con[max(0, k - 8) : max(0, k - 4)].sum(), 0.0) if k >= 8 else np.nan
            rev = _asof(revenue[e], ["revenue"], "avail", d)[:, 0] if e in revenue else np.full(len(ii), np.nan)
            rev = np.where(rev > 0, rev, np.nan)
            out.iloc[ii, out.columns.get_loc("g_fed_contract")] = c4 / rev
            out.iloc[ii, out.columns.get_loc("g_fed_contract_chg")] = np.log((c4 + 1e6) / (c4p + 1e6))
            out.iloc[ii, out.columns.get_loc("g_fed_assist")] = a4 / rev
        info["usaspending_entities"] = n_u
    if block == "J":
        nm = us_names(settings, cache, universe("US", settings, cache), chains)
        epa = epa_match(settings, nm)
        n_e = 0
        for e in np.unique(ents):
            ii = np.where(ents == e)[0]
            ev = epa.get(e)
            if ev is None or ev.empty:
                out.iloc[ii, out.columns.get_loc("j_epa_n_24m")] = 0.0
                out.iloc[ii, out.columns.get_loc("j_epa_pen_24m")] = 0.0
                continue
            n_e += 1
            cnt, pen = _windowed(ev["date"].to_numpy(), ev["penalty"].to_numpy(float), dates[ii], 730)
            out.iloc[ii, out.columns.get_loc("j_epa_n_24m")] = np.log1p(cnt)
            out.iloc[ii, out.columns.get_loc("j_epa_pen_24m")] = pen / mcap[ii]
        info["epa_matched_entities"] = n_e
    if block == "B":
        dd = pd.DatetimeIndex(frame["date"])
        baa10 = fred(settings, "baa10y")
        dbaa, daaa = fred(settings, "dbaa"), fred(settings, "daaa")
        if baa10 is not None:
            out["r_us_baa"] = _daily_regime(baa10 - baa10.shift(63), dd)
        if dbaa is not None and daaa is not None:
            out["r_us_def"] = _daily_regime(_z_daily((dbaa - daaa).dropna()), dd)
    return out, info


def block(name: str, market: str, frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    if name == "R":
        return block_r(frame, market, settings)
    if market == "KR":
        return block_kr(name, frame, settings)
    return block_us(name, frame, ds, settings, cache)


# ================================================================================================ main


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument(
        "--what",
        default="all",
        choices=["all", "kr_fin", "dart_bonds", "kr_company", "ecos", "fred", "epa", "xbrl", "usaspending", "openalex"],
    )
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    w = a.what
    kr = universe("KR", s, cache)
    us = universe("US", s, cache)
    codes = sorted(kr["entity"].astype(str))
    say(f"universe: KR {len(kr)} entities, US {len(us)} entities")
    if w in ("all", "kr_fin"):
        say(f"KR bulk statements: {kr_fin_fetch(s)}")
    if w in ("all", "ecos"):
        say(f"ECOS yields: {ecos_fetch(s)}")
    if w in ("all", "fred"):
        say(f"FRED: {fred_fetch(s)}")
    if w in ("all", "epa"):
        say(f"EPA: {epa_fetch(s)}")
    if w in ("all", "kr_company"):
        say(f"DART company: {kr_company_fetch(s, codes)}")
    if w in ("all", "dart_bonds"):
        say(f"DART 채무증권: {dart_bonds_fetch(s, codes)}")
    chains = us_chains(s, cache, us) if w in ("all", "xbrl", "usaspending", "openalex") else {}
    if w in ("all", "xbrl"):
        say(f"XBRL: {xbrl_fetch(s, {c for cs in chains.values() for c in cs})}")
    if w in ("all", "usaspending", "openalex"):
        un = us_names(s, cache, us, chains)
        if w in ("all", "usaspending"):
            say(f"USAspending: {usaspending_fetch(s, us_names(s, cache, us, chains, former=False))}")
        if w in ("all", "openalex"):
            say(f"OpenAlex: {openalex_fetch(s, {'KR': kr_names(s, kr), 'US': un})}")


if __name__ == "__main__":
    main()
