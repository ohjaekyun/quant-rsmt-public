"""Sector classification for the PIT factor backtest (docs/BACKTEST_MACRO.md section 1).

US
  1. Current S&P 500 members: GICS sector from the Wikipedia 'List of S&P 500 companies' table (by CIK, the table
     already parsed by ``data/pit/us_membership.parse_wiki_current``).
  2. Removed members: SEC submissions ``sic`` code -> approximate GICS sector with ``SIC_TO_GICS`` below.
  3. Nothing found: "기타" (counted).
  Limitation: today's GICS (incl. the 2016 Real Estate split, the 2018 Communication Services and the 2023 payments
  reshuffle) stands in for the classification in force at each past date.

KR
  1. The KSIC code (3-digit 소분류) that the company itself reported in its DART periodic filings - the
     "업종"/"업종명" columns of the OpenDART bulk financial statements already cached by ``data/pit/kr_fund`` - as of
     the latest report with period end + 90 days <= t (covers delisted names; the earliest report is back-filled for
     names without an earlier report, counted). KIND/KRX 업종 is the same KSIC classification.
  2. No DART bulk row: KIND 상장법인목록 업종 (current listing), mapped through the KSIC names seen in the bulk.
  3. Nothing found: "기타" (counted).
  KSIC codes are grouped into broad KRX-style sectors with ``KSIC_TO_SECTOR`` below.

Sectors with fewer than ``backtest.min_names_per_sector`` members on a date are merged into "기타" for neutralisation
(see ``factors.neutral_groups``) - counted separately from unclassified names.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from quantcio.config import Settings
from quantcio.data.cache import DiskCache

log = logging.getLogger(__name__)

OTHER = "기타"

# English display names for every sector label produced here (Korean is the canonical key)
SECTOR_EN = {
    OTHER: "Other",
    "정보기술": "Information Technology",
    "헬스케어": "Health Care",
    "금융": "Financials",
    "경기소비재": "Consumer Discretionary",
    "필수소비재": "Consumer Staples",
    "산업재": "Industrials",
    "에너지": "Energy",
    "소재": "Materials",
    "유틸리티": "Utilities",
    "부동산": "Real Estate",
    "커뮤니케이션": "Communication Services",
    "2차전지·전기장비": "Batteries & Electrical Equipment",
    "IT·전자": "IT & Electronics",
    "건설": "Construction",
    "반도체": "Semiconductors",
    "운송": "Transportation",
    "자동차": "Automobiles",
    "제약·바이오": "Pharma & Biotech",
    "철강·소재": "Steel & Materials",
    "화학": "Chemicals",
}


def sector_label(name: str | None, lang: str = "ko") -> str | None:
    """Display name of a sector label in the report language (unknown labels pass through unchanged)."""
    if name is None or lang != "en":
        return name
    return SECTOR_EN.get(name, name)


SECTOR_VERSION = 1
REPORT_LAG_DAYS = 90  # KR: a periodic report's KSIC code is used from period end + 90 days

# ------------------------------------------------------------------------------------------------------------ US

GICS_KO = {
    "Information Technology": "정보기술",
    "Health Care": "헬스케어",
    "Financials": "금융",
    "Consumer Discretionary": "경기소비재",
    "Consumer Staples": "필수소비재",
    "Industrials": "산업재",
    "Energy": "에너지",
    "Materials": "소재",
    "Utilities": "유틸리티",
    "Real Estate": "부동산",
    "Communication Services": "커뮤니케이션",
}

# SIC range (inclusive) -> GICS sector. First match wins, so specific ranges come before broad ones.
# Approximation used only for S&P 500 members that are no longer in the current Wikipedia table.
SIC_TO_GICS: list[tuple[int, int, str]] = [
    # --- specific carve-outs first
    (1220, 1241, "Energy"),  # coal mining
    (1311, 1389, "Energy"),  # crude petroleum & natural gas, oil & gas field services
    (1531, 1531, "Consumer Discretionary"),  # operative builders (homebuilders)
    (2830, 2836, "Health Care"),  # drugs, biologicals
    (2840, 2844, "Consumer Staples"),  # soap, detergents, cosmetics
    (2900, 2999, "Energy"),  # petroleum refining
    (3011, 3011, "Consumer Discretionary"),  # tires
    (3570, 3579, "Information Technology"),  # computers & office equipment
    (3630, 3639, "Consumer Discretionary"),  # household appliances
    (3651, 3652, "Consumer Discretionary"),  # household audio & video
    (3660, 3679, "Information Technology"),  # communications equipment, semiconductors, electronic components
    (3710, 3716, "Consumer Discretionary"),  # motor vehicles & parts
    (3751, 3751, "Consumer Discretionary"),  # motorcycles
    (3812, 3812, "Industrials"),  # defense electronics / navigation
    (3820, 3829, "Information Technology"),  # measuring & controlling instruments
    (3840, 3851, "Health Care"),  # medical instruments & supplies
    (3861, 3861, "Information Technology"),  # photographic equipment
    (3873, 3873, "Consumer Discretionary"),  # watches
    (4600, 4619, "Energy"),  # pipelines (except natural gas)
    (4922, 4923, "Energy"),  # natural gas transmission
    (4950, 4959, "Industrials"),  # sanitary / waste services
    (5047, 5047, "Health Care"),  # medical & hospital equipment wholesale
    (5122, 5122, "Health Care"),  # drugs wholesale
    (5140, 5149, "Consumer Staples"),  # groceries wholesale
    (5171, 5172, "Energy"),  # petroleum products wholesale
    (5180, 5182, "Consumer Staples"),  # beer, wine & spirits wholesale
    (5331, 5331, "Consumer Staples"),  # variety stores (discount / dollar stores; GICS 2023)
    (5399, 5399, "Consumer Staples"),  # misc general merchandise (warehouse clubs)
    (5400, 5499, "Consumer Staples"),  # food stores
    (5912, 5912, "Consumer Staples"),  # drug stores
    (6500, 6553, "Real Estate"),  # real estate operators, developers
    (6798, 6798, "Real Estate"),  # REITs
    (7370, 7370, "Communication Services"),  # computer programming / internet services (search, social)
    (7371, 7379, "Information Technology"),  # software, IT services
    (7810, 7849, "Communication Services"),  # motion pictures, video
    (8000, 8099, "Health Care"),  # health services
    (8731, 8731, "Health Care"),  # commercial physical & biological research
    # --- broad divisions
    (100, 999, "Consumer Staples"),  # agriculture, forestry, fishing
    (1000, 1499, "Materials"),  # metal & nonmetallic mining
    (1500, 1799, "Industrials"),  # construction
    (2000, 2199, "Consumer Staples"),  # food, beverages, tobacco
    (2200, 2399, "Consumer Discretionary"),  # textiles, apparel
    (2400, 2499, "Materials"),  # lumber & wood
    (2500, 2599, "Consumer Discretionary"),  # furniture
    (2600, 2699, "Materials"),  # paper
    (2700, 2799, "Communication Services"),  # printing & publishing
    (2800, 2899, "Materials"),  # chemicals
    (3000, 3099, "Materials"),  # rubber & plastics
    (3100, 3199, "Consumer Discretionary"),  # leather
    (3200, 3399, "Materials"),  # stone, glass, primary metals
    (3400, 3599, "Industrials"),  # fabricated metal, industrial machinery
    (3600, 3699, "Industrials"),  # electrical equipment
    (3700, 3799, "Industrials"),  # aerospace, other transportation equipment
    (3800, 3899, "Information Technology"),  # other instruments
    (3900, 3999, "Consumer Discretionary"),  # misc manufacturing (toys, jewelry)
    (4000, 4799, "Industrials"),  # transportation
    (4800, 4899, "Communication Services"),  # telephone, radio & TV, cable
    (4900, 4999, "Utilities"),  # electric, gas, water
    (5000, 5199, "Industrials"),  # wholesale (durable goods / distributors)
    (5200, 5999, "Consumer Discretionary"),  # retail, restaurants
    (6000, 6799, "Financials"),  # banks, insurance, brokers, holding & blank-check companies
    (7000, 7299, "Consumer Discretionary"),  # hotels, personal services
    (7300, 7399, "Industrials"),  # business services
    (7500, 7599, "Consumer Discretionary"),  # auto repair & rental
    (7900, 7999, "Consumer Discretionary"),  # amusement & recreation (casinos, cruise)
    (8200, 8299, "Consumer Discretionary"),  # education
    (8100, 8999, "Industrials"),  # legal, engineering, management services
    (9995, 9995, "Financials"),  # non-operating establishments (SPAC / shell)
]


def sic_to_gics(sic: Any) -> str | None:
    """Approximate GICS sector (English) of a SEC SIC code, None when unknown."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    for lo, hi, sector in SIC_TO_GICS:
        if lo <= code <= hi:
            return sector
    return None


def _wiki_sectors(net: Any) -> dict[int, str]:
    from quantcio.data.pit import us_membership

    raw = net.cached_bytes("sp500", "wiki_sp500.html", us_membership.WIKI_CURRENT, "wiki", None)
    cur = us_membership.parse_wiki_current(raw)
    return {int(c): str(s) for c, s in zip(cur["cik"], cur["sector"], strict=False) if pd.notna(c) and pd.notna(s)}


def us_entity_sectors(
    settings: Settings,
    entities: list[str],
    *,
    wiki: dict[int, str] | None = None,
    sic: dict[int, Any] | None = None,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """entity ("CIK##########") -> Korean GICS sector label, entity -> source ('wiki_gics' | 'sec_sic' | 'none'),
    notes. ``wiki`` (cik -> GICS) and ``sic`` (cik -> SIC) are injection points for tests (no network)."""
    notes: list[str] = []
    net = None
    if wiki is None or sic is None:
        from quantcio.data.pit.us_net import PitNet

        net = PitNet(settings)
    if wiki is None:
        try:
            wiki = _wiki_sectors(net)
        except Exception as exc:  # noqa: BLE001 - every entity then falls back to SIC, noted
            log.warning("wikipedia GICS table unavailable: %s", exc)
            notes.append(f"Wikipedia GICS 표 로드 실패({type(exc).__name__}) → 전 종목 SEC SIC 근사 섹터 사용")
            wiki = {}
    subs = None
    sectors: dict[str, str] = {}
    source: dict[str, str] = {}
    failed = 0
    for e in entities:
        try:
            cik = int(str(e).removeprefix("CIK"))
        except ValueError:
            sectors[e], source[e] = OTHER, "none"
            continue
        g = wiki.get(cik)
        if g is not None:
            sectors[e], source[e] = GICS_KO.get(g, g), "wiki_gics"
            continue
        code = None
        if sic is not None:
            code = sic.get(cik)
        else:
            if subs is None:
                from quantcio.data.pit.us_cik import Submissions

                subs = Submissions(net)
            try:
                code = (subs.get(cik) or {}).get("sic")
            except Exception as exc:  # noqa: BLE001 - counted
                failed += 1
                log.warning("SEC submissions CIK%s failed: %s", cik, exc)
        g = sic_to_gics(code)
        if g is not None:
            sectors[e], source[e] = GICS_KO[g], "sec_sic"
        else:
            sectors[e], source[e] = OTHER, "none"
    if failed:
        notes.append(f"SEC submissions 조회 실패 {failed}개 CIK → 섹터 '기타'")
    return sectors, source, notes


# ------------------------------------------------------------------------------------------------------------ KR

# KSIC prefix -> broad KRX-style sector. The longest matching prefix wins (3-digit 소분류 before 2-digit 중분류).
KSIC_TO_SECTOR: dict[str, str] = {
    # IT
    "261": "반도체",
    "262": "IT·전자",
    "263": "IT·전자",
    "264": "IT·전자",
    "265": "IT·전자",
    "266": "IT·전자",
    "272": "IT·전자",
    "273": "IT·전자",
    "274": "IT·전자",
    "62": "IT·전자",  # computer programming, system integration
    "28": "2차전지·전기장비",  # 281 motors/generators, 282 batteries, 283 cables, 284 lighting, 285 appliances
    # communication / media / internet / games
    "58": "커뮤니케이션",  # publishing incl. 582 software & games
    "59": "커뮤니케이션",
    "60": "커뮤니케이션",
    "61": "커뮤니케이션",
    "63": "커뮤니케이션",  # portals, information services
    "713": "커뮤니케이션",  # advertising
    "90": "커뮤니케이션",
    # health care
    "21": "제약·바이오",
    "271": "제약·바이오",  # medical devices
    "701": "제약·바이오",  # natural-science R&D (biotech)
    "86": "제약·바이오",
    "87": "제약·바이오",
    # consumer
    "30": "자동차",
    "01": "필수소비재",
    "02": "필수소비재",
    "03": "필수소비재",
    "10": "필수소비재",
    "11": "필수소비재",
    "12": "필수소비재",
    "13": "경기소비재",
    "14": "경기소비재",
    "15": "경기소비재",
    "32": "경기소비재",
    "33": "경기소비재",
    "45": "경기소비재",
    "46": "경기소비재",  # wholesale incl. trading houses
    "47": "경기소비재",
    "55": "경기소비재",
    "56": "경기소비재",
    "752": "경기소비재",  # travel agencies
    "85": "경기소비재",
    "91": "경기소비재",  # sports, amusement, casinos
    "96": "경기소비재",
    # materials / energy
    "20": "화학",
    "22": "화학",
    "16": "철강·소재",
    "17": "철강·소재",
    "23": "철강·소재",
    "24": "철강·소재",
    "25": "철강·소재",
    "07": "철강·소재",
    "08": "철강·소재",
    "05": "에너지",
    "06": "에너지",
    "19": "에너지",
    # industrials
    "29": "산업재",
    "31": "산업재",  # shipbuilding, aerospace, rail
    "34": "산업재",
    "70": "산업재",
    "71": "산업재",
    "72": "산업재",
    "73": "산업재",
    "74": "산업재",
    "75": "산업재",
    "76": "산업재",
    "41": "건설",
    "42": "건설",
    "49": "운송",
    "50": "운송",
    "51": "운송",
    "52": "운송",
    # financials / utilities
    "64": "금융",  # banks, holding companies (KSIC 649 기타 금융업 incl. 지주회사)
    "65": "금융",
    "66": "금융",
    "68": "금융",  # real estate
    "35": "유틸리티",
    "36": "유틸리티",
    "37": "유틸리티",
    "38": "유틸리티",
    "39": "유틸리티",
}


def ksic_to_sector(code: Any) -> str | None:
    """Broad sector of a KSIC code ('261', '26', '2612' ...), None when unknown."""
    s = "".join(ch for ch in str(code or "") if ch.isdigit())
    for k in (3, 2):
        if len(s) >= k and s[:k] in KSIC_TO_SECTOR:
            return KSIC_TO_SECTOR[s[:k]]
    return None


def read_bulk_industries(zip_paths: list[Path]) -> pd.DataFrame:
    """code, period_end, ksic, ksic_name of every report in OpenDART bulk BS zips (header columns only)."""
    rows: dict[tuple[str, str], tuple[str, str]] = {}
    for p in zip_paths:
        try:
            z = zipfile.ZipFile(p)
        except (OSError, zipfile.BadZipFile) as exc:
            log.warning("bulk zip %s unreadable: %s", p.name, exc)
            continue
        with z:
            for info in z.infolist():
                text = z.read(info).decode("cp949", errors="replace")
                rdr = csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE)
                header = [c.strip() for c in next(rdr, [])]
                if len(header) < 8 or header[4] != "업종" or header[1] != "종목코드":
                    continue
                for r in rdr:
                    if len(r) < 8:
                        continue
                    key = (r[1].strip().strip("[]"), r[7].strip())
                    if key not in rows:
                        rows[key] = (r[4].strip(), r[5].strip())
    if not rows:
        return pd.DataFrame(columns=["code", "period_end", "ksic", "ksic_name"])
    df = pd.DataFrame(
        [(c, pe, k, n) for (c, pe), (k, n) in rows.items()], columns=["code", "period_end", "ksic", "ksic_name"]
    )
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    return df.dropna(subset=["period_end"]).sort_values(["code", "period_end"]).reset_index(drop=True)


def kr_bulk_industries(settings: Settings, cache: DiskCache) -> pd.DataFrame:
    """KSIC history per stock code from the cached OpenDART bulk zips (cached per zip set)."""
    from quantcio.data.pit.kr import kr_root

    zdir = kr_root(settings) / "dart_bulk" / "zips"
    paths = sorted(zdir.glob("*_BS_*.zip")) if zdir.exists() else []
    if not paths:
        return pd.DataFrame(columns=["code", "period_end", "ksic", "ksic_name"])
    key = DiskCache.key("kr_bulk_industries", SECTOR_VERSION, [p.name for p in paths])
    hit = cache.get("backtest_sectors", key, None)
    if isinstance(hit, pd.DataFrame):
        return hit
    df = read_bulk_industries(paths)
    cache.set("backtest_sectors", key, df)
    return df


def kind_listing_industries(settings: Settings, cache: DiskCache) -> pd.DataFrame:
    """KIND 상장법인목록 (current listing): code, industry (KSIC name). Cached daily; empty frame on failure."""
    from datetime import date

    key = DiskCache.key("kind_listing", date.today().isoformat())
    hit = cache.get("backtest_sectors", key, 24.0)
    if isinstance(hit, pd.DataFrame):
        return hit
    try:
        from quantcio.data.http import PoliteSession

        session = PoliteSession(timeout=30)
        frames = []
        for mkt in ("stockMkt", "kosdaqMkt"):
            r = session.get(
                "https://kind.krx.co.kr/corpgeneral/corpList.do",
                params={"method": "download", "searchType": "13", "marketType": mkt},
            )
            r.raise_for_status()
            t = pd.read_html(io.StringIO(r.content.decode("euc-kr", errors="replace")), converters={"종목코드": str})[0]
            frames.append(t)
        df = pd.concat(frames, ignore_index=True).rename(columns={"종목코드": "code", "업종": "industry"})
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[["code", "industry"]].drop_duplicates("code", keep="last").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001 - optional fallback source
        log.warning("KIND listing fetch failed: %s", exc)
        return pd.DataFrame(columns=["code", "industry"])
    cache.set("backtest_sectors", key, df)
    return df


def kr_sector_panel(
    pairs: pd.DataFrame,
    bulk: pd.DataFrame,
    kind: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sector of every (date, entity) pair (KR). ``bulk``: code, period_end, ksic, ksic_name (kr_bulk_industries);
    ``kind``: code, industry (current listing, fallback). Returns date, entity, sector, sector_source
    ('dart_ksic' | 'dart_ksic_backfill' | 'kind_current' | 'none')."""
    out = pairs[["date", "entity"]].copy()
    out["code"] = out["entity"].astype(str).str.split("#").str[0]
    out["sector"] = None
    out["sector_source"] = "none"
    if len(bulk):
        b = bulk.copy()
        b["sector"] = b["ksic"].map(ksic_to_sector)
        b = b.dropna(subset=["sector"])
        # a report is usable once filed: period end + REPORT_LAG_DAYS (conservative; annual reports within 90 days)
        b = b.assign(avail=b["period_end"] + pd.Timedelta(REPORT_LAG_DAYS, "D")).sort_values("avail")
        left = out.reset_index().sort_values("date")
        asof = pd.merge_asof(
            left,
            b[["code", "avail", "sector"]].rename(columns={"sector": "s_asof"}),
            left_on="date",
            right_on="avail",
            by="code",
            direction="backward",
        ).set_index("index")
        first = b.groupby("code")["sector"].first()
        s_asof = asof["s_asof"].reindex(out.index)
        has = s_asof.notna()
        out.loc[has, "sector"] = s_asof[has]
        out.loc[has, "sector_source"] = "dart_ksic"
        back = ~has & out["code"].isin(first.index)
        out.loc[back, "sector"] = out.loc[back, "code"].map(first)
        out.loc[back, "sector_source"] = "dart_ksic_backfill"
    if kind is not None and len(kind):
        name_to_ksic = {}
        if len(bulk):
            for k, n in zip(bulk["ksic"], bulk["ksic_name"], strict=False):
                name_to_ksic.setdefault(str(n).replace(" ", ""), k)
        ind = kind.set_index("code")["industry"]
        miss = out["sector"].isna() & out["code"].isin(ind.index)
        mapped = (
            out.loc[miss, "code"]
            .map(ind)
            .map(lambda n: ksic_to_sector(name_to_ksic.get(str(n).replace(" ", ""))) if isinstance(n, str) else None)
        )
        ok = mapped.notna()
        out.loc[mapped.index[ok], "sector"] = mapped[ok]
        out.loc[mapped.index[ok], "sector_source"] = "kind_current"
    out["sector"] = out["sector"].fillna(OTHER)
    return out.drop(columns=["code"])


def sector_panel(
    market: Any,
    pairs: pd.DataFrame,
    settings: Settings,
    cache: DiskCache,
    meta: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """(date, entity, sector, sector_source) for every membership pair of ``market`` + Korean notes."""
    from quantcio.models import Market

    notes: list[str] = []
    if Market(market) is Market.US:
        ents = sorted(pairs["entity"].unique())
        sec, src, n = us_entity_sectors(settings, ents)
        notes += n
        out = pairs[["date", "entity"]].copy()
        out["sector"] = out["entity"].map(sec).fillna(OTHER)
        out["sector_source"] = out["entity"].map(src).fillna("none")
        return out, notes
    bulk = kr_bulk_industries(settings, cache)
    if bulk.empty:
        notes.append("DART 일괄 재무 파일(업종 코드) 없음 → KIND 현재 업종만 사용")
    out = kr_sector_panel(pairs, bulk, None)
    if (out["sector_source"] == "none").any():
        kind = kind_listing_industries(settings, cache)
        if len(kind):
            out = kr_sector_panel(pairs, bulk, kind)
    return out, notes


def sector_coverage(sp: pd.DataFrame) -> dict[str, Any]:
    """Counts of sector sources and unclassified ("기타") member-months by year."""
    by_year: dict[str, dict[str, int]] = {}
    for y, g in sp.groupby(sp["date"].dt.year):
        c = Counter(g["sector_source"])
        by_year[str(int(y))] = {
            "member_months": len(g),
            "unclassified": int((g["sector_source"] == "none").sum()),
            **{str(k): int(v) for k, v in c.items()},
        }
    return {
        "sources": {str(k): int(v) for k, v in Counter(sp["sector_source"]).items()},
        "unclassified_member_months": int((sp["sector_source"] == "none").sum()),
        "unclassified_entities": sorted(sp.loc[sp["sector_source"] == "none", "entity"].unique().tolist()),
        "by_year": by_year,
        "sectors": {str(k): int(v) for k, v in Counter(sp["sector"]).items()},
    }
