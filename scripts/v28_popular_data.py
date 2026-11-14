"""Everyday attention, consumption and public-investment data (docs/MODELING.md 69, pre-declared 2026-10-19).
Research only; keys only from .env (never printed).

    .venv/bin/python scripts/v28_popular_data.py fetch --what all     # or naver_search, naver_shop, wiki_top,
                                                                      #   wiki_topics, us_soc, census

History available (no new key): Naver DataLab search trend daily (2016-) for company names and theme keywords, Naver
Shopping Insight daily category clicks (2017-08-), Wikipedia daily top-1,000 articles (en / ko, 2015-07-) and topic
article pageviews, USAspending monthly obligations by programme / NAICS (FY2008-), Census monthly public construction
(VIP) and retail sales (MARTS) by type. Books / box office / YouTube / KONEPS contracts need keys (docs 69.1).
Files under <cache>/alt/keyed/v28/.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402

say = v13.say
HUB = "https://naverapihub.apigw.ntruss.com"
WIKI = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
WIKI_UA = {"User-Agent": "quantcio-research/1.0 (academic research script)"}
SEARCH_START, SHOP_START, WIKI_START = "2016-01-01", "2017-08-01", date(2015, 7, 1)
THEMES = [  # fixed before any result (69.2)
    "반도체",
    "2차전지",
    "배터리",
    "전기차",
    "인공지능",
    "바이오",
    "신약",
    "방산",
    "조선",
    "원전",
    "화장품",
    "게임",
    "엔터",
    "아이돌",
    "드라마",
    "영화",
    "여행",
    "항공",
    "면세점",
    "부동산",
    "아파트",
    "금리",
    "환율",
    "주식",
    "비트코인",
    "공매도",
    "수소",
    "태양광",
    "로봇",
    "우주",
]
SHOP_CATS = {
    "50000000": "패션의류",
    "50000001": "패션잡화",
    "50000002": "화장품/미용",
    "50000003": "디지털/가전",
    "50000004": "가구/인테리어",
    "50000005": "출산/육아",
    "50000006": "식품",
    "50000007": "스포츠/레저",
    "50000008": "생활/건강",
    "50000009": "여가/생활편의",
    "50000010": "면세점",
    "50005542": "도서",
}
WIKI_TOPICS = {
    "en.wikipedia": [
        "Artificial_intelligence",
        "Semiconductor",
        "Electric_vehicle",
        "Lithium-ion_battery",
        "Cryptocurrency",
        "Bitcoin",
        "Recession",
        "Inflation",
        "Interest_rate",
        "Tariff",
        "Oil",
        "Natural_gas",
        "Gold",
        "Housing_bubble",
        "Stock_market_crash",
        "Federal_Reserve",
        "Unemployment",
        "Pandemic",
        "Obesity",
        "Semaglutide",
        "Cloud_computing",
        "Data_center",
        "Nuclear_power",
        "Solar_power",
        "Defense_industry",
        "Video_game",
        "Streaming_media",
        "Box_office",
        "Black_Friday_(shopping)",
        "Cyber_Monday",
    ],
    "ko.wikipedia": [
        "인공지능",
        "반도체",
        "전기자동차",
        "리튬_이온_전지",
        "비트코인",
        "인플레이션",
        "금리",
        "환율",
        "부동산",
        "공매도",
        "원자력_발전",
        "조선업",
        "방위산업",
        "케이팝",
        "대한민국의_드라마",
        "넷플릭스",
        "유튜브",
        "면세점",
        "화장품",
        "비만",
    ],
}
US_SOC = {  # USAspending filters per public-investment type (69.1)
    "highway": {"program_numbers": ["20.205"]},
    "transit": {"program_numbers": ["20.500", "20.507"]},
    "airport": {"program_numbers": ["20.106"]},
    "rail": {"program_numbers": ["20.314", "20.319"]},
    "water": {"program_numbers": ["66.458", "66.468"]},
    "c_road_bridge": {"naics_codes": {"require": ["237310"]}},
    "c_public_building": {"naics_codes": {"require": ["236220"]}},
    "c_water_sewer": {"naics_codes": {"require": ["237110"]}},
    "c_power_comm_line": {"naics_codes": {"require": ["237130"]}},
    "c_other_civil": {"naics_codes": {"require": ["237990"]}},
}
VIP_TYPES = {  # Census VIP category code -> construction type (data type P = public); codes matched to C30 by value
    "XXXX": "total",
    "NRXX": "nonresidential",
    "00XX": "residential",
    "02XX": "office",
    "04XX": "health",
    "05XX": "educational",
    "07XX": "public_safety",
    "08XX": "amusement",
    "09XX": "transportation",
    "11XX": "power",
    "12XX": "highway_street",
    "13XX": "sewage_waste",
    "14XX": "water_supply",
    "15XX": "conservation",
}
MARTS = ["44X72", "441", "442", "443", "444", "445", "446", "447", "448", "451", "452", "453", "454", "722"]


def root(settings) -> Path:
    return kd.root(settings, "v28")


def _hub_post(path: str, body: dict) -> dict:
    cid, sec = kd.key("NAVER_CLIENT_ID"), kd.key("NAVER_CLIENT_SECRET")
    h = {"X-NCP-APIGW-API-KEY-ID": cid, "X-NCP-APIGW-API-KEY": sec, "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            r = requests.post(HUB + path, json=body, headers=h, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(30)
            continue
        if r.status_code != 200:
            raise RuntimeError(kd._redact(f"Naver {path} HTTP {r.status_code} {r.text[:120]}", cid, sec))
        time.sleep(0.2)
        return r.json()
    raise RuntimeError(f"Naver {path}: 응답 없음")


# ================================================================================================ Naver


def _search_groups(groups: list[tuple[str, str]], kind: str) -> pd.DataFrame:
    rows, end = [], date.today().strftime("%Y-%m-%d")
    for i in range(0, len(groups), 5):
        part = groups[i : i + 5]
        body = {
            "startDate": SEARCH_START,
            "endDate": end,
            "timeUnit": "date",
            "keywordGroups": [{"groupName": g, "keywords": [kw]} for g, kw in part],
        }
        js = _hub_post("/search-trend/v1/search", body)
        for g in js.get("results", []):
            rows += [
                {"key": g["title"], "date": x["period"], "ratio": float(x["ratio"]), "req": i // 5} for x in g["data"]
            ]
        if (i // 5) % 20 == 0:
            say(f"Naver search ({kind}) {min(i + 5, len(groups))}/{len(groups)}")
    t = pd.DataFrame(rows)
    t["date"] = pd.to_datetime(t["date"])
    return t


def fetch_naver_search(settings, cache) -> dict:
    """Daily relative search volume: every KR universe company name, and the THEMES keywords (5 per request; each
    request is scaled to its own maximum, so only within-series ratios are used)."""
    if not kd.key("NAVER_CLIENT_ID"):
        return {"skipped": "NAVER_CLIENT_ID 없음"}
    uni = v16.universe("KR", settings, cache)
    comp = _search_groups([(str(e), str(n)) for e, n in zip(uni["entity"], uni["name"], strict=True) if n], "회사명")
    comp.to_parquet(root(settings) / "naver_search_company.parquet", index=False)
    th = _search_groups([(k, k) for k in THEMES], "테마")
    th.to_parquet(root(settings) / "naver_search_theme.parquet", index=False)
    return {
        "companies": int(comp["key"].nunique()),
        "themes": int(th["key"].nunique()),
        "days": int(comp["date"].nunique()),
    }


def fetch_naver_shop(settings) -> dict:
    """Daily click share per first-level shopping category, one category per request (own scale)."""
    if not kd.key("NAVER_CLIENT_ID"):
        return {"skipped": "NAVER_CLIENT_ID 없음"}
    rows, end = [], date.today().strftime("%Y-%m-%d")
    for code, name in SHOP_CATS.items():
        body = {
            "startDate": SHOP_START,
            "endDate": end,
            "timeUnit": "date",
            "category": [{"name": name, "param": [code]}],
        }
        js = _hub_post("/shopping/v1/categories", body)
        for g in js.get("results", []):
            rows += [
                {"code": code, "category": name, "date": x["period"], "ratio": float(x["ratio"])} for x in g["data"]
            ]
    t = pd.DataFrame(rows)
    t["date"] = pd.to_datetime(t["date"])
    t.to_parquet(root(settings) / "naver_shop.parquet", index=False)
    return {"categories": int(t["code"].nunique()), "days": int(t["date"].nunique())}


# ================================================================================================ Wikipedia


def _wiki_get(sess: requests.Session, url: str) -> requests.Response | None:
    """GET with back-off on 429 / 5xx; None after the retries (the caller must not store a gap)."""
    for attempt in range(8):
        try:
            r = sess.get(url, headers=WIKI_UA, timeout=60)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(10 * 2**attempt, 300))
            continue
        return r
    return None


def fetch_wiki_top(settings) -> dict:
    """Top 1,000 articles of every day (en, ko), one parquet per project-month (resumable)."""
    sess, n = requests.Session(), 0
    d = root(settings) / "wiki_top"
    d.mkdir(parents=True, exist_ok=True)
    last = date.today() - timedelta(2)
    for proj in ("en.wikipedia", "ko.wikipedia"):
        m = WIKI_START.replace(day=1)
        while m <= last:
            p = d / f"{proj.split('.')[0]}_{m:%Y-%m}.parquet"
            nxt = (pd.Timestamp(m) + pd.offsets.MonthBegin(1)).date()
            complete = nxt <= last
            have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["date"])
            got = set(pd.to_datetime(have["date"]).dt.date)
            want = [x for x in pd.date_range(max(m, WIKI_START), min(nxt - timedelta(1), last)).date if x not in got]
            if complete and not want:
                m = nxt
                continue
            rows, stop = [], None
            for day in want:
                r = _wiki_get(sess, f"{WIKI}/top/{proj}/all-access/{day:%Y/%m/%d}")
                if r is None:
                    stop = f"Wikipedia top {proj} {day}: 응답 없음 — 다시 실행하면 빠진 날부터 이어 받음"
                    break
                if r.ok:
                    for a in r.json()["items"][0]["articles"]:
                        rows.append({"date": day, "rank": a["rank"], "article": a["article"], "views": a["views"]})
                n += 1
                time.sleep(1.0)  # the top endpoint rate-limits hard (429 at 4 requests / second)
            if rows:  # partial months are kept; missing days are fetched on the next run
                new = pd.DataFrame(rows).assign(date=lambda x: pd.to_datetime(x["date"]))
                pd.concat([have, new], ignore_index=True).to_parquet(p, index=False)
            if stop:
                raise RuntimeError(stop)
            m = nxt
        say(f"Wikipedia top {proj}: done ({n} requests)")
    return {"requests": n}


def fetch_wiki_topics(settings) -> dict:
    sess, rows = requests.Session(), []
    end = (date.today() - timedelta(1)).strftime("%Y%m%d")
    for proj, arts in WIKI_TOPICS.items():
        for a in arts:
            url = f"{WIKI}/per-article/{proj}/all-access/user/{quote(a, safe='')}/daily/20150701/{end}"
            r = _wiki_get(sess, url)
            time.sleep(1.0)
            if r is None or not r.ok:
                say(f"Wikipedia {proj} {a}: HTTP {getattr(r, 'status_code', 'none')}")
                continue
            rows += [
                {"project": proj, "article": a, "date": x["timestamp"][:8], "views": x["views"]}
                for x in r.json()["items"]
            ]
    t = pd.DataFrame(rows)
    t["date"] = pd.to_datetime(t["date"], format="%Y%m%d")
    t.to_parquet(root(settings) / "wiki_topics.parquet", index=False)
    return {"articles": int(t.groupby(["project", "article"]).ngroups), "days": int(t["date"].nunique())}


# ================================================================================================ US public investment


def fetch_us_soc(settings) -> dict:
    rows, today = [], date.today().strftime("%Y-%m-%d")
    for kind, flt in US_SOC.items():
        body = {"group": "month", "filters": {**flt, "time_period": [{"start_date": "2007-10-01", "end_date": today}]}}
        js = v16._usa_post("/search/spending_over_time/", body)
        time.sleep(0.5)
        for q in js.get("results") or []:
            tp = q.get("time_period") or {}
            fy, mo = int(tp.get("fiscal_year") or 0), int(tp.get("month") or 0)
            if fy and mo:
                cal = pd.Timestamp(fy - 1, 10, 1) + pd.DateOffset(months=mo - 1)  # fiscal month 1 = October
                rows.append({"kind": kind, "month": cal, "obligations": float(q.get("aggregated_amount") or 0.0)})
    t = pd.DataFrame(rows)
    t.to_parquet(root(settings) / "us_soc_monthly.parquet", index=False)
    return {"kinds": int(t["kind"].nunique()), "months": int(t["month"].nunique())}


def _census(ds: str, get: str, since: str) -> pd.DataFrame:
    k = kd.key("CENSUS_API_KEY")
    r = requests.get(
        f"https://api.census.gov/data/timeseries/eits/{ds}",
        params={"get": get, "for": "us:*", "time": f"from {since}", "key": k},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(kd._redact(f"Census {ds} HTTP {r.status_code} {r.text[:120]}", k))
    rows = r.json()
    return pd.DataFrame(rows[1:], columns=rows[0])


def fetch_census(settings) -> dict:
    vip = _census("vip", "cell_value,data_type_code,category_code,seasonally_adj,time_slot_id", "2008-01")
    code = vip["category_code"].str.removeprefix("A")  # seasonally adjusted annual rates carry an "A" prefix
    vip = vip[
        (vip["data_type_code"] == "P") & (vip["seasonally_adj"] == "yes") & vip["category_code"].str.startswith("A")
    ]
    vip = vip.assign(kind=code[vip.index].map(VIP_TYPES), value=pd.to_numeric(vip["cell_value"], errors="coerce"))
    vip = vip.dropna(subset=["kind"])
    vip.to_parquet(root(settings) / "census_vip_public.parquet", index=False)
    mr = _census("marts", "cell_value,data_type_code,category_code,seasonally_adj,time_slot_id", "2008-01")
    mr = mr[(mr["data_type_code"] == "SM") & mr["category_code"].isin(MARTS) & (mr["seasonally_adj"] == "yes")]
    mr = mr.assign(value=pd.to_numeric(mr["cell_value"], errors="coerce"))
    mr.to_parquet(root(settings) / "census_marts.parquet", index=False)
    return {
        "vip_types": int(vip["kind"].nunique()),
        "vip_months": int(vip["time"].nunique()),
        "marts_categories": int(mr["category_code"].nunique()),
    }


# ================================================================================================ keyed sources (69.1)

KOBIS = "https://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json"
LIBRARY = "https://data4library.kr/api/loanItemSrch"
NYT = "https://api.nytimes.com/svc/books/v3/lists/overview.json"
YOUTUBE = "https://www.googleapis.com/youtube/v3/videos"
KONEPS = "https://apis.data.go.kr/1230000/ao/CntrctInfoService/getCntrctInfoListCnstwk"


def _get_json(url: str, params: dict, secret: str, gap: float) -> dict | None:
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers={"User-Agent": WIKI_UA["User-Agent"]}, timeout=90)
        except requests.RequestException:
            time.sleep(5 * (attempt + 1))
            continue
        time.sleep(gap)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(30 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(kd._redact(f"HTTP {r.status_code} {r.text[:160]}", secret))
        return r.json()
    return None


def _daily_files(name: str, settings, first: date, last: date, fetch_day) -> dict:
    """One parquet per year of daily rows; days already stored are skipped (resumable across daily quotas)."""
    d = root(settings) / name
    d.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for y in range(first.year, last.year + 1):
        p = d / f"{y}.parquet"
        have = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        done = set(pd.to_datetime(have["date"]).dt.date) if len(have) else set()
        rows, day = [], max(first, date(y, 1, 1))
        while day <= min(last, date(y, 12, 31)):
            if day not in done:
                got = fetch_day(day)
                if got is None:  # quota or outage: keep what we have, resume next run
                    break
                rows += [{"date": pd.Timestamp(day), **x} for x in got] or [{"date": pd.Timestamp(day)}]
                n_new += 1
            day += timedelta(1)
        if rows:
            pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(p, index=False)
        if day <= min(last, date(y, 12, 31)):
            break
    return {"days_new": n_new}


def fetch_kobis(settings) -> dict:
    """KOBIS daily box office top 10 (2004-), KOBIS_API_KEY (kobis.or.kr open API, 3,000 calls / day)."""
    k = kd.key("KOBIS_API_KEY")
    if not k:
        return {"skipped": "KOBIS_API_KEY 없음 (kobis.or.kr/kobisopenapi 회원가입 후 키 발급)"}

    def day(dd: date):
        js = _get_json(KOBIS, {"key": k, "targetDt": f"{dd:%Y%m%d}"}, k, 0.4)
        if js is None or "faultInfo" in js:
            return None
        return [
            {
                "rank": int(x["rank"]),
                "movie": x["movieNm"],
                "movie_cd": x["movieCd"],
                "open": x.get("openDt"),
                "sales": float(x["salesAmt"] or 0),
                "audience": float(x["audiCnt"] or 0),
                "screens": float(x.get("scrnCnt") or 0),
            }
            for x in js["boxOfficeResult"]["dailyBoxOfficeList"]
        ]

    return _daily_files("kobis", settings, date(2004, 1, 1), date.today() - timedelta(1), day)


def fetch_library(settings) -> dict:
    """도서관 정보나루 popular loans per day (top 100), LIBRARY_API_KEY (data4library.kr)."""
    k = kd.key("LIBRARY_API_KEY")
    if not k:
        return {"skipped": "LIBRARY_API_KEY 없음 (data4library.kr 인증키 신청)"}

    def day(dd: date):
        js = _get_json(
            LIBRARY, {"authKey": k, "startDt": f"{dd}", "endDt": f"{dd}", "format": "json", "pageSize": 100}, k, 0.5
        )
        if js is None:
            return None
        docs = (js.get("response") or {}).get("docs") or []
        return [
            {
                "rank": int(x["doc"]["ranking"]),
                "title": x["doc"]["bookname"],
                "isbn": x["doc"].get("isbn13"),
                "kdc": x["doc"].get("class_no"),
                "loans": float(x["doc"].get("loan_count") or 0),
            }
            for x in docs
        ]

    return _daily_files("library", settings, date(2015, 1, 1), date.today() - timedelta(2), day)


def fetch_nyt(settings) -> dict:
    """NYT Best Sellers, every weekly overview (2008-06-), NYT_API_KEY (developer.nytimes.com, Books API)."""
    k = kd.key("NYT_API_KEY")
    if not k:
        return {"skipped": "NYT_API_KEY 없음 (developer.nytimes.com 앱 등록 후 Books API 사용)"}
    d = root(settings) / "nyt"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "overview.parquet"
    have = pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["published"])
    done = set(have["published"].astype(str))
    rows, wk = [], date(2008, 6, 8)
    while wk <= date.today():
        if str(wk) not in done:
            js = _get_json(NYT, {"published_date": f"{wk}", "api-key": k}, k, 12.5)  # 5 calls / minute
            if js is None:
                break
            res = js.get("results") or {}
            for lst in res.get("lists") or []:
                for b in lst.get("books") or []:
                    rows.append(
                        {
                            "published": res.get("published_date") or str(wk),
                            "list": lst.get("list_name"),
                            "rank": b.get("rank"),
                            "title": b.get("title"),
                            "publisher": b.get("publisher"),
                            "weeks": b.get("weeks_on_list"),
                        }
                    )
        wk += timedelta(7)
    if rows:
        pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_parquet(p, index=False)
    return {"rows_new": len(rows)}


def fetch_koneps(settings) -> dict:
    """조달청 나라장터 construction contracts per day (DATA_GO_KR_KEY with the 계약정보서비스 활용신청)."""
    k = kd.key("DATA_GO_KR_KEY")
    if not k:
        return {"skipped": "DATA_GO_KR_KEY 없음"}

    def day(dd: date):
        out, page = [], 1
        while True:
            js = _get_json(
                KONEPS,
                {
                    "ServiceKey": k,
                    "inqryDiv": "1",
                    "inqryBgnDt": f"{dd:%Y%m%d}0000",
                    "inqryEndDt": f"{dd:%Y%m%d}2359",
                    "numOfRows": "999",
                    "pageNo": str(page),
                    "type": "json",
                },
                k,
                0.3,
            )
            if js is None:
                return None
            body = (js.get("response") or {}).get("body") or {}
            items = body.get("items") or []
            items = items.get("item", []) if isinstance(items, dict) else items
            out += [
                {
                    "name": x.get("cnstwkNm"),
                    "agency": x.get("dminsttNm"),
                    "amount": x.get("thtmCntrctAmt"),
                    "kind": x.get("cntrctCnclsMthdNm"),
                    "region": x.get("cnstwkLctnm"),
                }
                for x in items
            ]
            if page * 999 >= int(body.get("totalCount") or 0):
                return out
            page += 1

    return _daily_files("koneps", settings, date(2015, 1, 1), date.today() - timedelta(1), day)


def youtube_ledger(settings) -> dict:
    """Today's YouTube most-popular chart (KR, US; 50 videos) -> cache (forward only), YOUTUBE_API_KEY."""
    k = kd.key("YOUTUBE_API_KEY")
    if not k:
        return {"skipped": "YOUTUBE_API_KEY 없음 (Google Cloud 콘솔에서 YouTube Data API v3 키 발급)"}
    rows = []
    for region in ("KR", "US"):
        js = _get_json(
            YOUTUBE,
            {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region, "maxResults": 50, "key": k},
            k,
            0.5,
        )
        for i, v in enumerate((js or {}).get("items") or [], 1):
            sn, st = v.get("snippet") or {}, v.get("statistics") or {}
            rows.append(
                {
                    "date": str(date.today()),
                    "region": region,
                    "rank": i,
                    "title": sn.get("title"),
                    "channel": sn.get("channelTitle"),
                    "category": sn.get("categoryId"),
                    "views": float(st.get("viewCount") or 0),
                }
            )
    d = root(settings) / "youtube"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(d / f"{date.today()}.parquet", index=False)
    return {"videos": len(rows)}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    names = [
        "naver_search",
        "naver_shop",
        "wiki_topics",
        "us_soc",
        "census",
        "wiki_top",
        "kobis",
        "library",
        "nyt",
        "koneps",
        "youtube",
    ]
    f.add_argument("--what", default="all", choices=["all", *names])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    steps = {
        "naver_search": lambda: fetch_naver_search(s, cache),
        "naver_shop": lambda: fetch_naver_shop(s),
        "wiki_topics": lambda: fetch_wiki_topics(s),
        "us_soc": lambda: fetch_us_soc(s),
        "census": lambda: fetch_census(s),
        "wiki_top": lambda: fetch_wiki_top(s),
        "kobis": lambda: fetch_kobis(s),
        "library": lambda: fetch_library(s),
        "nyt": lambda: fetch_nyt(s),
        "koneps": lambda: fetch_koneps(s),
        "youtube": lambda: youtube_ledger(s),
    }
    secrets = [
        kd.key(x)
        for x in (
            "NAVER_CLIENT_ID",
            "NAVER_CLIENT_SECRET",
            "CENSUS_API_KEY",
            "KOBIS_API_KEY",
            "LIBRARY_API_KEY",
            "NYT_API_KEY",
            "DATA_GO_KR_KEY",
            "YOUTUBE_API_KEY",
        )
    ]
    for name in names:
        if a.what in ("all", name):
            try:
                say(f"{name}: {steps[name]()}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *secrets)}")


if __name__ == "__main__":
    main()
