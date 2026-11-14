"""CEO / vision / strategy features (docs/MODELING.md 63, pre-declared 2026-10-16). Research only.

KR: CEO identity, change, age, tenure, owner-family relation and education (school tier, degree, major) from the
OpenDART executive rosters; strategy themes and year-over-year change of the annual report's business section.
US: CEO tenure, age and school / degree from Wikidata terms; tone and strategy themes of the CEO quotes inside the
8-K earnings releases (scripts/v22_tone.py texts). Features are 0/1 or numbers only - names never leave the cache.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lexicons as lx  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v22_tone as tone  # noqa: E402
import v24_ceo_data as cd  # noqa: E402

MAX_AGE = 550


def _rx(words: str, flags: int = 0) -> re.Pattern:
    """A pattern from a space-separated word list (alternatives; regex syntax allowed inside a word)."""
    return re.compile("|".join(words.split()), flags)


UNIV = _rx(
    "대학 大學 University Univ\\. College KAIST 카이스트 POSTECH 포항공 포스텍 MIT MBA 박사 석사 학사 Ph\\.?D", re.I
)
SKY = _rx("서울대 연세대 고려대 Seoul\\sNational Yonsei Korea\\sUniversity")
KP = _rx("KAIST 카이스트 한국과학기술원 POSTECH 포항공 포스텍", re.I)
FOREIGN = _rx(
    "미국 영국 일본 독일 프랑스 캐나다 호주 중국 University Univ\\. College MIT Stanford Harvard Wharton Columbia"
    " Cornell Yale Princeton Berkeley UCLA USC NYU Duke Chicago Northwestern Michigan Purdue Illinois Texas"
    " 도쿄 동경 와세다 게이오 교토 스탠 하버드 와튼 콜롬비아 컬럼비아 코넬 예일 버클리"
)
PHD = _rx("박사 Ph\\.?\\s?D Doctor", re.I)
MASTER = _rx("석사 Master M\\.S\\. MBA 경영전문대학원", re.I)
MBA = _rx("MBA 경영전문대학원 경영대학원", re.I)
MAJOR = {
    "ceo_major_eng": _rx(
        "공학 공과 전자 전기 기계 화공 재료 금속 컴퓨터 산업공학 건축 토목 조선"
        " 물리 화학 수학 자연과학 이학 Engineering",
        re.I,
    ),
    "ceo_major_biz": _rx("경영 경제 회계 무역 상학 금융 Business Economics Finance Accounting", re.I),
    "ceo_major_law": _rx("법학 법대 법률 Law", re.I),
    "ceo_major_med": _rx("의학 의과 약학 생명 생물 수의 간호 Medicine Pharm Biology", re.I),
}
KIN = _rx(
    "본인 친인척 자녀 아들 딸 배우자 형제 자매 부친 모친 손자 손녀 사위 며느리 조카"
    " 장남 차남 삼남 장녀 차녀 숙부 동생 남편 ^자$ 의자$ 의子"
)
NOT_OWNER = _rx("임원 특수관계 직원 해당 없음")
KR_THEMES = {
    "ai": r"인공지능|\bAI\b|머신러닝|딥러닝|빅데이터|디지털\s?전환|데이터센터|클라우드|생성형",
    "semi": r"반도체|파운드리|메모리|HBM|웨이퍼",
    "battery": r"2차전지|이차전지|배터리|전기차|양극재|음극재|전해질|분리막",
    "bio": r"바이오|신약|임상|헬스케어|의약품|백신|진단",
    "green": r"친환경|ESG|탄소\s?중립|수소|재생에너지|태양광|풍력|순환경제",
    "global": r"해외|글로벌|수출|현지법인",
    "newbiz": r"신사업|신성장|성장동력|신규\s?사업|미래\s?사업|다각화",
    "restructure": r"구조조정|효율화|원가\s?절감|비용\s?절감|합리화",
    "shareholder": r"주주환원|배당\s?확대|자사주|주주가치",
    "ma": r"인수|합병|M&A|지분\s?투자|전략적\s?투자|합작",
    "rnd": r"연구개발|R&D|특허|기술\s?개발|기술력",
}
US_THEMES = {
    "ai": r"artificial intelligence|\bAI\b|machine learning|generative",
    "growth": r"growth|expand|expansion|momentum|accelerat",
    "innovation": r"innovat|new product|launch|pipeline",
    "margin": r"margin|efficien|cost|productivity|discipline",
    "restructure": r"restructur|transformation|realign|simplif|turnaround",
    "capital": r"buyback|repurchase|dividend|return(?:ing)? capital|shareholder",
    "guidance_up": r"rais(?:e|ed|ing)[^.]{0,40}(?:guidance|outlook)|increas(?:e|ed|ing)[^.]{0,40}(?:guidance|outlook)",
    "guidance_dn": r"(?:lower(?:ed|ing)?|reduc(?:e|ed|ing)|cut(?:ting)?)[^.]{0,40}(?:guidance|outlook)",
}
IVY = _rx(
    "Harvard Yale Princeton Columbia Pennsylvania Wharton Brown\\sUniversity Dartmouth Cornell Stanford"
    " Massachusetts\\sInstitute\\sof\\sTechnology University\\sof\\sChicago Duke Northwestern"
    " California\\sInstitute\\sof\\sTechnology"
)
US_MBA = _rx(
    "Business\\sAdministration Business\\sSchool School\\sof\\sBusiness Wharton Kellogg Booth Sloan Tuck Haas"
    " Fuqua Stern Ross\\sSchool Darden Anderson\\sSchool INSEAD"
)
US_ENG = _rx("Institute\\sof\\sTechnology Engineering Polytechnic Caltech")
KR_CEO_COLS = [
    "ceo_n",
    "ceo_new_12m",
    "ceo_age",
    "ceo_tenure",
    "ceo_family",
    "ceo_registered",
    "ceo_edu_known",
    "ceo_sky",
    "ceo_kaist_postech",
    "ceo_foreign",
    "ceo_phd",
    "ceo_master",
    "ceo_mba",
    *MAJOR,
    "ceo_insider",
]
KR_THEME_KO = {
    "ai": "AI·디지털",
    "semi": "반도체",
    "battery": "2차전지·전기차",
    "bio": "바이오·헬스",
    "green": "친환경·ESG",
    "global": "해외·글로벌",
    "newbiz": "신사업·신성장",
    "restructure": "구조조정·효율",
    "shareholder": "주주환원",
    "ma": "인수합병·투자",
    "rnd": "연구개발·기술",
}
DOCS: dict[str, tuple[str, str]] = {
    "ceo_n": ("대표이사 수", "number of CEOs"),
    "ceo_new_12m": ("12개월 안에 대표이사가 바뀜", "CEO changed within 12 months"),
    "ceo_age": ("대표이사 나이", "CEO age"),
    "ceo_tenure": ("대표이사 재직 연수", "CEO tenure (years)"),
    "ceo_family": ("대표가 최대주주 본인·친인척(오너 경영)", "owner-family CEO"),
    "ceo_registered": ("대표가 등기임원", "CEO is a registered director"),
    "ceo_edu_known": ("대표 학력이 공시에 적힘", "CEO education disclosed"),
    "ceo_sky": ("대표 서울대·연세대·고려대", "CEO from SNU / Yonsei / Korea Univ."),
    "ceo_kaist_postech": ("대표 KAIST·포스텍", "CEO from KAIST / POSTECH"),
    "ceo_foreign": ("대표 해외 대학", "CEO studied abroad"),
    "ceo_phd": ("대표 박사", "CEO holds a PhD"),
    "ceo_master": ("대표 석사 이상", "CEO holds a master's"),
    "ceo_mba": ("대표 MBA", "CEO holds an MBA"),
    "ceo_major_eng": ("대표 전공 공학·자연과학", "CEO major engineering / science"),
    "ceo_major_biz": ("대표 전공 경영·경제·회계", "CEO major business / economics"),
    "ceo_major_law": ("대표 전공 법학", "CEO major law"),
    "ceo_major_med": ("대표 전공 의·약·생명", "CEO major medicine / life science"),
    "ceo_insider": ("대표 경력에 그 회사 이름(내부 승진)", "CEO career inside the firm"),
    **{
        f"st_{k}": (f"사업의 내용 '{v}' 단어 비율", f"business section theme share {k}") for k, v in KR_THEME_KO.items()
    },
    **{f"st_{k}_chg": (f"'{v}' 비율 전년 대비 변화", f"theme share change {k}") for k, v in KR_THEME_KO.items()},
    "st_sim": ("사업의 내용 전년 대비 유사도(글자 3-그램 코사인)", "business section similarity vs last year"),
    "st_len_chg": ("사업의 내용 길이 변화(log)", "business section length change"),
    "ceo_wd_known": ("위키데이터에 그날 CEO 기록 있음", "Wikidata CEO term known"),
    "ceo_ivy": ("CEO 아이비리그·명문 사립", "CEO from an elite US school"),
    "ceo_mba_wd": ("CEO MBA·경영대학원", "CEO MBA"),
    "ceo_eng": ("CEO 공대", "CEO engineering school"),
    "ceo_tenure_wd": ("CEO 재직 연수(위키데이터)", "CEO tenure (Wikidata)"),
    "ceo_age_wd": ("CEO 나이(위키데이터)", "CEO age (Wikidata)"),
    "ceo_new_wd": ("CEO 취임 1년 이내", "CEO in first year"),
    "cq_len": ("CEO 인용문 길이(log 단어 수)", "CEO quote length"),
    "cq_tone": ("CEO 인용문 어조", "CEO quote tone"),
    "cq_unc": ("CEO 인용문 불확실성 단어 비율", "CEO quote uncertainty share"),
    **{f"cq_{k}": (f"CEO 인용문 '{k}' 주제 비율", f"CEO quote theme {k}") for k in US_THEMES},
    "cq_new_name": ("발표문 CEO 이름이 바뀜(새 CEO 첫 발표)", "new CEO name in releases"),
}


def _num(s, pat=r"(\d{4})") -> float:
    m = re.search(pat, str(s or ""))
    return float(m.group(1)) if m else np.nan


_D4 = re.compile(r"((?:19|20)\d{2})\s*[.\-/년]\s*(?:(\d{1,2})(?!\s*개월)\s*[.\-/월]?\s*(?:(\d{1,2})(?!\s*개월))?)?")
_D2 = re.compile(r"(?<!\d)'?(\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})")


def _tenure(h, asof: pd.Timestamp) -> float:
    """Years in office from the roster's tenure text: the earliest start date written ('YYYY.MM.DD ~', 'YY.MM.DD',
    'YYYY년 M월'), else a duration ('X년 Y개월', 'X.Y년', 'Y개월'); bare numbers are ambiguous -> NaN."""
    h = str(h or "").strip()
    starts = []
    for rx, two in ((_D4, False), (_D2, True)):
        for m in rx.finditer(h):
            y = int(m.group(1)) + ((1900 if int(m.group(1)) > 50 else 2000) if two else 0)
            try:
                starts.append(pd.Timestamp(year=y, month=int(m.group(2) or 7), day=int(m.group(3) or 1)))
            except ValueError:
                continue
    if starts:
        return max((asof - min(starts)).days / 365.25, 0.0)
    y, mo = re.search(r"(\d+(?:\.\d+)?)\s*년", h), re.search(r"(\d+)\s*(?:개월|월)", h)
    if not y and not mo:
        return np.nan
    return (float(y.group(1)) if y else 0.0) + (int(mo.group(1)) if mo else 0) / 12


def _owner(rel) -> bool:
    """CEO is the largest holder or a relative of it (owner-family management)."""
    r = re.sub(r"\s", "", str(rel or ""))
    return bool(KIN.search(r)) or (r.startswith("최대주주") and not NOT_OWNER.search(r))


def _name(nm) -> str:
    return re.sub(r"\s", "", re.split(r"[\n(]", str(nm or ""))[0])


def _nanagg(fn, xs) -> float:
    xs = [x for x in xs if x == x]
    return float(fn(xs)) if xs else np.nan


# ================================================================================================ KR CEO


def kr_ceo_table(settings, code: str) -> pd.DataFrame | None:
    p = cd.root(settings) / "exctv" / f"{code}.json"
    if not p.exists():
        return None
    rosters = json.loads(p.read_text())
    recs = []
    for key, rows in rosters.items():
        ceos = [r for r in rows if "대표" in str(r.get("ofcps"))]
        if not rows or not ceos:
            continue
        rno = str(rows[0].get("rcept_no") or "")
        if not re.match(r"^\d{8}", rno):
            continue
        y = int(key.split("_")[0])
        corp = re.sub(r"\(주\)|㈜|주식회사|\s", "", str(rows[0].get("corp_name") or ""))
        careers = [str(r.get("main_career") or "") for r in ceos]
        edu = [c for c in careers if UNIV.search(c)]
        anyf = lambda rx, xs=careers: float(any(rx.search(c) for c in xs))  # noqa: E731
        asof = pd.to_datetime(str(rows[0].get("stlm_dt") or ""), errors="coerce")
        asof = asof if pd.notna(asof) else pd.Timestamp(year=y, month=12, day=31)
        names = sorted({_name(r.get("nm")) for r in ceos} - {""})
        rec = {
            "avail": pd.Timestamp(rno[:8]) + pd.Timedelta(1, "D"),
            "names": names,
            "ceo_n": float(max(len(names), 1)),
            "ceo_age": _nanagg(np.mean, [y - _num(r.get("birth_ym")) for r in ceos]),
            "ceo_tenure": _nanagg(max, [_tenure(r.get("hffc_pd"), asof) for r in ceos]),
            "ceo_family": float(any(_owner(r.get("mxmm_shrholdr_relate")) for r in ceos)),
            "ceo_registered": float(any("미등기" not in str(r.get("rgist_exctv_at")) for r in ceos)),
            "ceo_edu_known": float(bool(edu)),
            "ceo_insider": float(bool(corp) and any(corp[:4] in re.sub(r"\s", "", c) for c in careers)),
        }
        for col, rx in (
            ("ceo_sky", SKY),
            ("ceo_kaist_postech", KP),
            ("ceo_foreign", FOREIGN),
            ("ceo_phd", PHD),
            ("ceo_master", MASTER),
            ("ceo_mba", MBA),
            *MAJOR.items(),
        ):
            rec[col] = anyf(rx, edu) if edu else np.nan
        recs.append(rec)
    if not recs:
        return None
    t = pd.DataFrame(recs).sort_values("avail").drop_duplicates("avail", keep="last").reset_index(drop=True)
    prev = t["names"].shift(1)
    t["changed"] = [
        float(bool(p) and set(n) != set(p)) if isinstance(p, list) else 0.0
        for n, p in zip(t["names"], prev, strict=True)
    ]
    return t


def build_kr_ceo(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(np.nan, index=frame.index, columns=KR_CEO_COLS)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    snap_cols = [c for c in KR_CEO_COLS if c != "ceo_new_12m"]
    n = 0
    for e in np.unique(ents):
        t = kr_ceo_table(settings, e)
        if t is None:
            continue
        n += 1
        ii = np.where(ents == e)[0]
        out.iloc[ii, [out.columns.get_loc(c) for c in snap_cols]] = v16._asof(t, snap_cols, "avail", dates[ii], MAX_AGE)
        ch = t.loc[t["changed"] > 0, "avail"].to_numpy(dtype="datetime64[ns]")
        c, _ = v16._windowed(ch, np.ones(len(ch)), dates[ii], 365)
        out.iloc[ii, out.columns.get_loc("ceo_new_12m")] = (c > 0).astype(float)
    return out, {"kr_ceo_entities": n}


# ================================================================================================ KR strategy text


def _hash_vec(texts: list[str]):
    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(analyzer="char", ngram_range=(3, 3), n_features=2**18, alternate_sign=False, norm="l2")
    return hv.transform(texts)


def build_kr_strategy(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    cols = [f"st_{k}" for k in KR_THEMES] + [f"st_{k}_chg" for k in KR_THEMES] + ["st_sim", "st_len_chg"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    rx = {k: re.compile(p) for k, p in KR_THEMES.items()}
    n_docs = 0
    for e in np.unique(ents):
        p = cd.root(settings) / "business" / f"{e}.jsonl.gz"
        if not p.exists():
            continue
        recs = [r for r in tone.read_records(p) if r.get("text")]
        if not recs:
            continue
        t = pd.DataFrame(recs)
        t["avail"] = pd.to_datetime(t["rcept_no"].str[:8]) + pd.Timedelta(1, "D")
        t = t.sort_values("avail").drop_duplicates("year", keep="last").reset_index(drop=True)
        n_docs += len(t)
        lens = t["text"].str.len().clip(lower=1)
        for k, r_ in rx.items():
            t[f"st_{k}"] = t["text"].map(lambda s, r_=r_: len(r_.findall(s))) / lens * 1e4
            t[f"st_{k}_chg"] = t[f"st_{k}"] - t[f"st_{k}"].shift(1)
        vec = _hash_vec(t["text"].tolist())
        sims = [np.nan] + [float(vec[i].multiply(vec[i - 1]).sum()) for i in range(1, len(t))]
        t["st_sim"] = sims
        t["st_len_chg"] = np.log(lens) - np.log(lens.shift(1))
        ii = np.where(ents == e)[0]
        out.iloc[ii, :] = v16._asof(t, cols, "avail", dates[ii], MAX_AGE)
    return out, {"kr_business_docs": n_docs}


# ================================================================================================ US CEO


def build_us_ceo(frame: pd.DataFrame, settings, cache) -> tuple[pd.DataFrame, dict]:
    cols = ["ceo_wd_known", "ceo_ivy", "ceo_mba_wd", "ceo_eng", "ceo_tenure_wd", "ceo_age_wd", "ceo_new_wd"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    terms = pd.read_parquet(cd.root(settings) / "wd_ceo_terms.parquet")
    people = pd.read_parquet(cd.root(settings) / "wd_ceo_people.parquet")
    terms["cik"] = pd.to_numeric(terms["cik"].str.lstrip("0"), errors="coerce")
    for c in ("start", "end"):
        terms[c] = pd.to_datetime(terms[c].reindex(terms.index).astype("string").str[:10], errors="coerce")
    terms = terms.dropna(subset=["cik", "start"])
    prof = {}
    for q, g in people.groupby("ceo"):
        schools = " ".join(g["schoolLabel"].dropna().astype(str).unique()) if "schoolLabel" in g else ""
        degrees = " ".join(g["degreeLabel"].dropna().astype(str).unique()) if "degreeLabel" in g else ""
        births = g["birth"].dropna().astype(str).str[:10] if "birth" in g else pd.Series(dtype=str)
        birth = pd.to_datetime(births, errors="coerce").min()
        known = bool(schools)
        prof[q] = (
            1.0,
            float(bool(IVY.search(schools))) if known else np.nan,
            float(bool(US_MBA.search(schools + " " + degrees))) if known else np.nan,
            float(bool(US_ENG.search(schools + " " + degrees))) if known else np.nan,
            birth,
        )
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": np.unique(ents)}))
    by_cik = {int(k): g for k, g in terms.groupby("cik")}
    n = 0
    for e in np.unique(ents):
        tt = [by_cik[int(c)] for c in chains.get(e, set()) if int(c) in by_cik]
        if not tt:
            continue
        g = pd.concat(tt).sort_values("start").drop_duplicates(["ceo", "start"]).reset_index(drop=True)
        g["end"] = g["end"].fillna(g["start"].shift(-1))
        n += 1
        ii = np.where(ents == e)[0]
        d = dates[ii]
        st = g["start"].to_numpy(dtype="datetime64[ns]")
        en = g["end"].to_numpy(dtype="datetime64[ns]")
        k = np.searchsorted(st, d, "right") - 1
        kk = np.clip(k, 0, None)
        ok = (k >= 0) & (np.isnat(en[kk]) | (en[kk] > d))
        if not ok.any():
            continue
        pr = np.array([prof.get(q, (1.0, np.nan, np.nan, np.nan, pd.NaT))[:4] for q in g["ceo"]], dtype=float)
        birth = pd.to_datetime([prof.get(q, (0, 0, 0, 0, None))[4] for q in g["ceo"]]).to_numpy("datetime64[ns]")
        rows = ii[ok]
        out.iloc[rows, 0:4] = pr[kk[ok]]
        days = (d[ok] - st[kk[ok]]) / np.timedelta64(1, "D")
        out.iloc[rows, 4] = days / 365.25
        b = birth[kk[ok]]
        out.iloc[rows, 5] = np.where(np.isnat(b), np.nan, (d[ok] - b) / np.timedelta64(1, "D") / 365.25)
        out.iloc[rows, 6] = (days <= 365).astype(float)
    return out, {"us_ceo_entities": n}


_QUOTE = re.compile(r"[“\"]([^”\"]{40,1200})[”\"]\s*,?\s*(?:said|says|stated|commented|added)\s+([^.]{0,160})", re.S)
_CEO = re.compile(r"chief executive|\bCEO\b", re.I)
_NAME = re.compile(r"^([A-Z][a-zA-Z.\-']+(?:\s+[A-Z][a-zA-Z.\-']+){1,3})")


def ceo_quotes(text: str) -> tuple[str, str]:
    """(joined CEO quote text, the attributed name) of one release ('' when no CEO quote)."""
    qs, name = [], ""
    for m in _QUOTE.finditer(text or ""):
        tail = m.group(2)
        if _CEO.search(tail):
            qs.append(m.group(1))
            if not name:
                nm = _NAME.search(tail.strip())
                name = nm.group(1) if nm else ""
    return " ".join(qs), name


def build_us_quotes(frame: pd.DataFrame, settings, cache, spec: str = "lm") -> tuple[pd.DataFrame, dict]:
    lex = lx.load_lexicon(spec, settings)
    rx = {k: re.compile(p, re.I) for k, p in US_THEMES.items()}
    rows = []
    for p in sorted((tone.root(settings) / "ex991").glob("*.jsonl.gz")):
        cik = int(p.name.split(".")[0])
        for r in tone.read_records(p):
            q, name = ceo_quotes(r.get("text", ""))
            if not q:
                continue
            c = lx.counts(q, lex)
            n = max(c["n_words"], 1)
            rec = {
                "cik": cik,
                "published": pd.Timestamp(r["published"]),
                "name": name,
                "cq_len": np.log(n),
                "cq_tone": (c["positive"] - c["negative"]) / max(c["positive"] + c["negative"], 1),
                "cq_unc": c["uncertainty"] / n,
            }
            for k, r_ in rx.items():
                rec[f"cq_{k}"] = len(r_.findall(q)) / n * 100
            rows.append(rec)
    t = pd.DataFrame(rows)
    cols = [c for c in t.columns if c.startswith("cq_")] + ["cq_new_name"]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": np.unique(ents)}))
    by = {int(k): g for k, g in t.groupby("cik")}
    for e in np.unique(ents):
        parts = [by[int(c)] for c in chains.get(e, set()) if int(c) in by]
        if not parts:
            continue
        g = pd.concat(parts).sort_values("published")
        last = g["name"].str.split().str[-1].fillna("")
        g = g.assign(cq_new_name=((last != last.shift(1)) & (last != "") & last.shift(1).notna()).astype(float))
        g.index = pd.DatetimeIndex(g["published"]) + pd.Timedelta(1, "D")
        g = g[~g.index.duplicated(keep="last")]
        ii = np.where(ents == e)[0]
        out.iloc[ii, :] = v13._asof_rows(g[cols], dates[ii], tone.TX_MAX_AGE)
    return out, {"releases_with_ceo_quote": len(t), "lexicon": lex.tag}


def block(market: str, frame: pd.DataFrame, settings, cache) -> tuple[pd.DataFrame, dict]:
    if market == "KR":
        a, ia = build_kr_ceo(frame, settings)
        b, ib = build_kr_strategy(frame, settings)
        return pd.concat([a, b], axis=1), {**ia, **ib}
    a, ia = build_us_ceo(frame, settings, cache)
    b, ib = build_us_quotes(frame, settings, cache)
    return pd.concat([a, b], axis=1), {**ia, **ib}
