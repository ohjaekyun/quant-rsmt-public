"""Point-in-time S&P 500 membership: fja05680/sp500 snapshots + Wikipedia patch + known-error patches.

Sources (verified live 2026-09-25, docs/sources/pit/us-universe.json):
- fja05680/sp500 '(Updated).csv': one row per change date since 1996 (date, comma-joined tickers). Effective dates;
  a spell ends on the FIRST day out. fja lags real changes by weeks and uses ~2019 tickers before 2019.
- Wikipedia 'List of S&P 500 companies' (current table, with CIK) and 'Historical components of the S&P 500'
  (the changes table moved there on 2026-08-11). Changes dated after fja's last row are applied on top of fja.
- ``MANUAL_PATCHES``: fja errors found by the independent verification and while porting (TYC, ACE and Allergan Inc.
  missing because two issuers shared one back-filled ticker; JCI/CB/IR/FOXA/FOX/AGN covering two issuers in one
  spell; the CCEP duplicate). Each patch is guarded and reported.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from quantcio.data.pit.us_net import PitNet

FJA_CONTENTS = "https://api.github.com/repos/fja05680/sp500/contents/"
FJA_RAW = "https://raw.githubusercontent.com/fja05680/sp500/master/"
WIKI_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_HIST = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"

SPELL_COLUMNS = ["ticker", "start", "end", "cik", "name", "ftd_sym", "yahoo_hint", "reason", "patch"]
EVENT_COLUMNS = [
    "date",
    "start",
    "fja_ticker",
    "ticker",
    "name",
    "link",
    "reason",
    "rename_to",
    "cat",
    "cik",
    "cik_method",
    "patch",
]
COUNT_RANGE = (495, 510)  # month-end ticker counts (observed 497-506 in 2014-2026)
COUNT_HARD_RANGE = (450, 560)  # outside this the sources are broken -> raise
MAX_CURRENT_MISMATCH = 5  # current-set differences tolerated as a note; more -> raise


class MembershipError(RuntimeError):
    """The membership sources disagree structurally (parser/layout break): never build on it."""


def dash(t: str) -> str:
    """fja/Wikipedia spell share classes with '.', SEC/Yahoo with '-' (BRK.B -> BRK-B)."""
    return str(t).strip().upper().replace(".", "-")


def ftd_symbol(t: str) -> str:
    """SEC fails-to-deliver symbols drop the class separator (BRK.B -> BRKB)."""
    return dash(t).replace("-", "")


# ============================================================================ fja05680


def pick_fja_file(names: list[str]) -> str:
    """'(Updated)' > newest '(MM-DD-YYYY)' > bare name (a lexical sort mis-orders dated names across years)."""
    cands = [n for n in names if n.startswith("S&P 500 Historical Components") and n.endswith(".csv")]
    if not cands:
        raise MembershipError(f"fja05680 저장소에 'S&P 500 Historical Components*.csv' 파일 없음: {names[:10]}")

    def key(n: str) -> tuple[int, pd.Timestamp]:
        m = re.search(r"\((\d{2})-(\d{2})-(\d{4})\)", n)
        if "Updated" in n:
            return 2, pd.Timestamp.min
        if m:
            return 1, pd.Timestamp(f"{m[3]}-{m[1]}-{m[2]}")
        return 0, pd.Timestamp.min

    return max(cands, key=key)


def parse_fja_csv(raw: bytes) -> pd.DataFrame:
    """index=date, 'tickers'=frozenset (fja spelling, '.' classes)."""
    df = pd.read_csv(io.BytesIO(raw))
    if not {"date", "tickers"} <= set(df.columns):
        raise MembershipError(f"fja CSV 형식 변경: 열 {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    df["tickers"] = df["tickers"].map(lambda s: frozenset(t.strip() for t in str(s).split(",") if t.strip()))
    return df.set_index("date").sort_index()


def spells_from_snapshots(snaps: pd.DataFrame) -> pd.DataFrame:
    """ticker, start (first day in), end (first day out = removal effective date; NaT = still in)."""
    rows, open_, prev = [], {}, frozenset()
    for d, cur in snaps["tickers"].items():
        for t in cur - prev:
            open_[t] = d
        for t in prev - cur:
            rows.append((t, open_.pop(t), d))
        prev = cur
    rows += [(t, s, pd.NaT) for t, s in open_.items()]
    sp = pd.DataFrame(rows, columns=["ticker", "start", "end"])
    for c in SPELL_COLUMNS[3:]:
        sp[c] = np.nan if c == "cik" else None
    sp["patch"] = ""
    return sp[SPELL_COLUMNS].sort_values(["ticker", "start"], ignore_index=True)


def load_fja(net: PitNet, max_age_h: float | None = 24) -> pd.DataFrame:
    items = json.loads(net.cached_bytes("sp500", "gh_contents.json", FJA_CONTENTS, "github", max_age_h))
    fn = pick_fja_file([e.get("name", "") for e in items if isinstance(e, dict)])
    raw = net.cached_bytes("sp500", "fja_hist.csv", FJA_RAW + quote(fn), "github", max_age_h)
    return parse_fja_csv(raw)


# ============================================================================ Wikipedia


def parse_wiki_current(html: bytes | str) -> pd.DataFrame:
    text = html.decode() if isinstance(html, bytes) else html
    t = pd.read_html(io.StringIO(text))[0]
    cols = [str(c) for c in t.columns]
    if not cols or cols[0].lower() != "symbol" or "CIK" not in cols or t.shape[1] != 8:
        raise MembershipError(f"Wikipedia 구성종목 표 형식 변경: {cols}")  # never a silent column shift
    t.columns = ["symbol", "security", "sector", "sub_industry", "hq", "date_added", "cik", "founded"]
    t["cik"] = pd.to_numeric(t["cik"], errors="coerce").astype("Int64")
    t["symbol"] = t["symbol"].astype(str).str.strip()
    t["sym"] = t["symbol"].map(dash)
    return t


def parse_wiki_changes(html: bytes | str) -> pd.DataFrame:
    """One row per (added, removed) pair with article links; symbol renames are NOT listed there."""
    text = html.decode() if isinstance(html, bytes) else html
    tables = BeautifulSoup(text, "lxml").select("table.wikitable")
    if not tables:
        raise MembershipError("Wikipedia 'Historical components' 페이지에 변경 표 없음")

    def cell(td) -> tuple[str | None, str | None]:
        a = td.find("a", href=True)
        link = unquote(a["href"].split("/wiki/")[-1]) if a and "/wiki/" in a["href"] else None
        return td.get_text(" ", strip=True) or None, link

    def clean(t: str | None) -> str | None:
        return (re.sub(r"[^A-Za-z0-9.\-]", "", t) or None) if t else None

    rows = []
    for tr in tables[0].select("tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        try:
            d = pd.to_datetime(tds[0].get_text(" ", strip=True), format="%B %d, %Y")
        except (ValueError, TypeError):
            continue
        add_t, _ = cell(tds[1])
        add_n, add_l = cell(tds[2])
        rem_t, _ = cell(tds[3])
        rem_n, rem_l = cell(tds[4])
        rows.append(
            dict(
                date=d,
                add_t=clean(add_t),
                add_n=add_n,
                add_link=add_l,
                rem_t=clean(rem_t),
                rem_n=rem_n,
                rem_link=rem_l,
                reason=tds[5].get_text(" ", strip=True),
            )
        )
    if not rows:
        raise MembershipError("Wikipedia 변경 표에서 행을 읽지 못함 (형식 변경?)")
    return pd.DataFrame(rows).sort_values("date", ascending=False, ignore_index=True)


def load_wiki(net: PitNet, max_age_h: float | None = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    cur = parse_wiki_current(net.cached_bytes("sp500", "wiki_sp500.html", WIKI_CURRENT, "wiki", max_age_h))
    ch = parse_wiki_changes(net.cached_bytes("sp500", "wiki_hist.html", WIKI_HIST, "wiki", max_age_h))
    return cur, ch


# ============================================================================ patching


def patch_spells(
    snaps: pd.DataFrame, changes: pd.DataFrame, today: pd.Timestamp
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """fja spells + Wikipedia rows dated in (fja's last row, today]. Returns (spells, applied, warnings)."""
    sp = spells_from_snapshots(snaps)
    last = snaps.index[-1]
    applied: list[str] = []
    warnings: list[str] = []
    todo = changes[(changes["date"] > last)].sort_values("date")
    for _, r in todo.iterrows():
        if r["date"] > today:
            warnings.append(f"{r['date'].date()} 예정 변경(편입 {r['add_t']}/편출 {r['rem_t']})은 아직 미반영")
            continue
        if isinstance(r["rem_t"], str):
            hit = (sp["ticker"].map(dash) == dash(r["rem_t"])) & sp["end"].isna()
            if not hit.any():
                warnings.append(
                    f"{r['date'].date()} 편출 {r['rem_t']}: 열린 fja 구간 없음 (fja 마지막 행 이후 티커 변경?) — 무시"
                )
            sp.loc[hit, "end"] = r["date"]
        if isinstance(r["add_t"], str):
            row = {c: None for c in SPELL_COLUMNS}
            row.update(ticker=r["add_t"], start=r["date"], end=pd.NaT, cik=np.nan, patch="")
            sp = pd.concat([sp, pd.DataFrame([row])], ignore_index=True)
        applied.append(f"{r['date'].date()} +{r['add_t']} -{r['rem_t']}")
    sp["start"] = pd.to_datetime(sp["start"])
    sp["end"] = pd.to_datetime(sp["end"])
    return sp.sort_values(["ticker", "start"], ignore_index=True), applied, warnings


@dataclass(frozen=True)
class SpellPatch:
    """One known fja error. kind: 'split' (spell of ``ticker`` covering ``at`` -> [start, at) gets ``cik``),
    'add' (new spell [start, end) with ``cik``), 'drop' (spell of ``ticker`` starting ``start`` removed)."""

    kind: str
    ticker: str
    label: str
    at: str | None = None
    start: str | None = None
    end: str | None = None
    cik: int | None = None
    name: str | None = None
    ftd_sym: str | None = None
    yahoo_hint: str | None = None
    reason: str | None = None
    allow_overlap: bool = False  # 'add' next to an fja spell of the same back-filled ticker (two issuers)


MANUAL_PATCHES: tuple[SpellPatch, ...] = (
    SpellPatch(
        "add",
        "TYC",
        "Tyco International 누락 (fja): 2010-08-26~2016-09-06 추가",
        start="2010-08-26",
        end="2016-09-06",
        cik=833444,
        name="Tyco International",
        ftd_sym="TYC",
        yahoo_hint="JCI",
        reason="TYC acquired JCI and retained JCI ticker.",
    ),
    SpellPatch(
        "split",
        "JCI",
        "JCI 구간 분리 (fja): 2016-09-06 이전은 Johnson Controls Inc.",
        at="2016-09-06",
        cik=53669,
        name="Johnson Controls Inc.",
        reason="Tyco acquired Johnson Controls Inc. (JCI plc = former Tyco).",
    ),
    SpellPatch(
        "add",
        "ACE",
        "ACE Ltd 누락 (fja): 2010-07-14~2016-01-19 추가",
        start="2010-07-14",
        end="2016-01-19",
        cik=896159,
        name="ACE Limited",
        ftd_sym="ACE",
        yahoo_hint="CB",
        reason="ACE Ltd acquired Chubb and retained the CB ticker.",
    ),
    SpellPatch(
        "split",
        "CB",
        "CB 구간 분리 (fja): 2016-01-19 이전은 Chubb Corp",
        at="2016-01-19",
        cik=20171,
        name="Chubb Corp",
        reason="ACE Ltd acquired Chubb Corp.",
    ),
    SpellPatch(
        "split",
        "IR",
        "IR 구간 분리 (fja): 2020-03-03 이전은 Ingersoll-Rand plc (현 TT)",
        at="2020-03-03",
        cik=1466258,
        name="Ingersoll-Rand plc",
        yahoo_hint="TT",
        reason="Ingersoll-Rand plc renamed Trane Technologies (TT); Gardner Denver took the IR ticker.",
    ),
    SpellPatch(
        "split",
        "FOXA",
        "FOXA 구간 분리 (fja): 2019-03-19 이전은 21st Century Fox",
        at="2019-03-19",
        cik=1308161,
        name="21st Century Fox (Class A)",
        reason="21st Century Fox merged with Disney; Fox Corporation took the FOXA ticker.",
    ),
    SpellPatch(
        "split",
        "FOX",
        "FOX 구간 분리 (fja): 2019-03-19 이전은 21st Century Fox",
        at="2019-03-19",
        cik=1308161,
        name="21st Century Fox (Class B)",
        reason="21st Century Fox merged with Disney; Fox Corporation took the FOX ticker.",
    ),
    SpellPatch(
        "split",
        "AGN",
        "AGN 구간 분리 (fja): 2015-06-15 이전은 Actavis plc (티커 ACT)",
        at="2015-06-15",
        cik=1578845,
        name="Actavis plc",
        ftd_sym="ACT",
        reason="Actavis plc renamed Allergan plc (ticker ACT -> AGN).",
    ),
    SpellPatch(
        "add",
        "AGN",
        "Allergan Inc. 누락 (fja, 같은 티커로 합쳐짐): ~2015-03-23 추가",
        start="1996-01-02",
        end="2015-03-23",
        cik=850693,
        name="Allergan Inc.",
        ftd_sym="AGN",
        reason="Actavis acquired Allergan Inc. (Actavis later took the Allergan name and the AGN ticker).",
        allow_overlap=True,
    ),
    SpellPatch(
        "drop",
        "CCEP",
        "CCEP 중복 제거 (fja): 2016-01-04~2016-05-31은 CCE와 중복",
        start="2016-01-04",
        end="2016-05-31",
    ),
    # fja rows before 2019 carry the ~2019 ticker (Clenow back-fill). Where Yahoo has no history of the issuer
    # before the rename, or the rename came with a new CIK, the pre-rename part gets the ticker traded then, so SEC
    # FTD prices (tracked by symbol) can fill and validate those months. Yahoo-covered renames (WLP->ANTM,
    # MHFI->SPGI, PCLN->BKNG, ...) need no patch: the current ticker's series carries the history.
    SpellPatch(
        "split",
        "ARNC",
        "ARNC 티커 역보정 (fja): 2016-11-01 이전은 Alcoa Inc. (AA)",
        at="2016-11-01",
        cik=4281,
        name="Alcoa Inc.",
        ftd_sym="AA",
        reason="Alcoa Inc. renamed Arconic (AA -> ARNC) and spun off Alcoa Corp, which took the AA ticker.",
    ),
    SpellPatch(
        "split",
        "ANDV",
        "ANDV 티커 역보정 (fja): 2017-08-01 이전은 Tesoro (TSO)",
        at="2017-08-01",
        cik=50104,
        name="Tesoro Corp.",
        ftd_sym="TSO",
        reason="Tesoro renamed Andeavor (TSO -> ANDV).",
    ),
    SpellPatch(
        "split",
        "TGNA",
        "TGNA 티커 역보정 (fja): 2015-06-29 이전은 Gannett (GCI)",
        at="2015-06-29",
        cik=39899,
        name="Gannett Co. (현 Tegna)",
        ftd_sym="GCI",
        reason="Gannett renamed Tegna (GCI -> TGNA) and spun off the new Gannett, which took the GCI ticker.",
    ),
    SpellPatch(
        "split",
        "WBA",
        "WBA 티커 역보정 (fja): 2014-12-31 이전은 Walgreen Co. (WAG)",
        at="2014-12-31",
        cik=1618921,  # holdco reorganisation (1:1 share exchange): one entity; fundamentals chain 104207 -> 1618921
        name="Walgreen Co.",
        ftd_sym="WAG",
        reason="Walgreen Co. reorganised into Walgreens Boots Alliance (WAG -> WBA).",
    ),
    SpellPatch(
        "split",
        "BHGE",
        "BHGE 구간 분리 (fja): 2017-07-05 이전은 Baker Hughes Inc. (BHI, 다른 CIK)",
        at="2017-07-05",
        cik=808362,
        name="Baker Hughes Inc.",
        ftd_sym="BHI",
        yahoo_hint="BKR",  # Yahoo's BKR series carries the BHI history before 2017-07-05 (checked against FTD)
        reason="Baker Hughes Inc. merged with GE Oil & Gas into Baker Hughes, a GE company (BHGE; new registrant).",
    ),
)


def apply_manual_patches(
    spells: pd.DataFrame, patches: tuple[SpellPatch, ...] = MANUAL_PATCHES
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Apply guarded fja fixes. Returns (spells, applied labels, skipped labels with the reason)."""
    sp = spells.copy()
    applied: list[str] = []
    skipped: list[str] = []
    for p in patches:
        tk = sp["ticker"].map(dash) == dash(p.ticker)
        if p.kind == "add":
            s, e = pd.Timestamp(p.start), pd.Timestamp(p.end)
            overlap = tk & (sp["start"] < e) & (sp["end"].isna() | (sp["end"] > s))
            if p.allow_overlap:
                overlap &= sp["cik"] == float(p.cik or -1)
            if overlap.any():
                skipped.append(f"{p.label}: 이미 fja에 존재 — 건너뜀")
                continue
            row = dict(
                ticker=p.ticker,
                start=s,
                end=e,
                cik=float(p.cik) if p.cik else np.nan,
                name=p.name,
                ftd_sym=p.ftd_sym,
                yahoo_hint=p.yahoo_hint,
                reason=p.reason,
                patch=p.label,
            )
            sp = pd.concat([sp, pd.DataFrame([row])], ignore_index=True)
        elif p.kind == "split":
            at = pd.Timestamp(p.at)
            hit = tk & (sp["start"] < at) & (sp["end"].isna() | (sp["end"] > at)) & (sp["patch"] == "")
            if hit.sum() != 1:
                skipped.append(f"{p.label}: 분리할 fja 구간 {int(hit.sum())}개 — 건너뜀")
                continue
            i = hit[hit].index[0]
            old = sp.loc[i].to_dict()
            old.update(
                end=at,
                cik=float(p.cik) if p.cik else np.nan,
                name=p.name,
                ftd_sym=p.ftd_sym,
                yahoo_hint=p.yahoo_hint,
                reason=p.reason,
                patch=p.label,
            )
            sp.loc[i, "start"] = at
            sp = pd.concat([sp, pd.DataFrame([old])], ignore_index=True)
        elif p.kind == "drop":
            s, e = pd.Timestamp(p.start), pd.Timestamp(p.end)
            hit = tk & ((sp["start"] - s).abs() <= pd.Timedelta(3, "D"))
            hit &= sp["end"].notna() & ((sp["end"] - e).abs() <= pd.Timedelta(3, "D"))
            if not hit.any():
                skipped.append(f"{p.label}: 해당 구간 없음 — 건너뜀")
                continue
            sp = sp[~hit]
        else:  # pragma: no cover - programming error
            raise ValueError(p.kind)
        applied.append(p.label)
    sp["start"] = pd.to_datetime(sp["start"])
    sp["end"] = pd.to_datetime(sp["end"])
    return sp.sort_values(["ticker", "start"], ignore_index=True), applied, skipped


def members_at(spells: pd.DataFrame, when) -> pd.DataFrame:
    """Spell rows in force at the close of ``when`` (start <= when < end)."""
    when = pd.Timestamp(when)
    return spells[(spells["start"] <= when) & (spells["end"].isna() | (spells["end"] > when))]


def members_matrix(spells: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Long frame (date, spell index) of every spell in force at each date."""
    out = []
    for d in dates:
        m = members_at(spells, d)
        out.append(pd.DataFrame({"date": d, "spell": m.index}))
    if not out:
        return pd.DataFrame(columns=["date", "spell"])
    return pd.concat(out, ignore_index=True)


# ============================================================================ removal events

_RENAME_RX = r"changed its ticker|ticker symbol from|changed its name|renamed|changed to|retained the"
_ACQ_RX = r"acqui|merg|purchas|privati|taken private|bought|buyout"
_FAIL_RX = r"bankrupt|receivership|chapter 11|fdic"


def classify_reason(reason: str | None, name: str | None, ticker: str, rename_to: str | None) -> str:
    """Informational label (renamed/failed/acquired/demoted/unknown). Known after the fact and ~10% wrong on audit:
    never use it as a feature."""
    s = str(reason or "").lower()
    if rename_to or re.search(_RENAME_RX, s):
        return "renamed"
    if re.search(_FAIL_RX, s):
        return "failed"
    first = re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).split()[:1]
    if re.search(_ACQ_RX, s) and ((first and first[0] in s) or ticker.lower() in s):
        return "acquired"
    return "demoted" if reason else "unknown"


def removal_events(spells: pd.DataFrame, changes: pd.DataFrame, w0: pd.Timestamp) -> pd.DataFrame:
    """One row per ended (non-patched) spell with end >= w0: Wikipedia name/reason (+-10 d), fja same-day-add rename
    detection, and alias of fja back-filled tickers (BTUUQ->BTU, AABA->YHOO, DXC->CSC, WYND->WYN, KDP->DPS)."""
    ch = changes.copy()
    for c in ("add_t", "rem_t"):
        ch[c] = ch[c].map(lambda x: dash(x) if isinstance(x, str) else None)
    base = spells[spells["patch"] == ""]
    ended = base[base["end"].notna() & (base["end"] >= pd.Timestamp(w0))]
    starts = base.assign(tn=base["ticker"].map(dash))
    rem = ch[ch["rem_t"].notna() & (ch["date"] >= pd.Timestamp(w0) - pd.Timedelta(15, "D"))]
    used: set = set()
    out = []
    for _, s in ended.iterrows():
        tn = dash(s["ticker"])
        m = rem[(rem["rem_t"] == tn) & ((rem["date"] - s["end"]).abs() <= pd.Timedelta(10, "D"))]
        row = dict(
            date=s["end"],
            start=s["start"],
            fja_ticker=s["ticker"],
            ticker=tn,
            name=None,
            link=None,
            reason=None,
            rename_to=None,
        )
        if len(m):
            used.add(m.index[0])
            row.update(name=m["rem_n"].iloc[0], link=m["rem_link"].iloc[0], reason=m["reason"].iloc[0])
        else:
            adds = starts[starts["start"] == s["end"]]["tn"].tolist()
            wadds = set(ch[(ch["date"] - s["end"]).abs() <= pd.Timedelta(10, "D")]["add_t"].dropna())
            cand = [a for a in adds if a not in wadds]
            if len(cand) == 1:
                row["rename_to"] = cand[0]
        out.append(row)
    ev = pd.DataFrame(out, columns=["date", "start", "fja_ticker", "ticker", "name", "link", "reason", "rename_to"])
    left = rem[~rem.index.isin(used) & (rem["date"] >= pd.Timestamp(w0))]
    for i, r in ev[ev["name"].isna() & ev["rename_to"].isna()].iterrows():
        m = left[(left["date"] - r["date"]).abs() <= pd.Timedelta(3, "D")]
        m = m[~m["rem_t"].isin(ev["ticker"])]
        if len(m) == 1:
            ev.loc[i, ["ticker", "name", "link", "reason"]] = [
                m["rem_t"].iloc[0],
                m["rem_n"].iloc[0],
                m["rem_link"].iloc[0],
                m["reason"].iloc[0],
            ]
            left = left.drop(index=m.index)
    ev["cat"] = [classify_reason(r["reason"], r["name"], r["ticker"], r["rename_to"]) for _, r in ev.iterrows()]
    ev["cik"] = np.nan
    ev["cik_method"] = None
    ev["patch"] = ""
    return ev[EVENT_COLUMNS].sort_values("date", ignore_index=True)


def patch_events(spells: pd.DataFrame, w0: pd.Timestamp) -> pd.DataFrame:
    """Event rows for ended manual-patch spells (CIK, name and reason come from the patch)."""
    p = spells[(spells["patch"] != "") & spells["end"].notna() & (spells["end"] >= pd.Timestamp(w0))]
    rows = []
    for _, s in p.iterrows():
        rows.append(
            dict(
                date=s["end"],
                start=s["start"],
                fja_ticker=s["ticker"],
                ticker=dash(s["ftd_sym"] or s["ticker"]),
                name=s["name"],
                link=None,
                reason=s["reason"],
                rename_to=None,
                cat=classify_reason(s["reason"], s["name"], dash(s["ticker"]), None),
                cik=s["cik"],
                cik_method="manual_patch",
                patch=s["patch"],
            )
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# ============================================================================ checks


def membership_checks(
    spells: pd.DataFrame, current: pd.DataFrame, month_ends: pd.DatetimeIndex
) -> tuple[dict, list[str]]:
    """Current set == Wikipedia table and month-end counts in COUNT_RANGE. Raises MembershipError on a structural
    break (current mismatch > MAX_CURRENT_MISMATCH or counts outside COUNT_HARD_RANGE); smaller deviations -> notes."""
    notes: list[str] = []
    now = set(spells[spells["end"].isna()]["ticker"].map(dash))
    wiki = set(current["sym"])
    only_fja, only_wiki = sorted(now - wiki), sorted(wiki - now)
    diff = len(only_fja) + len(only_wiki)
    if diff > MAX_CURRENT_MISMATCH:
        raise MembershipError(
            f"현재 구성종목 불일치 {diff}개 (fja+패치에만 {only_fja[:10]}, Wikipedia에만 {only_wiki[:10]}) — "
            "소스 형식 변경 가능성, 빌드 중단"
        )
    if diff:
        notes.append(f"현재 구성종목 {diff}개 불일치: fja+패치에만 {only_fja}, Wikipedia에만 {only_wiki}")
    counts = pd.Series([len(members_at(spells, d)) for d in month_ends], index=month_ends, dtype="float64")
    out_of_range = counts[(counts < COUNT_RANGE[0]) | (counts > COUNT_RANGE[1])]
    if len(counts) and ((counts < COUNT_HARD_RANGE[0]) | (counts > COUNT_HARD_RANGE[1])).any():
        bad = counts[(counts < COUNT_HARD_RANGE[0]) | (counts > COUNT_HARD_RANGE[1])]
        raise MembershipError(f"월말 구성종목 수 비정상 {bad.head(5).to_dict()} — 소스 파싱 오류 가능성, 빌드 중단")
    if len(out_of_range):
        notes.append(
            f"월말 구성종목 수 {len(out_of_range)}개월이 {COUNT_RANGE[0]}-{COUNT_RANGE[1]} 범위 밖 "
            f"(예: {', '.join(f'{d.date()}={int(v)}' for d, v in out_of_range.head(5).items())})"
        )
    info = {
        "current_equal_wikipedia": diff == 0,
        "current_count": len(now),
        "wikipedia_count": len(wiki),
        "current_only_fja": only_fja,
        "current_only_wikipedia": only_wiki,
        "month_end_count_min": int(counts.min()) if len(counts) else None,
        "month_end_count_max": int(counts.max()) if len(counts) else None,
        "month_end_count_mean": round(float(counts.mean()), 1) if len(counts) else None,
        "month_end_count_out_of_range": int(len(out_of_range)),
    }
    return info, notes
