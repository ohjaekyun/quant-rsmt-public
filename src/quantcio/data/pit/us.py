"""US point-in-time universe (S&P 500 members, top-n by PIT market cap) and PIT fundamentals (SEC XBRL).

Implements the shared contract in ``quantcio.data.pit`` for ``Market.US`` (keyless; verified live 2026-09-25, see
docs/sources/pit/us-universe.json and us-fundamentals.json and the corrected reference code next to them).

Universe (``load_us_pit_universe``)
  1. Membership: fja05680 S&P 500 snapshots + Wikipedia changes after fja's last row + guarded manual fixes of known
     fja errors. Checked on every build: today's set == the Wikipedia table, month-end counts 495-510.
  2. Entity = SEC CIK (share classes and renames collapse; a new CIK after a merger/redomicile is a new entity).
     Current spells take the Wikipedia CIK; ended spells the CIK in force at removal (validated Wayback SEC ticker
     files -> Wikidata -> name match -> unvalidated pair).
  3. Prices: per spell the Yahoo series that is THIS issuer (cross-checked against SEC fails-to-deliver prices of the
     ticker traded then; wrong-issuer series rejected, partly wrong months masked); total-return adjusted OHLC
     (``attrs['adjusted']``), ending at the last traded session. Member-months without a Yahoo return are filled from
     FTD prices (monthly marks only) and labelled; terminal/delisting marks are in ``meta`` / :func:`terminal_price`.
  4. Rank: PIT market cap = XBRL shares (filed <= t, validated) x price at t; members without a share count get
     trailing-60-session dollar volume x the date's median cap/volume ratio (counted); when shares cover < 80% of the
     members of a date, that date is ranked by dollar volume and the note says so. n >= 500 keeps every member.
  5. Coverage: member-months by year (daily prices / FTD-only / missing), rejected/masked series, CIK methods,
     membership checks and an equal-weight PIT S&P 500 vs RSP monthly tracking check (gap <= 0.5 %p/yr, TE <= 1%).

Fundamentals (``load_us_pit_fundamentals``): companyfacts as-of snapshots per filing date (see ``us_fund``), CIK chains
for predecessors, available_from = filed + 1 NYSE session, cut 30 days after a delisting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantcio.config import Settings
from quantcio.data.base import clean_prices
from quantcio.data.cache import DiskCache
from quantcio.data.pit import (
    FUNDAMENTAL_COLUMNS,
    MEMBERSHIP_COLUMNS,
    PITFundamentals,
    PITUniverse,
    us_cik,
    us_fund,
    us_membership,
    us_prices,
)
from quantcio.data.pit.us_calendar import month_end_sessions, nyse_sessions
from quantcio.data.pit.us_net import PitNet, require_sec_user_agent
from quantcio.models import Instrument, Market

log = logging.getLogger(__name__)

BUILD_VERSION = (
    7  # 7: price frames carry the day's traded ``value`` (84.3); 6: open spells refreshed to the last session
)
FUND_VERSION = 1
BENCHMARK = "SPY"
EW_BENCHMARK = "RSP"  # Invesco S&P 500 Equal Weight ETF (tracking check)
MIN_SHARE_COVERAGE = 0.8  # below this share of members with a PIT share count, a date is ranked by dollar volume
TRACKING_MAX_GAP = 0.005  # |annualised EW PIT - RSP| alert level
TRACKING_MAX_TE = 0.01
ADV_SESSIONS = 60
PRICE_MAX_AGE_DAYS = 7
MASK_MIN_BAD_MONTHS = 3  # fewer FTD-contradicted months are kept as suspect (false positives had 1; splices 24-34)
DELIST_GRACE_DAYS = 30  # fundamentals rows filed later than delisting + this are dropped (buyer keeps filing)
COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# successor CIK -> predecessor CIKs (oldest first) of holdco reorganisations / redomiciles (verified in SEC names)
MANUAL_PREDECESSORS: dict[int, tuple[int, ...]] = {
    1652044: (1288776,),  # Alphabet <- Google Inc (2015-10)
    1613103: (64670,),  # Medtronic plc <- Medtronic Inc (2015-01)
    1744489: (1001039,),  # Walt Disney (TWDC Holdco 613) <- old Disney (2019-03)
    2115436: (34088,),  # ExxonMobil Holdings <- Exxon Mobil Corp (2026)
    1739940: (701221,),  # Cigna (Halfmoon Parent) <- old Cigna (2018-12)
    1730168: (1441634, 1649338),  # Broadcom Inc <- Broadcom Ltd <- Avago (2016-02, 2018-04)
    1618921: (104207,),  # Walgreens Boots Alliance <- Walgreen Co (2014-12)
    1623613: (69499,),  # Mylan N.V. <- Mylan Inc (2015-02)
    1578845: (884629,),  # Actavis plc / Allergan plc <- Actavis Inc / Watson (2013-10)
    1551182: (31277,),  # Eaton plc <- Eaton Corp (2012-11)
    1585364: (820096,),  # Perrigo plc <- Perrigo Co (2013-12)
}


def _ts(d: date | pd.Timestamp | str | None) -> pd.Timestamp | None:
    return None if d is None else pd.Timestamp(d).normalize()


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _txt(x: Any, fallback: str = "") -> str:
    """NaN/None-safe text (object columns mix None and NaN after joins)."""
    return fallback if x is None or (isinstance(x, float) and np.isnan(x)) or x == "" else str(x)


# ============================================================================ public API


def load_us_pit_universe(
    settings: Settings,
    cache: DiskCache,
    n: int,
    start: date,
    end: date | None = None,
) -> PITUniverse:
    """PIT top-``n`` S&P 500 universe for month-end rebalance dates in [start, end] (see module docstring).

    Needs ``settings.data.sec_user_agent`` (SEC EDGAR); raises ``SecUserAgentError`` when it is empty.
    Cost (2026-09-25): a cold build downloads ~2,700 SEC/Yahoo/Wayback/Wikipedia files (~30-45 min; ~0.5 GB under
    ``settings.cache_path/pit/us``) and computes ~750 XBRL snapshot panels in a process pool (~4 min on 8 cores);
    a warm rebuild takes ~40 s and the result is cached per session (``build_v1``, pruned after 3 days).
    Scripts calling it must guard with ``if __name__ == "__main__":`` (spawned pool workers import ``__main__``).
    """
    require_sec_user_agent(settings)
    net = PitNet(settings)
    full = _full_build(settings, cache, net, _ts(start), _ts(end))
    return select_top_n(full, n)


def load_us_pit_fundamentals(
    settings: Settings,
    cache: DiskCache,
    entities: list[str] | None = None,
    *,
    net: PitNet | None = None,
    yahoo_splits: dict[str, pd.Series] | None = None,
    chart_from: dict[str, pd.Timestamp] | None = None,
    delist: dict[str, pd.Timestamp] | None = None,
    since: str = "2009-06-01",
    workers: int | None = None,
) -> PITFundamentals:
    """PIT fundamentals panel (FUNDAMENTAL_COLUMNS + extras) for ``entities`` ("CIK##########"; None = every entity
    of the last universe build, else every S&P 500 member since 2014).

    ``yahoo_splits`` / ``chart_from`` / ``delist`` default to what the last universe build recorded.
    """
    require_sec_user_agent(settings)
    net = net or PitNet(settings)
    known = net.read_json("build", "known_entities.json") or {}
    if entities is None:
        entities = list(known.get("entities") or [])
        if not entities:
            entities = _membership_entities(settings, net)
    if yahoo_splits is None:
        stored = net.read_pickle("build", "entity_splits.pkl") or {}
        yahoo_splits = stored.get("splits", {})
        chart_from = chart_from or stored.get("chart_from", {})
    if delist is None:
        delist = {e: pd.Timestamp(d) for e, d in (known.get("delist") or {}).items()}
    return _fundamentals(
        settings, net, sorted(set(entities)), yahoo_splits or {}, chart_from or {}, delist or {}, since, workers
    )


def terminal_price(universe: PITUniverse, entity: str) -> tuple[pd.Timestamp, float, str] | None:
    """(date, price, source) of the entity's delisting / terminal mark, or None while it still trades.

    ``price`` is on the basis of ``universe.prices[entity]`` (total-return adjusted) when the entity has a frame
    (``meta.terminal_basis == 'frame'``), else the raw FTD price. Sources: 'yahoo_last_bar', 'ftd' (last SEC
    fails-to-deliver price on/before removal), 'ftd_otc' (OTC print of the same CUSIP after a failure),
    'failed_assumed_zero' (failed firm, trading halted, no later print: -100%)."""
    m = universe.meta
    row = m[m["entity"] == entity]
    if row.empty or pd.isna(row["terminal_date"].iloc[0]):
        return None
    r = row.iloc[0]
    return pd.Timestamp(r["terminal_date"]), float(r["terminal_price"]), str(r["terminal_source"])


# ============================================================================ full build (all members)


@dataclass
class FullBuild:
    """Everything needed to select any top-n: all members per date with rank values, frames, month sources."""

    dates: pd.DatetimeIndex
    members: pd.DataFrame  # date, entity, ticker, name, rank_value, rank_basis, value_source
    month_src: pd.DataFrame  # date, entity, src (yahoo/ftd/missing), ret
    frames: dict[str, pd.DataFrame]
    benchmarks: dict[str, pd.DataFrame]
    meta: pd.DataFrame
    coverage: dict[str, Any]
    notes: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _build_key(settings: Settings, start: pd.Timestamp, end: pd.Timestamp | None, today: pd.Timestamp) -> str:
    blob = json.dumps(
        [
            BUILD_VERSION,
            str(start.date()),
            str(end.date()) if end is not None else None,
            str(today.date()),
            settings.pit.model_dump(),
            MIN_SHARE_COVERAGE,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _today() -> pd.Timestamp:
    from quantcio.data.us import last_completed_us_session

    return pd.Timestamp(last_completed_us_session())


def _full_build(
    settings: Settings,
    cache: DiskCache,
    net: PitNet,
    start: pd.Timestamp,
    end: pd.Timestamp | None,
    *,
    spy_loader: Any = None,
    today: pd.Timestamp | None = None,
) -> FullBuild:
    today = today or _today()
    key = _build_key(settings, start, end, today)
    use_cache = getattr(cache, "enabled", True)
    if use_cache:
        hit = net.read_pickle("build", f"full_{key}.pkl")
        if isinstance(hit, FullBuild):
            return hit
    fb = _Builder(settings, cache, net, start, end, today, spy_loader).run()
    if use_cache:
        net.write_pickle("build", f"full_{key}.pkl", fb)
        _prune_builds(net, keep=f"full_{key}.pkl")
    return fb


def _prune_builds(net: PitNet, keep: str, max_age_days: int = 3) -> None:
    """Full builds are ~100 MB and keyed by session: drop the ones older than a few days."""
    cutoff = time.time() - max_age_days * 86400
    for p in net.path("build").glob("full_*.pkl"):
        if p.name != keep and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


class _Builder:
    def __init__(self, settings, cache, net, start, end, today, spy_loader):
        self.settings: Settings = settings
        self.cache: DiskCache = cache
        self.net: PitNet = net
        self.start: pd.Timestamp = start
        self.end: pd.Timestamp = min(end, today) if end is not None else today
        self.today: pd.Timestamp = today
        self.spy_loader = spy_loader
        self.chart_start = date(min(2013, start.year - 1), 1, 1)  # >= 1 year of warm-up before the first date
        if settings.pit.us.index != "sp500" or settings.pit.rebalance != "month_end":
            raise ValueError(
                f"지원하지 않는 PIT 설정: index={settings.pit.us.index}, rebalance={settings.pit.rebalance}"
            )
        self.ftd_fill = bool(settings.pit.us.ftd_fill)  # False: FTD only validates Yahoo series, no fills/marks
        self.subs = us_cik.Submissions(net)
        self.notes: list[str] = []
        self.cov: dict[str, Any] = {}
        self.timing: dict[str, float] = {}
        self._charts: dict[str, dict] = {}

    # ------------------------------------------------------------------ helpers

    def _tick(self, label: str, t0: float) -> float:
        self.timing[label] = round(time.time() - t0, 1)
        log.info("pit-us %s: %.1fs", label, self.timing[label])
        return time.time()

    def chart(self, sym: str, open_spell: bool = False, need_events: bool = False) -> dict:
        key = f"{sym}|{int(open_spell)}|{int(need_events)}"
        if key not in self._charts:
            # open spells need the last completed session's bar: a series one session behind the production
            # provider leaves the latest cross-section without its members (RSMT-Q degraded: xs_mom_z missing,
            # seen 2026-09-27 on every US ticker). It was "5 days behind" before
            refresh = self.today if open_spell else None
            self._charts[key] = us_prices.load_chart(
                self.net, sym, self.chart_start, refresh_before=refresh, need_events=need_events
            )
        return self._charts[key]

    # ------------------------------------------------------------------ steps

    def run(self) -> FullBuild:
        t0 = t_all = time.time()
        spy = self._spy()
        self.dates = self._rebalance_dates(spy)
        if not len(self.dates):
            raise ValueError(
                f"리밸런싱 날짜 없음: {self.start.date()}~{self.end.date()} (SPY 마지막 {spy.index[-1].date()})"
            )
        self.W0, self.W1 = self.dates[0], self.dates[-1]
        t0 = self._tick("benchmark", t0)
        self._membership()
        t0 = self._tick("membership_cik", t0)
        self._ftd()
        t0 = self._tick("ftd", t0)
        self._resolve()
        t0 = self._tick("yahoo_resolve", t0)
        self._entities()
        t0 = self._tick("entities_frames", t0)
        self._member_months()
        t0 = self._tick("member_months", t0)
        self._fundamentals_and_rank()
        t0 = self._tick("fundamentals_rank", t0)
        rsp = self._rsp()
        self._tracking(rsp)
        self._meta()
        self._tick("meta_tracking", t0)
        self.timing["total"] = round(time.time() - t_all, 1)
        self.notes += [f"데이터 소스: {n}" for n in self.net.notes]
        if not self.ftd_fill:
            self.notes.append("설정 pit.us.ftd_fill=false — FTD 월말가 보간·상폐 종가 없이 Yahoo 일별 가격만 사용")
        self._persist_known()
        return FullBuild(
            dates=self.dates,
            members=self.members,
            month_src=self.month_src,
            frames=self.frames,
            benchmarks={BENCHMARK: spy},
            meta=self.meta,
            coverage=self._coverage_common(),
            notes=list(self.notes),
            diagnostics=self.diag,
        )

    def _spy(self) -> pd.DataFrame:
        if self.spy_loader is not None:
            return clean_prices(self.spy_loader(self.chart_start))
        from quantcio.data.us import BENCHMARK_NAME, USProvider

        inst = Instrument(
            ticker=BENCHMARK,
            market=Market.US,
            name="SPDR S&P 500 ETF Trust",
            exchange="NYSE Arca",
            currency="USD",
            yf_symbol=BENCHMARK,
            benchmark=BENCHMARK,
            benchmark_name=BENCHMARK_NAME,
        )
        df = USProvider(self.settings, self.cache).benchmark(inst, self.chart_start)
        out = clean_prices(df)
        out.attrs.update(df.attrs)
        return out

    def _rebalance_dates(self, spy: pd.DataFrame) -> pd.DatetimeIndex:
        """Month-end sessions in [start, end] of COMPLETE months (last session <= end and <= SPY's last bar)."""
        last = min(self.end, spy.index[-1])
        me = month_end_sessions(nyse_sessions(self.start, self.end + pd.Timedelta(40, "D")))
        return pd.DatetimeIndex(me[(me >= self.start) & (me <= last)])

    def _membership(self) -> None:
        net = self.net
        fja = us_membership.load_fja(net)
        cur, ch = us_membership.load_wiki(net)
        sp0, wiki_applied, wiki_warn = us_membership.patch_spells(fja, ch, self.today)
        ev0 = us_membership.removal_events(sp0, ch, self.W0)
        sp, manual, skipped = us_membership.apply_manual_patches(sp0)
        keep = set(zip(sp["ticker"], sp["end"], strict=False))
        alive = np.array([(t, d) in keep for t, d in zip(ev0["fja_ticker"], ev0["date"], strict=False)], dtype=bool)
        ev0 = ev0.loc[alive] if len(ev0) else ev0
        pev = us_membership.patch_events(sp, self.W0)
        ev = pd.concat([ev0, pev], ignore_index=True) if len(pev) else ev0.reset_index(drop=True)
        checks, cnotes = us_membership.membership_checks(sp, cur, self.dates)
        self.notes += cnotes + [f"Wikipedia 보정: {w}" for w in wiki_warn]
        self.cov["membership_checks"] = checks | {
            "fja_last_row": str(fja.index[-1].date()),
            "wikipedia_patches": wiki_applied,
            "manual_patches": manual,
            "manual_patches_skipped": skipped,
        }
        if (self.today - fja.index[-1]).days > 45:
            self.notes.append(
                f"fja 마지막 행 {fja.index[-1].date()} — {(self.today - fja.index[-1]).days}일 지연, Wikipedia로 보정"
            )
        # CIK of ended spells (events)
        need = ev[ev["cik"].isna()]
        if len(need):
            pairs = us_cik.sec_ticker_history(net, self.today)
            lookup = us_cik.cik_lookup_index(net)
            wd = us_cik.wikidata_ciks(net, need["link"].dropna().tolist())
            for i, r in need.iterrows():
                c, m = us_cik.cik_for_event(r, pairs, lookup, wd, self.subs)
                ev.loc[i, "cik"] = float(c) if c else np.nan
                ev.loc[i, "cik_method"] = m
        self.events = ev
        # spells overlapping the window -> CIK / display ticker / FTD symbol / name
        win = sp[(sp["start"] <= self.W1) & (sp["end"].isna() | (sp["end"] > self.W0))].copy()
        cur_cik = {s: int(c) for s, c in zip(cur["sym"], cur["cik"], strict=False) if pd.notna(c)}
        cur_name = dict(zip(cur["sym"], cur["security"], strict=False))
        sec_cur = {t: (c, title) for t, c, title in us_cik.current_company_tickers(net)}
        evk = {(r["fja_ticker"], r["date"]): i for i, r in ev.iterrows()}
        rows = []
        for i, s in win.iterrows():
            tk = us_membership.dash(s["ticker"])
            e = evk.get((s["ticker"], s["end"])) if pd.notna(s["end"]) else None
            er = ev.loc[e] if e is not None else None
            if pd.notna(s["cik"]):
                cik, how = float(s["cik"]), "manual_patch"
            elif pd.isna(s["end"]):
                if tk in cur_cik:
                    cik, how = float(cur_cik[tk]), "wiki_current"
                elif tk in sec_cur:
                    cik, how = float(sec_cur[tk][0]), "sec_current"
                else:
                    cik, how = np.nan, "unmapped"
            elif er is not None:
                cik, how = er["cik"], er["cik_method"]
            else:
                cik, how = np.nan, "unmapped"
            disp = us_membership.dash(s["ftd_sym"]) if isinstance(s["ftd_sym"], str) and s["ftd_sym"] else None
            disp = disp or (str(er["ticker"]) if er is not None else tk)
            name = s["name"] if isinstance(s["name"], str) else None
            name = name or (er["name"] if er is not None and isinstance(er["name"], str) else None)
            name = name or cur_name.get(tk) or (sec_cur.get(tk, (None, None))[1] if pd.isna(s["end"]) else None)
            if not name and pd.notna(cik) and pd.notna(s["end"]):  # fja-only renames have no Wikipedia row
                name = _sec_name_at(self.subs.get(int(cik)) or {}, s["end"])
            rows.append(
                dict(
                    spell=i,
                    cik=cik,
                    cik_method=how,
                    disp=disp,
                    ftd_sym=us_membership.ftd_symbol(disp),
                    sname=name,
                    ev=e if e is not None else -1,
                    cat=(er["cat"] if er is not None else ("current" if pd.isna(s["end"]) else "unknown")),
                    reason=(er["reason"] if er is not None else None),
                )
            )
        extra = pd.DataFrame(rows).set_index("spell")
        win = win.drop(columns=["cik", "ftd_sym", "reason"]).join(extra)
        self.win = win
        self.current = cur
        methods = Counter(win["cik_method"])
        unm = win[win["cik"].isna()]
        self.cov["cik_methods"] = dict(methods)
        self.cov["unmapped_spells"] = [
            f"{r['ticker']} {r['start'].date()}~{r['end'].date() if pd.notna(r['end']) else '현재'}"
            for _, r in unm.iterrows()
        ]

    def _ftd(self) -> None:
        syms = set(self.win["ftd_sym"])
        lo = self.W0 - pd.Timedelta(60, "D")
        hi = min(self.today, self.W1 + pd.Timedelta(75, "D"))
        panel, missing = us_prices.ftd_panel(self.net, syms, lo, hi)
        self.ftd_missing = missing
        self.ftd_by = {k: g.reset_index(drop=True) for k, g in panel.groupby("key")} if len(panel) else {}
        self.cov["ftd"] = {
            "rows": len(panel),
            "symbols_with_rows": len(self.ftd_by),
            "symbols_needed": len(syms),
            "files_missing": [k for k, _ in missing],
            "last_settle": str(panel["settle"].max().date()) if len(panel) else None,
        }
        if missing:
            self.notes.append(f"SEC FTD 파일 {len(missing)}개 수집 실패: {[k for k, _ in missing][:6]}")

    def own_ftd(self, sym: str) -> pd.DataFrame | None:
        f = self.ftd_by.get(sym)
        if f is None:
            return None
        return f[f["symbol"] == f["key"]]

    def _resolve(self) -> None:
        cur_by_cik = us_cik.sec_current_tickers_by_cik(self.net)
        ev = self.events
        rename_map = {
            us_membership.dash(r["fja_ticker"]): r["rename_to"] for _, r in ev.dropna(subset=["rename_to"]).iterrows()
        }
        out, rejected, stubs = [], [], []
        for i, s in self.win.iterrows():
            if pd.isna(s["cik"]):
                out.append((None, "no_cik"))
                continue
            er = ev.loc[s["ev"]] if s["ev"] >= 0 else None
            if er is not None and er.get("patch"):
                er = None  # patch events carry no rename chain; the hint is on the spell
            open_spell = pd.isna(s["end"])
            sym, how, rej = us_prices.resolve_yahoo(
                s,
                er,
                int(s["cik"]),
                cur_by_cik,
                rename_map,
                self.own_ftd(s["ftd_sym"]),
                lambda x, o=open_spell: self.chart(x, open_spell=o),
                self.subs.names,
                self.today,
                self.W0,
            )
            out.append((sym, how))
            rejected += [(s["ticker"], x, why) for x, why in rej if why.startswith("ftd_issuer")]
            stubs += [f"{s['ticker']}->{x} {why}" for x, why in rej if why.startswith("stub_series")]
        self.win["ysym"] = [o[0] for o in out]
        self.win["yhow"] = [o[1] for o in out]
        self.rejected = rejected
        self.cov["yahoo_resolution"] = dict(Counter(self.win["yhow"]))
        self.cov["stub_series_rejected"] = sorted(set(stubs))
        if stubs:
            self.notes.append(
                f"Yahoo 잔존 시계열(편입 구간 내 {us_prices.MIN_PARTIAL_BARS}봉 미만) {len(set(stubs))}건 거부 — "
                f"FTD 월말가로 대체: {sorted(set(stubs))[:4]}"
            )

    # ------------------------------------------------------------------ entities and frames

    def _entities(self) -> None:
        win = self.win[self.win["cik"].notna()].copy()
        win["entity"] = win["cik"].map(us_cik.cik_entity)
        self.win.loc[win.index, "entity"] = win["entity"]
        mm = us_membership.members_matrix(win, self.dates)
        mm["entity"] = win.loc[mm["spell"], "entity"].to_numpy() if len(mm) else []
        self.mm = mm  # date, spell, entity
        self.mm_spells = mm.groupby(["date", "entity"])["spell"].apply(list).to_dict() if len(mm) else {}
        frames: dict[str, pd.DataFrame] = {}
        raw: dict[str, pd.DataFrame] = {}  # chosen chart df (split-adjusted close + volume), masked
        splits: dict[str, pd.Series] = {}
        info: dict[str, dict] = {}
        masked_total, suspect = 0, []
        by_sym_entities: dict[str, set] = {}
        for e, g in win.groupby("entity"):
            for s in g["ysym"].dropna():
                by_sym_entities.setdefault(s, set()).add(e)
        for e, g in win.groupby("entity"):
            dates_e = pd.DatetimeIndex(sorted(mm.loc[mm["entity"] == e, "date"].unique()))
            cands = [s for s in dict.fromkeys(g["ysym"].dropna())]
            best, best_score = None, None
            for s in cands:
                c = self.chart(s, open_spell=bool(g.loc[g["ysym"] == s, "end"].isna().any()))
                if not c.get("ok"):
                    continue
                idx = c["df"].index
                pos = idx.searchsorted(dates_e, side="right") - 1
                okp = pos >= 0
                cover = int(
                    ((dates_e[okp] - idx[pos[okp]]) <= pd.Timedelta(PRICE_MAX_AGE_DAYS, "D")).sum() if len(pos) else 0
                )
                dv = float((c["df"]["close"] * c["df"]["volume"]).tail(250).median())
                score = (cover, dv)
                if best_score is None or score > best_score:
                    best, best_score = s, score
            ent_info: dict[str, Any] = {"ysym": best, "partial": False, "masked_rows": 0}
            if best is not None:
                c = self.chart(best, need_events=True)
                if not c.get("ok"):  # refetch failed: fall back to the cached copy used for resolution
                    c = self.chart(best, open_spell=True)
                df = c["df"][c["df"].index <= self.today].copy()  # never an unfinished intraday bar
                bad: set = set()
                good: set = set()
                for _, s in g[g["ysym"] == best].iterrows():
                    f = self.own_ftd(s["ftd_sym"])
                    lo = max(s["start"], self.W0)
                    hi = s["end"] if pd.notna(s["end"]) else self.today
                    b, gd = _month_flags(df, f, lo, hi)
                    bad |= b
                    good |= gd
                    ent_info["partial"] |= str(s["yhow"]).endswith("_partial")
                df, minfo = _mask(df, bad, good)
                masked_total += minfo["dropped_rows"]
                ent_info["masked_rows"] = minfo["dropped_rows"]
                ent_info["mask_policy"] = minfo["policy"]
                ent_info["bad_months"] = sorted(str(m.date()) for m in bad)
                if minfo["policy"] == "suspect_kept":
                    suspect.append(f"{e}({best}:{len(bad)})")
                # the series continues under a later CIK (lineage): this entity's frame ends at its last spell end
                later = [
                    o
                    for o in by_sym_entities.get(best, set())
                    if o != e and win.loc[win["entity"] == o, "start"].min() >= g["end"].max()
                ]
                if later and g["end"].notna().all():
                    cut = g["end"].max()
                    df = df[df.index < cut]
                    ent_info["cut_for_successor"] = sorted(later)
                validated = bool(good)  # >= 1 month of FTD prints on a plateau with this series
                ent_info["validated"] = validated
                if len(df):
                    frame = us_prices.price_frame(df, adjusted=validated)
                    big = _large_distributions(df)
                    frame.attrs.update(
                        source="yahoo_chart",
                        symbol=best,
                        partial=bool(ent_info["partial"]),
                        masked_rows=int(minfo["dropped_rows"]),
                        split_adjusted=True,
                        validated=validated,
                        large_distributions=big,
                    )
                    if big:
                        ent_info["large_distributions"] = big
                    warn = []
                    if not validated:
                        warn.append("SEC FTD 대조 근거 부족(검증 불가) — 분할만 반영, 배당 미조정 종가 사용")
                    elif not frame.attrs["adjusted"]:
                        warn.append("수정주가(adjclose) 없음 — 분할만 반영, 배당 미조정")
                    if big:
                        warn.append(f"하루 30% 넘는 배당/합병 현금 조정 {len(big)}건(계열 시계열 가능성): {big[:3]}")
                    if ent_info["partial"]:
                        warn.append("Yahoo 시계열이 편입 구간 시작을 덮지 못함(부분 시계열)")
                    if minfo["dropped_rows"]:
                        warn.append(f"FTD 대조로 다른 발행사 가격 {minfo['dropped_rows']}행 제거")
                    frame.attrs["warnings"] = warn
                    frames[e] = frame
                    raw[e] = df
                    sp_ = c.get("splits")
                    splits[e] = sp_ if isinstance(sp_, pd.Series) else pd.Series(dtype="float64")
                    if c.get("splits") is None:
                        self.net.note(f"{best}: Yahoo 분할 이벤트 없음(캐시) — 시가총액 분할 보정 생략")
            info[e] = ent_info
        self.ent_win = win
        self.frames, self.raw, self.splits, self.ent_info = frames, raw, splits, info
        self.cov["masked_rows"] = masked_total
        self.cov["suspect_months_kept"] = suspect

    # ------------------------------------------------------------------ member-months

    def _member_months(self) -> None:
        dates = self.dates
        nxt = dict(zip(dates[:-1], dates[1:], strict=False))
        win = self.ent_win
        rows = []
        capped: list[str] = []
        splitlike: list[str] = []
        arrays = {
            e: (df.index.values, df["close"].to_numpy(), df.index[-1]) for e, df in self.frames.items() if len(df)
        }
        for (d, e), spells in sorted(self.mm_spells.items()):
            d1 = nxt.get(d, self.today)
            src, ret = "missing", np.nan
            a = arrays.get(e)
            if a is not None:
                idx, close, _last = a
                i0 = np.searchsorted(idx, np.datetime64(d), side="right") - 1
                if i0 >= 0 and (d - pd.Timestamp(idx[i0])).days <= PRICE_MAX_AGE_DAYS:
                    i1 = np.searchsorted(idx, np.datetime64(d1), side="right") - 1
                    if i1 > i0:
                        src, ret = "yahoo", close[i1] / close[i0] - 1
                    elif d1 >= self.today - pd.Timedelta(3, "D") or d == dates[-1]:
                        src = "yahoo"  # last period: no bar after d yet
            if src == "missing" and self.ftd_fill:
                for sp_i in spells:
                    s = win.loc[sp_i]
                    f = self.own_ftd(s["ftd_sym"])
                    end = s["end"] if pd.notna(s["end"]) and s["end"] <= d1 else None
                    failed = s["cat"] == "failed"
                    if end is not None and failed:
                        r = self._terminal_return(s, f, d)
                    else:  # a halted failing firm keeps its last print for up to 45 days; its crashes are real
                        r = us_prices.ftd_month_return(f, d, d1, end, max_age_days=45 if failed else 7, cap=False)
                        lo, hi = us_prices.FTD_MOVE_RANGE
                        if r is not None and not failed and not (lo < r < hi):
                            capped.append(f"{s['disp']} {d1.date()} {r:+.1%}")
                            r = None
                        elif r is not None and us_prices.split_like(r):
                            splitlike.append(f"{s['disp']} {d1.date()} {r:+.1%}")
                    if r is not None:
                        src, ret = "ftd", r
                        break
            rows.append((d, e, src, ret))
        self.month_src = pd.DataFrame(rows, columns=["date", "entity", "src", "ret"])
        self.cov["ftd_moves_rejected"] = {
            "count": len(capped),
            "examples": capped[:20],
            "range": us_prices.FTD_MOVE_RANGE,
        }
        self.cov["ftd_splitlike_moves_kept"] = {"count": len(splitlike), "examples": splitlike[:20]}
        if capped:
            self.notes.append(
                f"FTD 월간 변동 {len(capped)}건이 허용 범위({us_prices.FTD_MOVE_RANGE[0]:+.0%}~"
                f"{us_prices.FTD_MOVE_RANGE[1]:+.0%}) 밖이라 결측 처리(미조정 분할·오류 가격 가능성): {capped[:3]}"
            )

    def _terminal_return(self, s: pd.Series, f: pd.DataFrame | None, d: pd.Timestamp) -> float | None:
        p0 = us_prices.ftd_price_at(f, d, 45) if f is not None else None
        if p0 is None:
            return None
        mark = us_prices.terminal_mark(self.ftd_by.get(s["ftd_sym"]), s["end"], failed=True)
        if mark is None:
            return None
        return mark[1] / p0[1] - 1

    # ------------------------------------------------------------------ ranking

    def _fundamentals_and_rank(self) -> None:
        ents = sorted(self.ent_win["entity"].unique())
        chart_from = {e: self.raw[e].index[0] for e in self.raw}
        delist = self._delist_dates()
        self.delist = delist
        t0 = time.time()
        try:
            fund = _fundamentals(
                self.settings,
                self.net,
                ents,
                self.splits,
                chart_from,
                delist,
                "2009-06-01",
                None,
            )
            self.fund = fund
        except Exception as exc:  # noqa: BLE001 - ranking falls back to dollar volume, loudly
            log.exception("pit-us fundamentals failed")
            self.fund = None
            self.notes.append(
                f"XBRL 주식수 로드 실패({type(exc).__name__}: {str(exc)[:80]}) — 전 기간 60일 평균 거래대금으로 순위"
            )
        self.timing["fundamentals"] = round(time.time() - t0, 1)
        mm = self.mm.drop_duplicates(["date", "entity"])[["date", "entity"]].copy()
        shares = self._shares_asof(mm)
        vals = []
        for (d, e), sh in zip(zip(mm["date"], mm["entity"], strict=False), shares, strict=False):
            vals.append(self._value_at(d, e, sh))
        mm["mcap"] = [v[0] for v in vals]
        mm["adv"] = [v[1] for v in vals]
        mm["value_source"] = [v[2] for v in vals]
        out = []
        basis_count: Counter = Counter()
        imputed = missing = 0
        for d, g in mm.groupby("date"):
            g = g.copy()
            share = g["mcap"].notna().mean()
            if share >= MIN_SHARE_COVERAGE:
                ratio = (g["mcap"] / g["adv"]).replace([np.inf, -np.inf], np.nan).dropna()
                med = float(ratio.median()) if len(ratio) else np.nan
                g["rank_value"] = g["mcap"]
                g["rank_basis"] = "mcap"
                need = g["mcap"].isna() & g["adv"].notna() & np.isfinite(med)
                g.loc[need, "rank_value"] = g.loc[need, "adv"] * med
                g.loc[need, "rank_basis"] = "adv_imputed_mcap"
                imputed += int(need.sum())
                basis_count["mcap"] += 1
            else:
                g["rank_value"] = g["adv"]
                g["rank_basis"] = "adv"
                basis_count["adv"] += 1
            missing += int(g["rank_value"].isna().sum())
            g = g.sort_values(["rank_value", "entity"], ascending=[False, True], na_position="last")
            g["rank"] = np.arange(1, len(g) + 1)
            out.append(g)
        members = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        disp = self._display(members)
        members["ticker"] = [disp[(d, e)][0] for d, e in zip(members["date"], members["entity"], strict=False)]
        members["name"] = [disp[(d, e)][1] for d, e in zip(members["date"], members["entity"], strict=False)]
        self.members = members
        self.cov["rank"] = {
            "dates_by_basis": dict(basis_count),
            "imputed_from_dollar_volume": imputed,
            "rank_value_missing": missing,
            "value_sources": dict(Counter(mm["value_source"])),
        }
        if basis_count.get("adv"):
            self.notes.append(
                f"주식수(XBRL) 확보율 {int(MIN_SHARE_COVERAGE * 100)}% 미만인 월말 {basis_count['adv']}회는 "
                f"시가총액 대신 {ADV_SESSIONS}일 평균 거래대금으로 순위"
            )
        if imputed:
            self.notes.append(
                f"주식수 없는 회원-월 {imputed}개는 {ADV_SESSIONS}일 평균 거래대금 × 해당 월말 "
                "시가총액/거래대금 중앙값으로 시가총액을 추정해 순위 (다중 클래스·XBRL 미보유 종목 등)"
            )

    def _delist_dates(self) -> dict[str, pd.Timestamp]:
        """Entity -> last traded session for entities that stopped trading: the Yahoo frame ends > 10 days before
        today, or (no frame) the removal was an acquisition/failure or the FTD prints of the ticker stop within
        30 days of the removal."""
        out = {}
        ftd_last = self.cov.get("ftd", {}).get("last_settle")
        ftd_last = pd.Timestamp(ftd_last) if ftd_last else None
        for e, g in self.ent_win.groupby("entity"):
            if g["end"].isna().any():
                continue
            df = self.raw.get(e)
            last = self._last_spell(g)
            end = last["end"]
            if df is not None and len(df):
                if df.index[-1] < self.today - pd.Timedelta(10, "D"):
                    out[e] = df.index[-1]
                continue
            # no daily series: acquisition/failure, or the CUSIP printed last before removal stops printing (a
            # reused symbol with a new CUSIP - Actavis taking 'AGN' - does not count as still trading)
            f = self.own_ftd(last["ftd_sym"])
            pre = f[f["px_date"] <= end] if f is not None else None
            traded = pre is not None and len(pre) > 0
            quiet = True
            if traded:
                cus = pre["cusip"].iloc[-1]
                allrows = self.ftd_by.get(last["ftd_sym"])
                quiet = not bool(((allrows["cusip"] == cus) & (allrows["px_date"] > end + pd.Timedelta(30, "D"))).any())
            stopped = traded and quiet and ftd_last is not None and end + pd.Timedelta(45, "D") < ftd_last
            if last["cat"] in ("acquired", "failed") or stopped:
                out[e] = end  # the removal (effective) date; the last FTD print is the terminal mark's date
        return out

    def _last_spell(self, g: pd.DataFrame) -> pd.Series:
        """The entity's last spell; among spells ending together (share classes) the one printing last on SEC FTD."""
        g = g[g["end"] == g["end"].max()] if g["end"].notna().all() else g[g["end"].isna()]
        if len(g) == 1:
            return g.iloc[0]
        best, best_d = g.iloc[0], pd.Timestamp.min
        for _, s in g.iterrows():
            f = self.own_ftd(s["ftd_sym"])
            if f is not None and len(f):
                d = f.loc[f["px_date"] <= (s["end"] if pd.notna(s["end"]) else self.today), "px_date"].max()
                if pd.notna(d) and d > best_d:
                    best, best_d = s, d
        return best

    def _shares_asof(self, mm: pd.DataFrame) -> list[tuple[float, pd.Timestamp] | None]:
        if self.fund is None or self.fund.panel.empty:
            return [None] * len(mm)
        p = self.fund.panel[["entity", "available_from", "period_end", "shares_out", "filed"]].dropna(
            subset=["shares_out"]
        )
        p = p.sort_values("available_from")
        left = mm.reset_index().sort_values("date")
        j = pd.merge_asof(
            left, p, left_on="date", right_on="available_from", by="entity", direction="backward"
        ).set_index("index")
        stale = (j["date"] - j["period_end"]).dt.days > self.settings.pit.fundamentals.max_staleness_days
        j.loc[stale, "shares_out"] = np.nan
        j = j.loc[mm.index]
        return [
            (float(s), pd.Timestamp(f)) if pd.notna(s) else None
            for s, f in zip(j["shares_out"], j["filed"], strict=False)
        ]

    def _value_at(self, d: pd.Timestamp, e: str, sh: tuple[float, pd.Timestamp] | None) -> tuple[float, float, str]:
        """(PIT market cap or NaN, 60-session average dollar volume or NaN, price source)."""
        df = self.raw.get(e)
        mcap, adv, src = np.nan, np.nan, "none"
        if df is not None and len(df):
            i = df.index.searchsorted(d, side="right") - 1
            if i >= 0 and (d - df.index[i]).days <= PRICE_MAX_AGE_DAYS:
                src = "yahoo"
                px = float(df["close"].iloc[i])
                w = df.iloc[max(0, i - ADV_SESSIONS + 1) : i + 1]
                if len(w) >= 20:
                    adv = float((w["close"] * w["volume"]).mean())
                if sh is not None:
                    shares, filed = sh
                    spl = self.splits.get(e)
                    mult = float(np.prod(spl[spl.index > filed].to_numpy())) if spl is not None and len(spl) else 1.0
                    mcap = px * shares * mult
                return mcap, adv, src
        for sp_i in self.mm_spells.get((d, e), []):
            x = us_prices.ftd_price_at(self.own_ftd(self.ent_win.loc[sp_i, "ftd_sym"]), d)
            if x is not None:
                src = "ftd"
                if sh is not None:
                    mcap = x[1] * sh[0]
                break
        return mcap, adv, src

    def _display(self, members: pd.DataFrame) -> dict[tuple, tuple[str, str]]:
        win = self.ent_win
        out = {}
        for (d, e), spells in self.mm_spells.items():
            rows = win.loc[spells]
            ysym = self.ent_info.get(e, {}).get("ysym")
            pick = rows[rows["ysym"] == ysym] if ysym is not None else rows.iloc[0:0]
            r = (pick if len(pick) else rows.sort_values("disp")).iloc[0]
            out[(d, e)] = (_txt(r["disp"]), _txt(r["sname"], _txt(r["disp"])))
        return out

    # ------------------------------------------------------------------ tracking check

    def _rsp(self) -> pd.Series | None:
        c = self.chart(EW_BENCHMARK, open_spell=True)
        if not c.get("ok"):
            self.notes.append(f"RSP 가격 수집 실패({c.get('err')}) — 동일가중 추적 점검 생략")
            return None
        return c["df"]["adjclose"].dropna()

    def _tracking(self, rsp: pd.Series | None) -> None:
        ms = self.month_src
        dates = self.dates
        res: dict[str, Any] = {"universe": "S&P 500 전 구성종목(CIK 동일가중)", "benchmark": EW_BENCHMARK}
        if rsp is None or len(dates) < 13:
            res["ok"] = None
            res["reason"] = "RSP 없음 또는 기간 13개월 미만"
            self.cov["tracking_check"] = res
            return
        rows = []
        for d0, d1 in zip(dates[:-1], dates[1:], strict=False):
            g = ms[ms["date"] == d0]
            y = g.loc[g["src"] == "yahoo", "ret"].dropna()
            yf = g.loc[g["src"].isin(["yahoo", "ftd"]), "ret"].dropna()
            i0 = rsp.index.searchsorted(d0, side="right") - 1
            i1 = rsp.index.searchsorted(d1, side="right") - 1
            r = rsp.iloc[i1] / rsp.iloc[i0] - 1 if i0 >= 0 and i1 > i0 else np.nan
            rows.append(dict(month=d1, n=len(g), ew_y=y.mean(), ew_yf=yf.mean(), rsp=r))
        bt = pd.DataFrame(rows).set_index("month")

        def ann(s: pd.Series) -> float:
            s = s.dropna()
            return float((1 + s).prod() ** (12 / len(s)) - 1) if len(s) else np.nan

        both = bt.dropna(subset=["ew_yf", "rsp"])
        gap = ann(both["ew_yf"]) - ann(both["rsp"])
        te = float((both["ew_yf"] - both["rsp"]).std() * np.sqrt(12))
        gap_y = ann(bt.dropna(subset=["ew_y", "rsp"])["ew_y"]) - ann(bt.dropna(subset=["ew_y", "rsp"])["rsp"])
        ok = bool(abs(gap) <= TRACKING_MAX_GAP and te <= TRACKING_MAX_TE)
        res.update(
            months=len(both),
            first=str(both.index[0].date()) if len(both) else None,
            last=str(both.index[-1].date()) if len(both) else None,
            ann_pit_yahoo=round(ann(bt["ew_y"]), 5),
            ann_pit_yahoo_ftd=round(ann(both["ew_yf"]), 5),
            ann_rsp=round(ann(both["rsp"]), 5),
            gap=round(gap, 5),
            gap_yahoo_only=round(gap_y, 5),
            te=round(te, 5),
            max_gap=TRACKING_MAX_GAP,
            max_te=TRACKING_MAX_TE,
            ok=ok,
        )
        self.cov["tracking_check"] = res
        self.tracking_bt = bt
        msg = (
            f"추적 점검(동일가중 PIT S&P 500 vs RSP, {res['first']}~{res['last']} {len(both)}개월): "
            f"PIT 연 {_pct(res['ann_pit_yahoo_ftd'])} vs RSP {_pct(res['ann_rsp'])}, 차이 {100 * gap:+.2f}%p, "
            f"TE {_pct(te)} — {'기준 충족' if ok else '경고: 기준(0.5%p, TE 1%) 초과'}"
        )
        self.notes.append(msg)

    # ------------------------------------------------------------------ meta

    def _meta(self) -> None:
        win = self.ent_win
        rows = []
        term_counts: Counter = Counter()
        span = self.mm.groupby("entity")["date"].agg(["min", "max"])
        ftd_entities = set(self.month_src.loc[self.month_src["src"] == "ftd", "entity"])
        for e, g in win.groupby("entity"):
            g = g.sort_values("start")
            last = g.iloc[-1]
            info = self.ent_info.get(e, {})
            frame = self.frames.get(e)
            chart = self.chart(info["ysym"]) if info.get("ysym") else None
            is_open = g["end"].isna().any()
            delisted = e in self.delist
            delist_date = self.delist.get(e)
            term = self._terminal(e, g, frame, delisted)
            if term is not None:
                term_counts[term[2]] += 1
            f_first = frame.index[0] if frame is not None and len(frame) else pd.NaT
            f_last = frame.index[-1] if frame is not None and len(frame) else pd.NaT
            if frame is None:  # FTD prints of each spell's symbol inside that spell (symbols get reused)
                parts = []
                for _, s in g.iterrows():
                    x = self.own_ftd(s["ftd_sym"])
                    if x is not None and len(x):
                        hi = s["end"] if pd.notna(s["end"]) else self.today + pd.Timedelta(1, "D")
                        parts.append(x[(x["px_date"] >= s["start"]) & (x["px_date"] < hi)])
                ftd_rows = pd.concat(parts) if parts else None
                if ftd_rows is not None and len(ftd_rows):
                    f_first, f_last = ftd_rows["px_date"].min(), ftd_rows["px_date"].max()
            price_source = "none"
            if frame is not None:
                price_source = "yahoo_partial" if info.get("partial") or info.get("masked_rows") else "yahoo"
            elif e in ftd_entities:
                price_source = "ftd"
            notes = []
            if info.get("masked_rows"):
                notes.append(f"FTD 불일치 {len(info.get('bad_months', []))}개월 마스킹({info.get('mask_policy')})")
            if info.get("cut_for_successor"):
                notes.append(f"후속 CIK {','.join(info['cut_for_successor'])}로 이어지는 시계열 — 편출일에서 절단")
            if set(g["cik_method"]) & {"m1_unvalidated", "m3_name", "m2_wikidata"}:
                notes.append(f"CIK 경로 {sorted(set(g['cik_method']))}")
            reason_ko = {
                "acquired": "인수·합병",
                "failed": "파산·영업정지",
                "renamed": "상호·티커 변경/재편",
                "demoted": "지수 제외(시총 등)",
                "unknown": "불명",
                "current": "",
            }
            rows.append(
                dict(
                    entity=e,
                    ticker=_txt(last["disp"]),
                    name=_txt(last["sname"], _txt(last["disp"])),
                    market=us_prices.chart_exchange(chart) if chart and chart.get("ok") else "US",
                    first_date=f_first,
                    last_date=f_last,
                    delisted=bool(delisted),
                    delist_date=delist_date if delisted else pd.NaT,
                    delist_reason=reason_ko.get(str(last["cat"]), str(last["cat"])) if not is_open else "",
                    price_source=price_source,
                    cik=int(e[3:]),
                    notes="; ".join(notes),
                    terminal_date=term[0] if term else pd.NaT,
                    terminal_price=term[1] if term else np.nan,
                    terminal_source=term[2] if term else None,
                    terminal_basis=term[3] if term else None,
                    yahoo_symbol=info.get("ysym"),
                    cik_method=",".join(sorted(set(map(str, g["cik_method"])))),
                    spells=len(g),
                    first_member=span.loc[e, "min"] if e in span.index else pd.NaT,
                    last_member=span.loc[e, "max"] if e in span.index else pd.NaT,
                    removal_reason=_txt(last["reason"]) if not is_open else "",
                )
            )
        self.meta = pd.DataFrame(rows)
        self.cov["terminal_marks"] = dict(term_counts)
        self.diag = {"rejected": self.rejected, "ent_info": self.ent_info}

    def _terminal(self, e: str, g: pd.DataFrame, frame: pd.DataFrame | None, delisted: bool):
        """(date, price, source, basis) for delisted entities (see :func:`terminal_price`)."""
        if not delisted:
            return None
        last = self._last_spell(g)
        f = self.ftd_by.get(last["ftd_sym"])
        failed = last["cat"] == "failed"
        removal = last["end"] if pd.notna(last["end"]) else self.today
        mark = us_prices.terminal_mark(f, removal, failed) if self.ftd_fill else None
        if frame is not None and len(frame):
            fl = frame.index[-1]
            later = mark is not None and (mark[0] > fl + pd.Timedelta(3, "D") or (mark[2] != "ftd" and mark[0] > fl))
            if later:  # the FTD mark is after the last Yahoo bar: put the raw price on the frame's (TR) basis
                raw = self.raw[e]
                basis = float(frame["close"].iloc[-1] / raw["close"].iloc[-1])
                return mark[0], mark[1] * basis, mark[2], "frame"
            return fl, float(frame["close"].iloc[-1]), "yahoo_last_bar", "frame"
        if mark is None:
            return None
        return mark[0], mark[1], mark[2], "raw"

    # ------------------------------------------------------------------ coverage and persistence

    def _coverage_common(self) -> dict[str, Any]:
        cov = dict(self.cov)
        cov["rejected_series"] = len(self.rejected)
        cov["rejected_examples"] = [f"{t}->{s} {why}" for t, s, why in self.rejected[:30]]
        cov["partial_series"] = sorted(e for e, i in self.ent_info.items() if i.get("partial"))
        cov["masked_entities"] = sorted(e for e, i in self.ent_info.items() if i.get("masked_rows"))
        cov["unvalidated_frames"] = sorted(
            e for e, i in self.ent_info.items() if i.get("ysym") and not i.get("validated") and e in self.frames
        )
        cov["large_distribution_events"] = {
            e: i["large_distributions"] for e, i in self.ent_info.items() if i.get("large_distributions")
        }
        cov["frames_cut_for_successor"] = {
            e: i["cut_for_successor"] for e, i in self.ent_info.items() if i.get("cut_for_successor")
        }
        cov["timing_sec"] = dict(self.timing)
        cov["net"] = dict(self.net.stats)
        cov["source_notes"] = list(self.net.notes)
        cov["rebalance"] = {"first": str(self.W0.date()), "last": str(self.W1.date()), "count": len(self.dates)}
        um = self.win[self.win["cik"].isna()]
        cov["unmapped_member_months"] = int(
            sum(len(us_membership.members_at(um, d)) for d in self.dates) if len(um) else 0
        )
        if self.fund is not None:
            cov["fundamentals"] = self.fund.coverage
        return cov

    def _persist_known(self) -> None:
        """Entity list, delisting dates and split series for load_us_pit_fundamentals(entities=None)."""
        try:
            self.net.write_json(
                "build",
                "known_entities.json",
                {
                    "built": str(self.today.date()),
                    "entities": sorted(self.ent_win["entity"].unique()),
                    "delist": {e: str(d.date()) for e, d in self.delist.items()},
                },
            )
            self.net.write_pickle(
                "build",
                "entity_splits.pkl",
                {"splits": self.splits, "chart_from": {e: df.index[0] for e, df in self.raw.items() if len(df)}},
            )
        except OSError as exc:
            log.warning("pit-us: could not persist known entities: %s", exc)


def _sec_name_at(sub: dict, when: pd.Timestamp) -> str | None:
    """Issuer name in force on ``when`` from submissions formerNames (from/to), else the current name."""
    for f in sub.get("formerNames") or []:
        try:
            lo, hi = pd.Timestamp(f.get("from")).tz_localize(None), pd.Timestamp(f.get("to")).tz_localize(None)
        except (TypeError, ValueError):
            continue
        if lo <= when <= hi and f.get("name"):
            return str(f["name"])
    return sub.get("name") or None


def _large_distributions(df: pd.DataFrame, threshold: float = 0.3) -> list[str]:
    """Days where Yahoo's adjclose/close factor steps by > ``threshold`` (merger cash booked as a distribution, or a
    lineage splice: CCEP 2016-05-31, DD/PSKY predecessors). Reported, not changed."""
    if "adjclose" not in df or df["adjclose"].isna().any() or len(df) < 2:
        return []
    f = df["adjclose"] / df["close"]
    step = (f / f.shift(1) - 1).abs()
    return [str(d.date()) for d in step[step > threshold].index]


def _month_flags(df: pd.DataFrame, f: pd.DataFrame | None, lo, hi) -> tuple[set, set]:
    """(bad months, good months) of a Yahoo series vs the FTD prices of the ticker traded then (month-end dates)."""
    if f is None or df is None or not len(f):
        return set(), set()
    n, _, okf = us_prices.ftd_consistency(us_prices.ftd_ratio(df, f[(f["settle"] > lo) & (f["settle"] <= hi)]))
    if n < 5:
        return set(), set()
    g = okf.groupby(okf.index.to_period("M")).agg(["size", "mean"])
    bad = {p.to_timestamp(how="end").normalize() for p, r in g.iterrows() if r["size"] >= 2 and r["mean"] < 0.5}
    good = {p.to_timestamp(how="end").normalize() for p, r in g.iterrows() if r["size"] >= 2 and r["mean"] >= 0.5}
    return bad, good


def _mask(df: pd.DataFrame, bad: set, good: set) -> tuple[pd.DataFrame, dict]:
    """Drop FTD-contradicted Yahoo rows. Up to ``MASK_MIN_BAD_MONTHS - 1`` bad months -> kept and counted (false
    positives: SPGI 2016-04, the first FTD month of the new symbol, would otherwise have dropped 3 years of good
    history; ARNC); a run of bad months before every good month (another issuer's history spliced in front: JCI/CB
    pre-2016 24-31 months, IR 2017-20 34) -> drop everything up to the last bad month; else drop those months."""
    if not bad:
        return df, {"dropped_rows": 0, "policy": None}
    if len(bad) < MASK_MIN_BAD_MONTHS:
        return df, {"dropped_rows": 0, "policy": "suspect_kept"}
    last_bad = max(bad)
    earlier_good = [m for m in good if m < last_bad]
    if not earlier_good:
        cut = df.index > last_bad
        return df[cut], {"dropped_rows": int((~cut).sum()), "policy": "prefix_dropped"}
    months = df.index.to_period("M").to_timestamp(how="end").normalize()
    drop = months.isin(list(bad))
    out = df[~drop]
    out.attrs["gaps"] = sorted(str(m.date()) for m in bad)
    return out, {"dropped_rows": int(drop.sum()), "policy": "months_dropped"}


# ============================================================================ top-n selection


def select_top_n(full: FullBuild, n: int) -> PITUniverse:
    """PITUniverse of the top-``n`` members per date (``n >= 500`` = every member) from a full build."""
    mem = full.members
    sel = mem if n >= 500 else mem[mem["rank"] <= n]
    membership = sel[["date", "entity", "ticker", "name", "rank", "rank_value"]].copy()
    membership = membership.sort_values(["date", "rank"]).reset_index(drop=True)
    membership = membership[MEMBERSHIP_COLUMNS]
    ents = set(membership["entity"])
    prices = {e: full.frames[e] for e in sorted(ents) if e in full.frames}
    ms = full.month_src.merge(membership[["date", "entity"]], on=["date", "entity"], how="inner")
    cov = dict(full.coverage)
    cov.update(_month_coverage(ms))
    cov["n"] = n
    cov["no_price_entities"] = sorted(ents - set(prices))
    cov["entities"] = len(ents)
    cov["entities_with_prices"] = len(prices)
    if n < 500:  # members without any rank value are ranked last and can be cut wrongly: counted
        cov["rank_value_missing_outside_top_n"] = int(mem.loc[mem["rank"] > n, "rank_value"].isna().sum())
    meta = full.meta[full.meta["entity"].isin(ents)].reset_index(drop=True)
    notes = _summary_notes(n, membership, cov) + list(full.notes)
    return PITUniverse(
        market=Market.US,
        n=n,
        membership=membership,
        prices=prices,
        benchmarks=dict(full.benchmarks),
        entity_benchmark={e: BENCHMARK for e in prices},
        meta=meta,
        coverage=cov,
        notes=notes,
    )


def _month_coverage(ms: pd.DataFrame) -> dict[str, Any]:
    by_year = {}
    for y, g in ms.groupby(ms["date"].dt.year):
        c = Counter(g["src"])
        tot = len(g)
        by_year[int(y)] = {
            "member_months": tot,
            "yahoo": c.get("yahoo", 0),
            "ftd": c.get("ftd", 0),
            "missing": c.get("missing", 0),
            "yahoo_share": round(c.get("yahoo", 0) / tot, 4) if tot else None,
            "covered_share": round((c.get("yahoo", 0) + c.get("ftd", 0)) / tot, 4) if tot else None,
        }
    c = Counter(ms["src"])
    return {
        "member_months": len(ms),
        "member_months_with_prices": c.get("yahoo", 0),
        "member_months_ftd_only": c.get("ftd", 0),
        "member_months_missing": c.get("missing", 0),
        "by_year": by_year,
        "missing_examples": sorted(set(ms.loc[ms["src"] == "missing", "entity"]))[:40],
    }


def _summary_notes(n: int, membership: pd.DataFrame, cov: dict) -> list[str]:
    tot = max(cov["member_months"], 1)
    y, f, m = cov["member_months_with_prices"], cov["member_months_ftd_only"], cov["member_months_missing"]
    dates = membership["date"]
    head = (
        f"미국 PIT 유니버스: S&P 500 시점별 구성(fja05680 + Wikipedia 보정, CIK 단위) "
        f"{dates.min().date() if len(dates) else '-'}~{dates.max().date() if len(dates) else '-'} 월말 "
        f"{dates.nunique()}회, " + ("전 구성종목" if n >= 500 else f"시가총액 상위 {n}개")
    )
    lines = [
        head,
        f"회원-월 {cov['member_months']}개: 일별 가격 {y}({_pct(y / tot)}), FTD 월말가만 {f}({_pct(f / tot)}), "
        f"결측 {m}({_pct(m / tot)})",
    ]
    if cov.get("rejected_series") or cov.get("masked_rows"):
        msg = f"SEC FTD 가격 대조로 다른 발행사 Yahoo 시계열 {cov.get('rejected_series', 0)}건 거부"
        if cov.get("masked_rows"):
            msg += f", 부분 불일치 {len(cov.get('masked_entities', []))}개 종목 {cov['masked_rows']}행 제거"
        lines.append(msg)
    if cov.get("no_price_entities"):
        lines.append(
            f"일별 가격 없는 종목 {len(cov['no_price_entities'])}개(인수·상폐 후 Yahoo 삭제) — FTD 월말가/종가 표시만, "
            "RSMT-Q 학습에서는 제외"
        )
    if cov.get("unmapped_spells"):
        lines.append(
            f"CIK 미확정 구간 {len(cov['unmapped_spells'])}개 제외(회원-월 {cov.get('unmapped_member_months', 0)}개): "
            f"{cov['unmapped_spells'][:4]}"
        )
    tm = cov.get("terminal_marks") or {}
    if tm:
        lines.append(
            "상폐 종가 출처: "
            + ", ".join(
                f"{k} {v}건"
                for k, v in sorted(tm.items())
                if k in ("yahoo_last_bar", "ftd", "ftd_otc", "failed_assumed_zero")
            )
        )
    lines.append(
        "잔여 편향: FTD 월말가는 배당 미포함·최대 7일 지연, S&P 위원회 편입 기준(순수 시총 규칙 아님), "
        "fja 2019년 이전 티커 역보정 오류 일부 수작업 보정"
    )
    return lines


# ============================================================================ fundamentals


def _membership_entities(settings: Settings, net: PitNet) -> list[str]:
    """Every CIK of an S&P 500 spell overlapping 2014-01-01..today (no universe build cached yet)."""
    today = _today()
    b = _Builder(
        settings, DiskCache(settings.cache_path, enabled=False), net, pd.Timestamp("2014-01-01"), None, today, None
    )
    b.dates = month_end_sessions(nyse_sessions(b.start, today))
    b.dates = b.dates[b.dates <= today]
    b.W0, b.W1 = b.dates[0], b.dates[-1]
    b._membership()
    return sorted({us_cik.cik_entity(c) for c in b.win["cik"].dropna()})


def _companyfacts(net: PitNet, cik: int) -> dict:
    """Trimmed companyfacts (cached gzip). Refreshed weekly for filers active within 400 days, else kept; and
    refreshed after 12 hours when the cached submissions list (refreshed daily by the dataset build) shows a newer
    10-K / 10-Q than the copy holds (docs/MODELING.md 84.5: a copy up to a week old could miss the last filing
    before a vintage cutoff, which would then be frozen as missing)."""
    rel = f"facts/{cik}.json.gz"
    p = net.path("xbrl", rel)
    hit = net.read_json("xbrl", rel)
    if hit is not None:
        age_h = (time.time() - p.stat().st_mtime) / 3600
        last = _last_filed(hit)
        active = last is not None and (pd.Timestamp.today() - pd.Timestamp(last)).days < 400
        newest = us_cik.newest_periodic(net.read_json("sec", f"subs/{int(cik)}.json.gz"))
        behind = bool(hit) and newest is not None and newest > (_last_periodic_filed(hit) or "") and age_h >= 12
        if not behind and (not hit or not active or age_h < 24 * 7):
            if hit or age_h < 24 * 30:
                return hit
    try:
        r = net.http_get(COMPANYFACTS.format(cik=int(cik)), "sec", ok=(200, 404))
    except Exception as exc:  # noqa: BLE001 - keep the stale copy, reported
        if hit is not None:
            net.note(f"companyfacts CIK{cik} 갱신 실패({type(exc).__name__}) — 캐시 사용")
            return hit
        raise
    j = us_fund.trim_companyfacts(r.json()) if r.status_code == 200 else {}
    net.write_json("xbrl", rel, j)
    return j


def _last_filed(j: dict) -> str | None:
    best = None
    for ns in ("us-gaap", "dei"):
        for c in ((j.get("facts") or {}).get(ns) or {}).values():
            for arr in (c.get("units") or {}).values():
                for f in arr:
                    if best is None or f["filed"] > best:
                        best = f["filed"]
    return best


def _last_periodic_filed(j: dict) -> str | None:
    """Latest ``filed`` of a 10-K / 10-Q (incl. amendments, transition reports) fact in a companyfacts JSON."""
    best = None
    for ns in ("us-gaap", "dei"):
        for c in ((j.get("facts") or {}).get(ns) or {}).values():
            for arr in (c.get("units") or {}).values():
                for f in arr:
                    if f.get("form") in us_fund.PERIODIC_FORMS and (best is None or f["filed"] > best):
                        best = f["filed"]
    return best


def _strict_name(a: str, b: str) -> bool:
    """Squashed normalised names contain one another or share a >= 5-letter prefix ('exxonmobil' ~ 'exxon mobil',
    'apache' ~ 'apa'); a shared common word is not enough ('Michael Baker' is not 'Baker Hughes')."""
    sa, sb = us_cik.norm_name(a).replace(" ", ""), us_cik.norm_name(b).replace(" ", "")
    if not sa or not sb:
        return False
    return sa in sb or sb in sa or (len(sa) >= 5 and len(sb) >= 5 and sa[:5] == sb[:5])


def _chains(
    net: PitNet,
    entities: list[str],
    first_filed: dict[int, str | None],
    universe_ciks: set[int],
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """entity -> [predecessor CIKs..., CIK].

    Manual table first (verified holdco reorganisations). Otherwise a predecessor is inferred from a same-ticker CIK
    hand-over in the Wayback SEC ticker files, but only for a NEW registrant (its first periodic filing within
    150 days of taking the ticker) whose predecessor is not an S&P entity of its own and has a strictly matching name
    — an acquirer that takes the target's ticker (Eldorado->CZR, II-VI->COHR, Healthpeak->DOC, Tyco->JCI) is not a
    successor."""
    chains: dict[str, list[int]] = {}
    how: dict[str, str] = {}
    pairs = None
    try:
        pairs = us_cik.sec_ticker_history(net, _today())
    except Exception as exc:  # noqa: BLE001 - automatic chains are optional; reported
        net.note(f"CIK 승계 자동 탐지 생략(Wayback SEC 티커 파일 실패: {type(exc).__name__})")
    for e in entities:
        cik = us_cik.entity_cik(e)
        if cik in MANUAL_PREDECESSORS:
            chains[e] = [*MANUAL_PREDECESSORS[cik], cik]
            how[e] = "manual"
            continue
        chains[e] = [cik]
        ff = first_filed.get(cik)
        if pairs is None or not ff:
            continue
        mine = pairs[pairs["cik"] == cik]
        preds = []
        for _, r in mine.iterrows():
            if abs((pd.Timestamp(ff) - r["first_seen"]).days) > 150:
                continue  # an established filer that took the ticker (acquirer), not a new holdco
            other = pairs[(pairs["ticker"] == r["ticker"]) & (pairs["cik"] != cik)]
            for _, o in other.iterrows():
                gap = (r["first_seen"] - o["last_seen"]).days
                if (
                    int(o["cik"]) not in universe_ciks
                    and o["first_seen"] < r["first_seen"]
                    and -60 <= gap <= 180
                    and _strict_name(o["title"], r["title"])
                ):
                    preds.append((o["first_seen"], int(o["cik"])))
        if preds:
            chains[e] = [c for _, c in sorted(set(preds))] + [cik]
            how[e] = "wayback_handover"
    return chains, how


def _panel_sig(entity: str, chain: list[int], cutoffs: dict, splits: pd.Series | None, cfrom, since, age) -> str:
    """Everything that shapes an old snapshot row except the facts themselves (those are fingerprinted per day)."""
    blob = json.dumps(
        [
            FUND_VERSION,
            entity,
            chain,
            {str(k): v for k, v in cutoffs.items()},
            [] if splits is None else [(str(pd.Timestamp(k).date()), float(v)) for k, v in splits.items()],
            str(cfrom),
            since,
            age,
        ],
        sort_keys=True,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _panel_worker(args: tuple) -> tuple[str, pd.DataFrame, dict]:
    entity, facts, cutoffs, splits, cfrom, since, age, events_from = args
    df, stats = us_fund.entity_panel(entity, facts, cutoffs, splits, cfrom, since, age, events_from)
    return entity, df, stats


def _run_jobs(jobs: list[tuple], workers: int | None, on_result: Any) -> None:
    """Compute snapshot panels, handing each result to ``on_result(entity, df, stats)`` as soon as it arrives (so an
    interrupted cold build resumes from the cache). A process pool for cold builds (the shared quarterly derivation
    is ~1-2 s per company), in-process otherwise. Scripts calling the loaders need ``if __name__ == "__main__":``
    (spawn start method)."""
    done: set[str] = set()
    n_workers = workers if workers is not None else min(8, max(1, (os.cpu_count() or 2) - 1))
    if len(jobs) >= 8 and n_workers > 1 and multiprocessing.parent_process() is None:
        pool = ProcessPoolExecutor(max_workers=n_workers)
        try:
            futures = [pool.submit(_panel_worker, j) for j in jobs]
            for fut in as_completed(futures):
                ent, df, stats = fut.result()
                on_result(ent, df, stats)
                done.add(ent)
        except Exception as exc:  # noqa: BLE001 - e.g. no spawn allowed here: the rest in-process
            log.warning("pit-us fundamentals process pool failed (%s) - running in-process", exc)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    for job in jobs:
        if job[0] not in done:
            on_result(*_panel_worker(job))


def _fundamentals(
    settings: Settings,
    net: PitNet,
    entities: list[str],
    yahoo_splits: dict[str, pd.Series],
    chart_from: dict[str, pd.Timestamp],
    delist: dict[str, pd.Timestamp],
    since: str,
    workers: int | None,
) -> PITFundamentals:
    t0 = time.time()
    age = settings.pit.fundamentals.max_staleness_days
    facts: dict[int, dict] = {}
    failed: dict[int, str] = {}

    def fetch(c: int) -> tuple[int, dict | None, str | None]:
        try:
            return c, _companyfacts(net, c), None
        except Exception as exc:  # noqa: BLE001 - counted per CIK
            return c, None, f"{type(exc).__name__}: {str(exc)[:80]}"

    def fetch_all(ciks: list[int]) -> None:
        with ThreadPoolExecutor(max_workers=3) as pool:  # SEC pacing is enforced inside PitNet (<= 5 req/s)
            for c, j, err in pool.map(fetch, [c for c in ciks if c not in facts and c not in failed]):
                if j is None:
                    failed[c] = err or "?"
                else:
                    facts[c] = j

    own = sorted({us_cik.entity_cik(e) for e in entities})
    fetch_all(own)
    known = net.read_json("build", "known_entities.json") or {}
    universe_ciks = set(own) | {us_cik.entity_cik(e) for e in known.get("entities") or []}
    first = {c: us_fund.first_periodic_filed(facts.get(c) or {}) for c in own}
    chains, chain_how = _chains(net, entities, first, universe_ciks)
    fetch_all(sorted({c for ch in chains.values() for c in ch}))
    t_fetch = time.time() - t0
    jobs, cached, meta = [], {}, {}
    for e in entities:
        chain = [c for c in chains[e] if c in facts and facts[c]]
        if not chain:
            continue
        fb = {c: facts[c] for c in chain}
        cutoffs: dict[int, str] = {}
        for a, b in zip(chains[e][:-1], chains[e][1:], strict=False):
            first_b = us_fund.first_periodic_filed(facts.get(b) or {})
            if first_b:
                cut = pd.Timestamp(first_b) + pd.Timedelta(us_fund.PREDECESSOR_OVERLAP_DAYS, "D")
                cutoffs[a] = cut.strftime("%Y-%m-%d")
        spl, cfrom = yahoo_splits.get(e), chart_from.get(e)
        sig = _panel_sig(e, chain, cutoffs, spl, cfrom, since, age)
        fp = us_fund.event_fingerprint(fb, cutoffs)
        meta[e] = (sig, fp, fb, cutoffs)
        hit = net.read_pickle("xbrl", f"panels/{e}.pkl")
        events_from = None
        if isinstance(hit, dict) and hit.get("sig") == sig:
            old_fp = hit.get("fp") or {}
            changed = sorted(d for d in set(fp) | set(old_fp) if fp.get(d) != old_fp.get(d))
            if not changed:
                cached[e] = (hit["df"], hit["stats"])
                continue
            events_from = changed[0]  # snapshots before the first changed filing day are still exact
            keep = hit["df"][hit["df"]["filed"] < pd.Timestamp(events_from)]
            cached[e] = (keep, {**hit["stats"], "incremental_from": events_from})
        jobs.append((e, fb, cutoffs, spl, cfrom, since, age, events_from))
    results: dict[str, tuple[pd.DataFrame, dict]] = dict(cached)

    def on_result(e: str, new: pd.DataFrame, new_stats: dict) -> None:
        old, old_stats = cached.get(e, (pd.DataFrame(), {}))
        parts = [x for x in (old, new) if len(x)]
        df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else (parts[0] if parts else new)
        stats = {**old_stats, **new_stats}
        if len(df):  # restated_later looks forward: refresh it whenever new filings arrived
            sig, fp, fb, cutoffs = meta[e]
            df = us_fund.drop_unchanged(df.sort_values(["filed"], kind="stable"))
            df["restated_later"] = us_fund.restate_flags(df, fb, cutoffs).to_numpy()
            net.write_pickle("xbrl", f"panels/{e}.pkl", {"sig": sig, "fp": fp, "df": df, "stats": stats})
        results[e] = (df, stats)

    _run_jobs(jobs, workers, on_result)
    frames = []
    dropped_after_delist = 0
    for e in entities:
        if e not in results:
            continue
        df = results[e][0]
        if e in delist and len(df):
            cut = delist[e] + pd.Timedelta(DELIST_GRACE_DAYS, "D")
            dropped_after_delist += int((df["filed"] > cut).sum())
            df = df[df["filed"] <= cut]
        frames.append(df)
    cols = FUNDAMENTAL_COLUMNS + us_fund.PANEL_EXTRA
    frames = [f[cols] for f in frames if len(f)]
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    panel = panel.sort_values(["entity", "available_from", "period_end"], ignore_index=True)
    with_rows = set(panel["entity"])
    no_xbrl = sorted(e for e in entities if e not in with_rows)
    by_year = {}
    if len(panel):
        for y, g in panel.groupby(panel["available_from"].dt.year):
            by_year[int(y)] = {
                "rows": len(g),
                "entities": int(g["entity"].nunique()),
                "shares_out": round(float(g["shares_out"].notna().mean()), 4),
                "revenue_ttm": round(float(g["revenue_ttm"].notna().mean()), 4),
                "ni_parent_ttm": round(float(g["ni_parent_ttm"].notna().mean()), 4),
                "equity_parent": round(float(g["equity_parent"].notna().mean()), 4),
                "op_income_ttm": round(float(g["op_income_ttm"].notna().mean()), 4),
            }
    cov = {
        "entities": len(entities),
        "entities_with_rows": len(with_rows),
        "entities_without_xbrl": no_xbrl,
        "companyfacts_failed": {str(k): v for k, v in failed.items()},
        "rows": len(panel),
        "restated_later_rows": int(panel["restated_later"].sum()) if len(panel) else 0,
        "rows_dropped_after_delisting": dropped_after_delist,
        "cik_chains": {e: ch for e, ch in chains.items() if len(ch) > 1},
        "chain_method": chain_how,
        "by_year": by_year,
        "max_staleness_days": age,
        "timing_sec": {
            "fetch": round(t_fetch, 1),
            "total": round(time.time() - t0, 1),
            "computed": len(jobs),
            "incremental": sum(1 for j in jobs if j[-1]),
        },
    }
    notes = [
        f"미국 PIT 재무: SEC XBRL companyfacts as-of 스냅샷 {len(panel)}행, {len(with_rows)}/{len(entities)}개 종목 "
        "(제출일 다음 거래일부터 사용, 이후 정정은 원래 값 유지·restated_later 표시)",
    ]
    if no_xbrl:
        notes.append(f"XBRL 재무 없음 {len(no_xbrl)}개 종목(FDIC 제출 은행·XBRL 이전 상폐 등): {no_xbrl[:6]}")
    if failed:
        notes.append(f"companyfacts 수집 실패 CIK {len(failed)}개: {list(failed)[:6]}")
    if dropped_after_delist:
        notes.append(f"상장폐지 {DELIST_GRACE_DAYS}일 이후 제출분 {dropped_after_delist}행 제외(인수자 명의 계속 제출)")
    multi = [e for e in with_rows if panel.loc[panel["entity"] == e, "shares_out"].isna().all()]
    if multi:
        notes.append(
            f"주식수 검증 불가(다중 클래스 등) {len(multi)}개 종목은 shares_out=NaN (잘못된 값 대신 결측): {multi[:6]}"
        )
    return PITFundamentals(market=Market.US, panel=panel, coverage=cov, notes=notes + list(net.notes))
