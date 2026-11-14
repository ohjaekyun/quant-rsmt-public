"""Corrected end-to-end build (verifier): FTD issuer check inside resolve_yahoo, per-month FTD mask of Yahoo
returns contradicted by the traded-then symbol's FTD prices, and FTD fill of EVERY member-month that lacks a
Yahoo return (not only spells with no Yahoo symbol). Split-like FTD monthly moves are counted and dropped."""
import sys
import time

import numpy as np
import pandas as pd

import us_pit_universe_fixed as U

T = time.time()
W0, W1 = pd.Timestamp("2014-01-01"), pd.Timestamp("2026-09-24")
snaps = U.fja_snapshots(max_age_h=None)
ch = U.wiki_changes(max_age_h=None)
cur = U.wiki_current(max_age_h=None)
sp = U.patched_spells(snaps, ch)
ev = U.removal_events(sp, ch)
pairs = U.sec_ticker_history()
lookup = U.cik_lookup_index()
wd = U.wikidata_ciks(ev.link.dropna().tolist())
res = [U.cik_for_event(r, pairs, lookup, wd) for _, r in ev.iterrows()]
ev["cik"] = [c for c, _ in res]
ev["cik_method"] = [m for _, m in res]
print(f"events {len(ev)} CIK mapped {ev.cik.notna().sum()} ({time.time() - T:.0f}s)")

cur_cik = dict(zip(cur.sym, cur.cik))
cur_by_cik = U.sec_current_tickers_by_cik()
win = sp[(sp.end.isna() | (sp.end > W0)) & (sp.start <= W1)].copy()
evk = ev.set_index(["fja_ticker", "date"])
alias = {(r.fja_ticker, r.date): r.ticker for r in ev.itertuples()}
win["ftd_sym"] = [alias.get((s.ticker, s.end), s.ticker).replace("-", "").replace(".", "") for s in win.itertuples()]
rename_map = {r.fja_ticker.replace(".", "-"): r.rename_to for _, r in ev.dropna(subset=["rename_to"]).iterrows()}
t0 = time.time()
ftd = U.ftd_panel(set(win.ftd_sym))
ftd_by = dict(tuple(ftd.groupby("symbol")))
print(f"FTD panel {len(ftd)} rows, {ftd.symbol.nunique()} of {win.ftd_sym.nunique()} symbols ({time.time() - t0:.0f}s)")

out = []
for _, s in win.iterrows():
    e = evk.loc[(s.ticker, s.end)] if pd.notna(s.end) and (s.ticker, s.end) in evk.index else None
    cik = cur_cik.get(s.ticker.replace(".", "-")) if pd.isna(s.end) else (e.cik if e is not None else None)
    sym, how = U.resolve_yahoo(s, e, cik if pd.notna(cik) else None, cur_by_cik, rename_map, ftd_rows=ftd_by.get(s.ftd_sym))
    out.append((cik, sym, how, e["cat"] if e is not None else "current"))
win[["cik", "ysym", "yhow", "cat"]] = pd.DataFrame(out, index=win.index)
print(f"spells {len(win)}; Yahoo resolved {win.ysym.notna().sum()} (removed {win[win.end.notna()].ysym.notna().sum()}/{win.end.notna().sum()})")
print(pd.crosstab(win["cat"], win.yhow, margins=True).to_string())

syms = sorted(win.ysym.dropna().unique())
charts = {s: U.yahoo_chart(s)["df"] for s in syms}
adj = pd.DataFrame({s: charts[s]["adjclose"] for s in syms})
adj = adj[adj.index <= W1]
mret = adj.resample("BME").last().pct_change(fill_method=None)
months = mret.index[(mret.index >= pd.Timestamp("2014-02-28")) & (mret.index <= pd.Timestamp("2026-08-31"))]

# per-spell months whose Yahoo prices are contradicted by the FTD price of the traded-then symbol
bad = {}
for i, s in win[win.ysym.notna()].iterrows():
    bm = U.ftd_bad_months(charts[s.ysym], ftd_by.get(s.ftd_sym), max(s.start, W0), s.end if pd.notna(s.end) else W1)
    if bm:
        bad[i] = bm
nbm = sum(len(v) for v in bad.values())
print(f"FTD-contradicted Yahoo spell-months: {nbm} in {len(bad)} spells:",
      sorted({win.loc[i, 'ticker'] + '->' + win.loc[i, 'ysym'] + f'({len(v)})' for i, v in bad.items()}))

SPLITLIKE = np.array([1 / 2, 1 / 3, 1 / 4, 2.0, 3.0])   # reported only: 2/3 and 1.5 matches were real crashes/rallies


def ftd_month_ret(f, s, m, prev):
    p0 = f[(f.px_date > prev - pd.Timedelta(7, unit="D")) & (f.px_date <= prev)].price
    if pd.notna(s.end) and s.end <= m:
        t = f[(f.px_date <= s.end) & (f.px_date > prev)].price
    else:
        t = f[(f.px_date > m - pd.Timedelta(7, unit="D")) & (f.px_date <= m)].price
    if len(p0) and len(t):
        r = t.iloc[-1] / p0.iloc[-1] - 1
        return r if -0.6 < r < 1.0 else None
    return None


rows, splitlike = [], []
for m in months:
    prev = mret.index[mret.index.get_loc(m) - 1]
    mem = win[(win.start <= prev) & (win.end.isna() | (win.end > prev))]
    ys, fs, seen = [], [], set()
    for i, s in mem.iterrows():
        r = np.nan
        if isinstance(s.ysym, str) and m not in bad.get(i, ()) and prev not in bad.get(i, ()):
            if s.ysym in seen:
                continue                                   # two share classes resolved to one series
            r = mret.at[m, s.ysym]
        if pd.notna(r):
            seen.add(s.ysym)
            ys.append(r)
            continue
        f = ftd_by.get(s.ftd_sym)
        if f is None:
            continue
        f = f[f.px_date >= s.start - pd.Timedelta(40, unit="D")]
        x = ftd_month_ret(f, s, m, prev)
        if x is None:
            continue
        if np.abs(SPLITLIKE - (1 + x)).min() < 0.01:
            splitlike.append((m.date(), s.ticker, round(x, 4)))
        fs.append(x)
    rows.append(dict(month=m, n=mem.ticker.nunique(), n_y=len(ys), n_f=len(fs), ew_y=np.mean(ys), ew_yf=np.mean(ys + fs)))
bt = pd.DataFrame(rows).set_index("month")
print("split-like FTD monthly moves (kept, check by hand):", len(splitlike), splitlike[:15])
today = [t for t in cur.sym if t in mret.columns]
bt["ew_today"] = mret.loc[bt.index, today].mean(axis=1)
rsp = U.yahoo_chart("RSP")["df"]["adjclose"]
bt["rsp"] = rsp[rsp.index <= W1].resample("BME").last().pct_change(fill_method=None).reindex(bt.index)
ann = lambda s: (1 + s.dropna()).prod() ** (12 / s.dropna().size) - 1  # noqa: E731
te = lambda a: (a - bt.rsp).dropna().std() * np.sqrt(12)  # noqa: E731
cy, cf = bt.n_y / bt.n, (bt.n_y + bt.n_f) / bt.n
print("coverage yahoo %.3f (min %.3f) | yahoo+FTD %.3f (min %.3f)" % (cy.mean(), cy.min(), cf.mean(), cf.min()))
print("by year yahoo+FTD:", cf.groupby(cf.index.year).mean().round(3).to_dict())
print("annualised 2014-02..2026-08: PIT-yahoo %.4f  PIT-yahoo+FTD %.4f  RSP %.4f  today's-members %.4f"
      % (ann(bt.ew_y), ann(bt.ew_yf), ann(bt.rsp), ann(bt.ew_today)))
print("TE vs RSP: PIT-yahoo %.4f  PIT-yahoo+FTD %.4f  today's-members %.4f" % (te(bt.ew_y), te(bt.ew_yf), te(bt.ew_today)))
for a, b in [("2014", "2018"), ("2019", "2022"), ("2023", "2026")]:
    x = bt.loc[a:b]
    print(f"  {a}-{b}: PIT-y {ann(x.ew_y):.4f} PIT-yf {ann(x.ew_yf):.4f} RSP {ann(x.rsp):.4f} today {ann(x.ew_today):.4f}")
win.to_pickle(U.CACHE / "fixed_spells.pkl")
bt.to_pickle(U.CACHE / "fixed_bt.pkl")
print("total secs", round(time.time() - T, 1), file=sys.stdout)
