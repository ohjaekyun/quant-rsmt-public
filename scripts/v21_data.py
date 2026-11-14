"""Blocks I (US insider detail), U (US federal awards, monthly), Z (KR lending rates x interest coverage) and F2 (KR
business groups) plus the PCA models - docs/MODELING.md 55, pre-declared 2026-10-13. Research only.

    .venv/bin/python scripts/v21_data.py fetch --what all     # or form345_owners, usa_monthly, dart_j, ecos_loans

Sources: SEC Form 3/4/5 data sets (REPORTINGOWNER: relationship and officer title; the transactions themselves come
from the cached quarterly parquet of data/alt/us_sec.py), USAspending spending_over_time by month for the recipients
matched in scripts/v16_data.py, OpenDART lists of type J (Fair Trade Commission disclosures) and the cached B/E/I lists,
ECOS 121Y006 (new-loan rates of deposit banks) and the DART bulk statements of scripts/v16_data.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v19_data as v19  # noqa: E402

say = v13.say
ECOS_LOANS = {"corp": "BECBLA02", "large": "BECBLA0201", "sme": "BECBLA0202", "hh": "BECBLA03"}
ECOS_LAG_D = 35
USA_LAG_D = 90
C_LEVEL = re.compile(
    r"\bCEO\b|CHIEF EXECUTIVE|\bCFO\b|CHIEF FINANCIAL|\bCOO\b|CHIEF OPERATING|PRESIDENT|CHAIRMAN|CHAIR\b"
)
REGIME = {"r_kr_loan_corp_chg", "r_kr_loan_sme_spread", "r_kr_loan_hh_chg"}


def root(settings) -> Path:
    return kd.root(settings, "v21")


# ================================================================================================ fetch


def fetch_form345_owners(settings) -> dict:
    """REPORTINGOWNER of every quarterly Form 3/4/5 data set -> accession, owner, relationship flags, title."""
    from quantcio.data.alt import us_sec

    d = root(settings) / "form345_owners"
    d.mkdir(exist_ok=True)
    hdr = v13._sec_headers()
    html = requests.get(us_sec.FORM345_INDEX, headers=hdr, timeout=120).text
    links = us_sec.parse_form345_index(html)
    done = 0
    for key, url in links.items():
        if key < "2010q1":
            continue
        out = d / f"{key}.parquet"
        if out.exists() and key < pd.Timestamp.today().to_period("Q").strftime("%Yq%q").lower():
            continue
        zp = d / f"{key}.zip"
        v13._download(url, zp, hdr)
        time.sleep(0.5)
        with zipfile.ZipFile(zp) as z:
            names = {Path(n).name.upper(): n for n in z.namelist()}
            o = pd.read_csv(z.open(names["REPORTINGOWNER.TSV"]), sep="\t", dtype=str)
        zp.unlink(missing_ok=True)
        rel = o.get("RPTOWNER_RELATIONSHIP", pd.Series("", index=o.index)).fillna("").str.upper()
        pd.DataFrame(
            {
                "accession": o["ACCESSION_NUMBER"],
                "owner_cik": pd.to_numeric(o["RPTOWNERCIK"], errors="coerce"),
                "director": rel.str.contains("DIRECTOR"),
                "officer": rel.str.contains("OFFICER"),
                "ten": rel.str.contains("TENPERCENT"),
                "title": o.get("RPTOWNER_TITLE", pd.Series("", index=o.index)).fillna("").str.upper(),
            }
        ).to_parquet(out, index=False)
        done += 1
        if done % 10 == 0:
            say(f"Form 345 owners {done} quarters")
    return {"quarters_new": done, "quarters": len(list(d.glob("*.parquet")))}


def fetch_usa_monthly(settings) -> dict:
    """Monthly obligations (contracts + IDVs, assistance) of the recipients matched in v16 (USAspending, free API)."""
    p = v16.root(settings) / "usaspending.json"
    if not p.exists():
        return {"skipped": "v16 usaspending.json 없음"}
    have_p = root(settings) / "usa_monthly.json"
    have = json.loads(have_p.read_text()) if have_p.exists() else {}
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    calls = 0
    for e, v in sorted(json.loads(p.read_text()).items()):
        rids = list((v.get("recipients") or {}).keys())
        if not rids or e in have:
            continue
        months: dict[str, dict[str, float]] = {}
        for rid in rids:
            body = {
                "group": "month",
                "filters": {"recipient_id": rid, "time_period": [{"start_date": "2007-10-01", "end_date": today}]},
            }
            js = v16._usa_post("/search/spending_over_time/", body)
            calls += 1
            time.sleep(0.5)
            for q in js.get("results") or []:
                tp = q.get("time_period") or {}
                fy, mo = int(tp.get("fiscal_year") or 0), int(tp.get("month") or 0)
                if not fy or not mo:
                    continue
                cal = pd.Timestamp(fy - 1, 10, 1) + pd.DateOffset(months=mo - 1)  # fiscal month 1 = October
                acc = months.setdefault(f"{cal:%Y-%m}", {"contract": 0.0, "assist": 0.0})
                acc["contract"] += float(q.get("Contract_Obligations") or 0) + float(q.get("Idv_Obligations") or 0)
                acc["assist"] += sum(
                    float(q.get(f) or 0)
                    for f in ("Grant_Obligations", "Loan_Obligations", "Direct_Obligations", "Other_Obligations")
                )
        have[e] = months
        if len(have) % 25 == 0:
            have_p.write_text(json.dumps(have))
            say(f"USAspending monthly {len(have)} entities ({calls} calls)")
    have_p.write_text(json.dumps(have))
    return {"entities": len(have), "calls": calls}


def fetch_dart_j(settings, cache) -> dict:
    """OpenDART list.json type J (Fair Trade Commission disclosures) of every universe company, 2013+."""
    k = kd.key("DART_API_KEY")
    cc = v16.kr_corp_codes(settings)
    codes = sorted(v16.universe("KR", settings, cache)["entity"].astype(str))
    d = root(settings) / "dart_j"
    d.mkdir(exist_ok=True)
    sess, calls = requests.Session(), 0
    today = pd.Timestamp.today().strftime("%Y%m%d")
    for i, code in enumerate(codes, 1):
        corp = cc.get(code)
        p = d / f"{corp}.json"
        if not corp or p.exists():
            continue
        rows, page = [], 1
        while True:
            params = {"corp_code": corp, "bgn_de": "20130101", "end_de": today, "pblntf_ty": "J"}
            js = v16._dart_get(sess, k, "list", {**params, "page_no": page, "page_count": 100})
            calls += 1
            rows += [{c: x.get(c) for c in ("rcept_no", "rcept_dt", "report_nm")} for x in js.get("list") or []]
            if js.get("status") != "000" or page >= int(js.get("total_page") or 1):
                break
            page += 1
            time.sleep(0.1)
        p.write_text(json.dumps({"fetched": today, "rows": rows}, ensure_ascii=False))
        time.sleep(0.1)
        if i % 100 == 0:
            say(f"DART J {i}/{len(codes)} ({calls} calls)")
    return {"codes": len(codes), "calls": calls}


def fetch_ecos_loans(settings) -> dict:
    k = kd.key("ECOS_API_KEY")
    end = pd.Timestamp.today().strftime("%Y%m")
    out = {}
    for name, item in ECOS_LOANS.items():
        url = f"{kd.ECOS}/StatisticSearch/{k}/json/kr/1/2000/121Y006/M/199601/{end}/{item}"
        js = requests.get(url, timeout=60).json()
        rows = (js.get("StatisticSearch") or {}).get("row") or []
        if not rows:
            raise RuntimeError(kd._redact(f"ECOS 121Y006 {item}: {str(js)[:200]}", k))
        out[name] = pd.Series(
            {pd.Period(r["TIME"], "M").to_timestamp("M"): float(r["DATA_VALUE"]) for r in rows}
        ).sort_index()
        time.sleep(0.5)
    pd.DataFrame(out).rename_axis("month").reset_index().to_parquet(root(settings) / "ecos_loans.parquet", index=False)
    return {k2: len(v) for k2, v in out.items()}


# ================================================================================================ PCA models


class PcaLinear:
    """PCA (k components of the NaN->0 feature matrix) followed by a linear model; ``coef_`` maps back to the
    original features so ``Fitted.importance`` works. ``kind`` 'logit' (decision_function) or 'ridge' (predict)."""

    def __init__(self, k: int, reg: float, kind: str, seed: int = 0):
        self.k, self.reg, self.kind, self.seed = int(k), float(reg), kind, int(seed)

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> PcaLinear:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression, Ridge

        self.pca = PCA(n_components=min(self.k, x.shape[1]), svd_solver="randomized", random_state=self.seed).fit(x)
        z = self.pca.transform(x)
        if self.kind == "logit":
            self.lm = LogisticRegression(C=self.reg, max_iter=1000).fit(z, y.astype(int), sample_weight=w)
        else:
            self.lm = Ridge(alpha=self.reg * len(z)).fit(z, y, sample_weight=w)
        self.coef_ = self.pca.components_.T @ np.ravel(self.lm.coef_)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return self.lm.decision_function(self.pca.transform(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.lm.predict(self.pca.transform(x))


# ================================================================================================ blocks

DOCS = {
    "in_exec_buy_180": ("최고경영진(CEO·CFO·COO·사장·회장) 장내 매수 수(180일)", "C-level open-market buys, 180d"),
    "in_exec_buy_val_180": (
        "최고경영진 장내 매수 금액 ÷ 시가총액(180일)",
        "C-level open-market buy value / mcap, 180d",
    ),
    "in_dir_buy_180": ("이사 장내 매수 수(180일)", "director open-market buys, 180d"),
    "in_exec_sell_180": ("최고경영진 계획 외 장내 매도 수(180일)", "C-level unplanned open-market sales, 180d"),
    "in_cluster_180": ("클러스터 매수(30일 안 3명 이상) 사건 수(180일)", "cluster buys (3+ insiders in 30d), 180d"),
    "u_award_3m": ("연방 의무액 최근 3개월 ÷ 직전 12개월 3개월 평균(log)", "federal obligations 3m vs prior 12m"),
    "u_award_6m": ("연방 의무액 최근 6개월 ÷ 그 전 6개월(log)", "federal obligations 6m vs prior 6m"),
    "u_award_spike": ("최근 달 연방 의무액 급증(3배 이상, 1천만 달러 이상)", "federal obligation spike"),
    "zi_icr": ("이자보상배율 log((세전이익 + 이자) ÷ 이자)", "interest coverage (log)"),
    "zi_zombie": ("한계기업(2년 연속 이자보상배율 < 1)", "zombie firm (ICR < 1 two years)"),
    "zi_leverage": ("부채 ÷ 자산", "liabilities / assets"),
    "zi_rate_exposure": ("부채비율 × 기업대출 금리 12개월 변화", "leverage x corporate loan-rate change"),
    "zi_int_chg": ("이자 지급 증가 ÷ 자산", "interest-paid change / assets"),
    "r_kr_loan_corp_chg": ("기업대출 금리 12개월 변화(%p)", "corporate loan rate, 12m change"),
    "r_kr_loan_sme_spread": ("중소기업 − 대기업 대출금리 차 36개월 z", "SME minus large-firm loan rate, z"),
    "r_kr_loan_hh_chg": ("가계대출 금리 12개월 변화(%p)", "household loan rate, 12m change"),
    "fg_group": ("대규모기업집단 소속(최근 15개월 현황공시)", "member of a large business group"),
    "fg_internal_12m": ("특수관계인 내부거래·계열 상품용역거래 공시(12개월)", "related-party trade filings, 12m"),
    "fg_related_fin_12m": (
        "특수관계인 출자·증여·자금·유가증권 거래 공시(12개월)",
        "related-party financing filings, 12m",
    ),
    "fg_guarantee_12m": ("타인 채무보증·담보제공 결정(12개월)", "debt guarantees / collateral for others, 12m"),
}
BLOCK_COLS = {
    "KR": {
        "Z": ["zi_icr", "zi_zombie", "zi_leverage", "zi_rate_exposure", "zi_int_chg", *sorted(REGIME)],
        "F2": ["fg_group", "fg_internal_12m", "fg_related_fin_12m", "fg_guarantee_12m"],
    },
    "US": {
        "I": ["in_exec_buy_180", "in_exec_buy_val_180", "in_dir_buy_180", "in_exec_sell_180", "in_cluster_180"],
        "U": ["u_award_3m", "u_award_6m", "u_award_spike"],
    },
}
_J_GROUP = re.compile(r"대규모기업집단현황공시")
_J_INTERNAL = re.compile(r"특수관계인과의내부거래|상품ㆍ?용역거래|특수관계인과의(?:상품|용역)")
_J_FIN = re.compile(r"특수관계인(?:에대한|으로부터|에게|과의).*(?:출자|증여|자금|유가증권|가지급|대여)")
_GUARANTEE = re.compile(r"타인에대한채무보증|타인에대한담보제공")


def _cnt(ev, dates, days) -> np.ndarray:
    return v19._cnt(np.asarray(ev, dtype="datetime64[ns]"), dates, days)


def _ecos_regime(settings) -> pd.DataFrame | None:
    p = root(settings) / "ecos_loans.parquet"
    if not p.exists():
        return None
    t = pd.read_parquet(p).set_index("month").sort_index()
    spread = t["sme"] - t["large"]
    reg = pd.DataFrame(
        {
            "r_kr_loan_corp_chg": t["corp"] - t["corp"].shift(12),
            "r_kr_loan_sme_spread": (spread - spread.rolling(36, min_periods=24).mean())
            / spread.rolling(36, min_periods=24).std(),
            "r_kr_loan_hh_chg": t["hh"] - t["hh"].shift(12),
        }
    )
    reg.index = pd.DatetimeIndex(reg.index) + pd.Timedelta(ECOS_LAG_D, "D")
    return reg


def build_kr(frame: pd.DataFrame, settings) -> tuple[pd.DataFrame, dict]:
    cols = [c for b in BLOCK_COLS["KR"].values() for c in b]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    info: dict = {"fin": 0, "zombie_share": None, "dart_j": 0, "group_members": 0}
    reg = _ecos_regime(settings)
    if reg is not None:
        for c in sorted(REGIME):
            out[c] = v13._asof_rows(reg[[c]], dates, 62)[:, 0]
        info["ecos_months"] = int(reg.notna().all(axis=1).sum())
    rate_chg = out["r_kr_loan_corp_chg"].to_numpy(float)
    fin = v16.kr_fin_table(settings)
    cc = v16.kr_corp_codes(settings)
    zcols = ["zi_icr", "zi_zombie", "zi_leverage", "zi_int_chg"]
    for e in np.unique(ents):
        ii = np.where(ents == e)[0]
        d = dates[ii]
        g = fin.get(e)
        if g is not None and len(g):
            info["fin"] += 1
            ip = g["interest_paid"].astype(float) if "interest_paid" in g else pd.Series(np.nan, index=g.index)
            ebit = g["pretax"].astype(float) + ip.fillna(0.0)
            icr = np.where(ip > 0, np.log(np.clip(ebit / ip.where(ip > 0), 1e-3, None)), np.nan)
            icr = np.where((ip > 0) & (ebit <= 0), np.log(1e-3), icr)
            icr = np.where(ip.fillna(0) <= 0, np.log(100.0), icr)  # no interest paid: no debt-service risk
            low = pd.Series(icr < 0, index=g.index) & pd.Series(np.isfinite(icr), index=g.index)
            prev_low = low.shift(1, fill_value=False).astype(bool)
            consecutive = g["pe_m"].diff().dt.days.between(330, 400)
            tab = pd.DataFrame(
                {
                    "avail": g["avail"],
                    "zi_icr": np.clip(icr, np.log(1e-3), np.log(100.0)),
                    "zi_zombie": (low & prev_low & consecutive).astype(float),
                    "zi_leverage": g["liabilities"] / g["assets"].where(g["assets"] > 0),
                    "zi_int_chg": (ip - ip.shift(1)) / g["assets_prev"].where(g["assets_prev"] > 0),
                }
            )
            out.iloc[ii, [out.columns.get_loc(c) for c in zcols]] = v16._asof(tab, zcols, "avail", d, 550)
        corp = cc.get(e)
        if corp:
            p = root(settings) / "dart_j" / f"{corp}.json"
            if p.exists():
                info["dart_j"] += 1
                rows = [r for r in json.loads(p.read_text()).get("rows", []) if not str(r["report_nm"]).startswith("[")]
                t = pd.DataFrame(rows, columns=["rcept_no", "rcept_dt", "report_nm"])
                t["date"] = pd.to_datetime(t["rcept_dt"], format="%Y%m%d", errors="coerce")
                nm = t["report_nm"].astype(str).str.replace(r"\s+", "", regex=True)
                grp = t.loc[nm.str.contains(_J_GROUP), "date"].to_numpy()
                info["group_members"] += bool(len(grp))
                _put(out, ii, "fg_group", (_cnt(grp, d, 450) > 0).astype(float))
                _put(out, ii, "fg_internal_12m", _cnt(t.loc[nm.str.contains(_J_INTERNAL), "date"].to_numpy(), d, 365))
                _put(out, ii, "fg_related_fin_12m", _cnt(t.loc[nm.str.contains(_J_FIN), "date"].to_numpy(), d, 365))
            lt = v19.kr_list_table(settings, corp)
            if lt is not None:
                orig = lt[~lt["corrected"]]
                ev = orig.loc[orig["nm"].str.contains(_GUARANTEE), "date"].to_numpy()
                _put(out, ii, "fg_guarantee_12m", _cnt(ev, d, 365))
    out["zi_rate_exposure"] = out["zi_leverage"].to_numpy(float) * rate_chg
    me = frame["is_me"].to_numpy(bool) if "is_me" in frame else np.ones(len(frame), bool)
    info["zombie_share"] = round(float(np.nanmean(out.loc[me, "zi_zombie"])), 3)
    return out, info


def _put(out: pd.DataFrame, ii: np.ndarray, col: str, vals) -> None:
    out.iloc[ii, out.columns.get_loc(col)] = vals


def _clusters(buys: pd.DataFrame) -> list[np.datetime64]:
    """Dates (filing date of the 3rd distinct buyer) of cluster buys: 3+ distinct insiders within 30 days."""
    if buys.empty:
        return []
    b = buys.sort_values("trans_date")
    td = b["trans_date"].to_numpy(dtype="datetime64[ns]")
    fd = b["filing_date"].to_numpy(dtype="datetime64[ns]")
    own = b["owner"].to_numpy()
    out, last = [], None
    for i in range(len(b)):
        lo = np.searchsorted(td, td[i] - np.timedelta64(30, "D"), "left")
        if len(set(own[lo : i + 1])) >= 3 and (last is None or td[i] - last > np.timedelta64(30, "D")):
            out.append(fd[lo : i + 1].max())
            last = td[i]
    return out


def build_us(frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    from quantcio.data.alt import us_sec

    cols = [c for b in BLOCK_COLS["US"].values() for c in b]
    out = pd.DataFrame(np.nan, index=frame.index, columns=cols)
    ents = frame["entity"].astype(str).to_numpy()
    dates = frame["date"].to_numpy(dtype="datetime64[ns]")
    info: dict = {}
    # ---- I: insider detail
    tr_parts = [pd.read_parquet(p) for p in sorted(us_sec.form345_cache_dir(settings).glob("*.parquet"))]
    ow_parts = [pd.read_parquet(p) for p in sorted((root(settings) / "form345_owners").glob("*.parquet"))]
    if tr_parts and ow_parts:
        tr = pd.concat(tr_parts, ignore_index=True)
        tr = tr[tr["doc_type"].astype(str).str.startswith("4")].copy()
        tr["owner"] = pd.to_numeric(tr["owners"].astype(str).str.split(r"[;,|]").str[0], errors="coerce")
        ow = pd.concat(ow_parts, ignore_index=True).drop_duplicates(["accession", "owner_cik"])
        tr = tr.merge(ow, left_on=["accession", "owner"], right_on=["accession", "owner_cik"], how="left")
        tr["exec"] = tr["officer"].fillna(False).astype(bool) & tr["title"].fillna("").str.contains(C_LEVEL)
        tr["value"] = tr["shares"].astype(float) * tr["price"].astype(float)
        tr["filing_date"] = pd.to_datetime(tr["filing_date"])
        tr["trans_date"] = pd.to_datetime(tr["trans_date"], errors="coerce").fillna(tr["filing_date"])
        info["form4_rows"] = len(tr)
        info["exec_share_of_buys"] = round(float(tr.loc[tr["code"] == "P", "exec"].mean()), 3)
        by_issuer = {int(k): g for k, g in tr.groupby("issuer_cik")}
        chains = v16.us_chains(settings, cache, pd.DataFrame({"entity": np.unique(ents)}))
        mcap = np.exp(frame["log_mcap"].astype(float)).to_numpy()
        for e in np.unique(ents):
            parts = [by_issuer[int(c)] for c in chains.get(e, set()) if int(c) in by_issuer]
            ii = np.where(ents == e)[0]
            d = dates[ii]
            if not parts:
                for c in BLOCK_COLS["US"]["I"]:
                    _put(out, ii, c, 0.0)
                continue
            g = pd.concat(parts, ignore_index=True)
            buy = g[g["code"] == "P"]
            exb = buy[buy["exec"]]
            fd = lambda x: x["filing_date"].to_numpy(dtype="datetime64[ns]")  # noqa: E731
            _put(out, ii, "in_exec_buy_180", _cnt(fd(exb), d, 180))
            _, val = v16._windowed(fd(exb), exb["value"].fillna(0.0).to_numpy(float), d, 180)
            _put(out, ii, "in_exec_buy_val_180", val / np.where(mcap[ii] > 0, mcap[ii], np.nan))
            _put(out, ii, "in_dir_buy_180", _cnt(fd(buy[buy["director"].fillna(False).astype(bool)]), d, 180))
            sell = g[(g["code"] == "S") & g["exec"] & (g["aff10b5one"].fillna(0) != 1)]
            _put(out, ii, "in_exec_sell_180", _cnt(fd(sell), d, 180))
            _put(out, ii, "in_cluster_180", _cnt(np.array(_clusters(buy), dtype="datetime64[ns]"), d, 180))
    # ---- U: federal awards by month
    p = root(settings) / "usa_monthly.json"
    if p.exists():
        months = json.loads(p.read_text())
        n_u = 0
        for e, m in months.items():
            ii = np.where(ents == e)[0]
            if not len(ii) or not m:
                continue
            s = pd.Series({pd.Timestamp(k) + pd.offsets.MonthEnd(0): v["contract"] + v["assist"] for k, v in m.items()})
            s = s.sort_index().asfreq("ME", fill_value=0.0).clip(lower=0.0)  # de-obligations: no negative month
            s3 = s.rolling(3).sum()
            base = s.shift(3).rolling(12, min_periods=12).mean() * 3
            tab = pd.DataFrame(
                {
                    "u_award_3m": np.log((s3 + 1e5) / (base + 1e5)),
                    "u_award_6m": np.log((s.rolling(6).sum() + 1e5) / (s.shift(6).rolling(6).sum() + 1e5)),
                    "u_award_spike": ((s > 3 * s.shift(1).rolling(12, min_periods=12).mean()) & (s > 1e7)).astype(
                        float
                    ),
                }
            )
            tab.index = tab.index + pd.Timedelta(USA_LAG_D, "D")
            out.iloc[ii, [out.columns.get_loc(c) for c in BLOCK_COLS["US"]["U"]]] = v13._asof_rows(tab, dates[ii], 62)
            n_u += 1
        info["usa_entities"] = n_u
    return out, info


def block(market: str, frame: pd.DataFrame, ds, settings, cache) -> tuple[pd.DataFrame, dict]:
    return build_kr(frame, settings) if market == "KR" else build_us(frame, ds, settings, cache)


# ================================================================================================ main


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "form345_owners", "usa_monthly", "dart_j", "ecos_loans"])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    steps = {
        "ecos_loans": lambda: fetch_ecos_loans(s),
        "dart_j": lambda: fetch_dart_j(s, cache),
        "usa_monthly": lambda: fetch_usa_monthly(s),
        "form345_owners": lambda: fetch_form345_owners(s),
    }
    for name, fn in steps.items():
        if a.what in ("all", name):
            try:
                say(f"{name}: {fn()}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                keys = [kd.key(x) for x in ("DART_API_KEY", "ECOS_API_KEY")]
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), *keys)}")


if __name__ == "__main__":
    main()
