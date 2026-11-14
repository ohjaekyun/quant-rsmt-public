"""More KRX data: institution-type flows, market-warning designations, single-stock futures
(docs/MODELING.md 66, pre-declared 2026-10-17). Research only.

    .venv/bin/python scripts/v26_krx_data.py fetch --what ssf     # KRX [15001] single-stock futures, weekly (session)
    .venv/bin/python scripts/v26_krx_data.py fetch --what kind    # KIND 투자주의 / 투자경고 / 투자위험, 2013+

KRX: one process, the paced reader of scripts/v19_krx.py (>= 2 s + jitter, stops on an expired session; rerun to
resume). KIND: 3 s between requests. Files under <cache>/alt/keyed/v26/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402

say = v13.say
KIND = "https://kind.krx.co.kr/investwarn/investattentwarnrisky.do"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research bot)"}
KIND_TYPES = {
    "caution": (1, "invstcautnisu_sub", "4"),
    "warn": (2, "invstwarnisu_sub", "3"),
    "risk": (3, "invstriskisu_sub", "3"),
}
KIND_GAP = 3.0
SSF_MULT = 10  # shares per single-stock futures contract


def root(settings) -> Path:
    return kd.root(settings, "v26")


# ================================================================================================ KRX futures


def fetch_ssf(settings) -> dict:
    import v19_krx as vk

    d = root(settings) / "ssf"
    d.mkdir(parents=True, exist_ok=True)
    k = vk.Krx()
    dates = vk._weekly_dates(settings)
    todo = [x for x in dates if not (d / f"{x:%Y%m%d}.parquet").exists()]
    n = 0
    for day in todo:
        r = k.read(
            "dbms/MDC/STAT/standard/MDCSTAT12501", trdDd=f"{day:%Y%m%d}", prodId="KRDRVFUEQU", subProdId="KRDRVFUEQU"
        )
        df = pd.DataFrame(r.get("output", []))
        if df.empty:
            vk._stop_if_dead(k, "주식선물", n)
        df.astype(str).to_parquet(d / f"{day:%Y%m%d}.parquet", index=False)
        n += 1
        if n % 50 == 0:
            say(f"주식선물 {n}/{len(todo)} 주")
    return {"weeks_new": n, "weeks": len(dates)}


# ================================================================================================ KIND warnings


def _cells(html: str) -> list[list[str]]:
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S)
    out = [[re.sub(r"<[^>]+>|\s+", " ", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)] for r in rows]
    return [c for c in out if len(c) >= 4 and c[0].isdigit()]


def fetch_kind(settings) -> dict:
    d = root(settings) / "kind"
    d.mkdir(parents=True, exist_ok=True)
    sess, n_req, n_rows = requests.Session(), 0, 0
    this_year = pd.Timestamp.today().year
    for kind, (menu, fwd, order) in KIND_TYPES.items():
        for y in range(2013, this_year + 1):
            p = d / f"{kind}_{y}.json"
            if p.exists() and y < this_year:
                continue
            rows, page = [], 1
            while True:
                form = {
                    "method": "investattentwarnriskySub", "forward": fwd, "menuIndex": str(menu),
                    "currentPageSize": "100", "pageIndex": str(page), "orderMode": order, "orderStat": "D",
                    "marketType": "", "searchCodeType": "", "searchCorpName": "", "repIsuSrtCd": "",
                    "startDate": f"{y}-01-01", "endDate": f"{y}-12-31",
                }  # fmt: skip
                for attempt in range(4):
                    try:
                        r = sess.post(KIND, data=form, headers=UA, timeout=60)
                        r.raise_for_status()
                        break
                    except requests.RequestException:
                        time.sleep(15 * (attempt + 1))
                else:
                    raise RuntimeError(f"KIND {kind} {y} p{page}: 4번 실패")
                n_req += 1
                time.sleep(KIND_GAP)
                got = _cells(r.text)
                rows += got
                total = int(got[0][0]) if page == 1 and got else 0
                if page == 1:
                    pages = max(1, int(np.ceil(total / 100)))
                if not got or page >= pages:
                    break
                page += 1
            p.write_text(json.dumps(rows, ensure_ascii=False))
            n_rows += len(rows)
        say(f"KIND {kind}: done ({n_req} requests so far)")
    return {"requests": n_req, "rows": n_rows}


def main() -> None:
    from quantcio.config import load_settings

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", required=True, choices=["ssf", "kind"])
    a = ap.parse_args()
    s = load_settings()
    fn = {"ssf": lambda: fetch_ssf(s), "kind": lambda: fetch_kind(s)}[a.what]
    try:
        say(f"{a.what}: {fn()}")
    except Exception as exc:  # noqa: BLE001 - resumable: report and stop
        say(f"{a.what}: FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
