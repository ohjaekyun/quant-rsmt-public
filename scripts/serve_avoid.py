"""US "avoid" reference flag for the report (docs/MODELING.md 103.4 / 104.3 rule 5; post hoc, display only).

    .venv/bin/python scripts/serve_avoid.py            # write the flag file from today's US dataset
    .venv/bin/python scripts/serve_avoid.py --remove   # delete it (the report stops showing the flag)

The flag: S&P 500 PIT members whose 12-1 momentum (return of the last 12 months without the latest month) is in the
bottom 20% of the cross-section on the dataset's latest date. The report reads ``<cache>/reference/avoid_US.json``
(``report.render._avoid_view``); nothing in the signal, the sizing or the served models uses it. The evidence lines
come from reports/modeling/US_v103_french.json (Kenneth French library, big caps), US_v103_prior.json (our data) and,
when it exists, US_v104_books.json (our data with the acquired names filled in).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _k in ("KRX_ID", "KRX_PW"):
    os.environ.pop(_k, None)

import pandas as pd  # noqa: E402

from quantcio.config import load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402
from quantcio.modeling.dataset import build_dataset  # noqa: E402
from quantcio.models import Market  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "modeling"
Q = 0.20
SIGNAL = "mom_12_1"


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def flag_path(settings) -> Path:
    return Path(settings.cache_path) / "reference" / "avoid_US.json"


def evidence() -> list[dict]:
    """The bottom-quintile CAPM alphas behind the flag (annual, decimal) with their t statistics."""
    out = []
    fr = json.loads((OUT / "US_v103_french.json").read_text())["signals"]["momentum"]
    for key, ko, en in (
        ("big_long", "프렌치 대형주 {a}~{b}", "French big caps {a}-{b}"),
        ("big_recent", "프렌치 대형주 {a}~{b}", "French big caps {a}-{b}"),
    ):
        a, b = (x[:4] for x in fr[f"{key}_span"])
        e = fr[f"{key}_bottom"]
        out.append({"ko": ko.format(a=a, b=b), "en": en.format(a=a, b=b), "alpha": e["alpha_ann"] / 100, "t": e["t"]})
    pr = json.loads((OUT / "US_v103_prior.json").read_text())
    e = pr["books"]["mom_12_1/Q1"]["net"]
    a, b = pr["span"][0][:4], pr["span"][1][:4]
    out.append({"ko": f"우리 자료 {a}~{b}", "en": f"our data {a}-{b}", "alpha": e["alpha_ann"], "t": e["t"]})
    p4 = OUT / "US_v104_books.json"
    if p4.exists():
        e = (json.loads(p4.read_text()).get("avoid") or {}).get("filled")
        if e:
            out.append({"ko": "빠진 종목을 메운 자료", "en": "our data, filled", "alpha": e["alpha_ann"], "t": e["t"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    s = load_settings()
    path = flag_path(s)
    if a.remove:
        path.unlink(missing_ok=True)
        say(f"removed {path}")
        return
    ds = build_dataset(Market.US, s, DiskCache(s.cache_path), use_cache=True, progress=say)
    f = ds.frame
    last = f[f["date"] == f["date"].max()].dropna(subset=[SIGNAL]).copy()
    last["pct"] = last[SIGNAL].rank(method="average", pct=True)
    last["avoid"] = last["pct"] <= Q
    names = {
        str(r.ticker).upper().replace(".", "-"): {
            "name": str(r.name),
            "mom": round(float(getattr(r, SIGNAL)), 4),
            "pct": round(float(r.pct), 4),
            "avoid": bool(r.avoid),
        }
        for r in last.itertuples(index=False)
    }
    low = last[last["avoid"]].sort_values("pct")
    o = {
        "market": "US",
        "as_of": str(pd.Timestamp(last["date"].max()).date()),
        "snapshot": ds.snapshot,
        "signal": SIGNAL,
        "q": Q,
        "n": len(last),
        "n_avoid": int(last["avoid"].sum()),
        "threshold": round(float(low[SIGNAL].max()), 4),
        "tickers": names,
        "evidence": evidence(),
        "doc": "103.4",
        "generated": time.strftime("%Y-%m-%d %H:%M KST"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(o, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [
        f"# 미국 '피할 쪽' 참고 표시 ({o['as_of']}, 사후 관찰 · 신호·비중에 미반영)",
        "",
        f"- S&P 500 {o['n']}종목 중 12-1 모멘텀 하위 20%: {o['n_avoid']}종목 "
        f"(12-1 수익 {100 * o['threshold']:+.1f}% 이하)",
        "- 근거(하위 20% 장부의 시장 위험 감안 초과수익, 연): "
        + " · ".join(f"{e['ko']} {100 * e['alpha']:+.1f}%(t {e['t']:+.1f})" for e in o["evidence"]),
        "",
        "| 종목 | 코드 | 12-1 수익 | 아래에서 위치 |",
        "|---|---|---|---|",
    ]
    for r in low.itertuples(index=False):
        lines.append(f"| {r.name} | {r.ticker} | {100 * getattr(r, SIGNAL):+.1f}% | {100 * r.pct:.1f}% |")
    (OUT / "US_avoid_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"wrote {path} ({o['n_avoid']} of {o['n']} names, as of {o['as_of']}) and {OUT / 'US_avoid_latest.md'}")


if __name__ == "__main__":
    main()
