"""LLM filings pillar backfill on post-training-cutoff earnings releases (docs/MODELING.md 46, pre-declared 2026-10-12).

The report's LLM (claude-opus-5) has a training-data cutoff of May 2026 (Anthropic model docs), so only S&P 500 8-K
Item 2.02 earnings releases filed 2026-06-01 .. 2026-09-10 are scored - earlier releases would let the model use what
it already knows about the later returns. The report's own filings analysis (``quantcio.llm.filings``, same model,
effort and excerpt rules) is run on each release with no fundamentals (no point-in-time snapshot). COSTS MONEY.

    .venv/bin/python scripts/llm_backfill_study.py score --limit 10    # pilot: measures the cost per release
    .venv/bin/python scripts/llm_backfill_study.py score               # all (after the user approved the budget)
    .venv/bin/python scripts/llm_backfill_study.py eval                # IC vs the next 20 sessions' excess return

Writes .cache/quantcio/alt/llm_backfill/scores.jsonl (resumable) and reports/modeling/US_llm_backfill_summary.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantcio.config import PROJECT_ROOT, load_settings  # noqa: E402
from quantcio.data.cache import DiskCache  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "modeling"
START, END = pd.Timestamp("2026-06-01", tz="UTC"), pd.Timestamp("2026-09-11", tz="UTC")
H = 20


def say(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def store(settings) -> Path:
    p = Path(settings.cache_path) / "alt" / "llm_backfill"
    p.mkdir(parents=True, exist_ok=True)
    return p / "scores.jsonl"


def members(settings, cache) -> pd.DataFrame:
    """S&P 500 PIT members during the window: entity, ticker, name."""
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.US, settings)
    m = load_pit_universe(Market.US, settings, cache, n, start).membership
    m = m[(m["date"] >= "2026-05-29") & (m["date"] <= "2026-09-30")]
    return m.sort_values("date").groupby("entity", as_index=False).agg(ticker=("ticker", "last"), name=("name", "last"))


def releases(settings, cache, mem: pd.DataFrame, prov) -> list[dict]:
    from quantcio.data.pit.us_cik import entity_cik

    out = []
    for e, t, nm in mem[["entity", "ticker", "name"]].itertuples(index=False):
        try:
            _, df = prov._submissions(str(entity_cik(e)))
        except Exception as exc:  # noqa: BLE001 - one company's submissions failing must not stop the run
            say(f"{t}: submissions failed {type(exc).__name__}")
            continue
        k = df[df["form"].astype(str).str.startswith("8-K") & df["items"].astype(str).str.contains("2.02", regex=False)]
        k = k[(k["filed_utc"] >= START) & (k["filed_utc"] < END)]
        for _, row in k.iterrows():
            out.append({"entity": e, "ticker": t, "name": nm, "row": row})
    return out


async def score_all(limit: int) -> None:
    from quantcio.data.us import USProvider
    from quantcio.llm.client import LLMClient, LLMRecorder
    from quantcio.llm.filings import analyze_filings
    from quantcio.models import Instrument, Market

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    prov = USProvider(settings, cache)
    p = store(settings)
    done = {json.loads(x)["acc"] for x in p.read_text().splitlines()} if p.exists() else set()
    rel = releases(settings, cache, members(settings, cache), prov)
    say(f"releases in the window: {len(rel)} ({len(done)} already scored)")
    todo = [r for r in rel if r["row"]["accessionNumber"] not in done]
    if limit:
        todo = todo[:limit]
    client = LLMClient(settings)
    total, n = 0.0, 0
    for r in todo:
        row = r["row"]
        notes: list[str] = []
        try:
            item = prov._earnings_release_item(row, settings.data.max_filing_chars, notes)
        except Exception as exc:  # noqa: BLE001
            say(f"{r['ticker']}: release text failed {type(exc).__name__}")
            continue
        inst = Instrument(
            ticker=r["ticker"],
            market=Market.US,
            name=str(r["name"]),
            currency="USD",
            yf_symbol=r["ticker"],
            benchmark="SPY",
            benchmark_name="S&P 500 (SPY)",
        )
        rec = LLMRecorder(lang="ko")
        fi = await analyze_filings(client, inst, [item], None, settings, recorder=rec)
        from quantcio.llm import llm_pillar

        res = {
            "acc": row["accessionNumber"],
            "entity": r["entity"],
            "ticker": r["ticker"],
            "accepted_utc": str(row.get("accepted_utc") or row["filed_utc"]),
            "filed": str(pd.Timestamp(row["filingDate"]).date()),
            "tone": None if fi is None else fi.tone,
            "guidance": None if fi is None else fi.guidance,
            "n_risks": None if fi is None else len(fi.risk_factors),
            "pillar": None if fi is None else llm_pillar(None, fi),
            "cost_usd": rec.cost_usd,
        }
        with p.open("a") as fh:
            fh.write(json.dumps(res, ensure_ascii=False) + "\n")
        total += rec.cost_usd
        n += 1
        if n % 10 == 0 or limit:
            say(f"{n}/{len(todo)} scored, cost so far ${total:.2f} (${total / n:.3f} per release)")
    say(f"done: {n} scored, ${total:.2f} total, ${total / max(n, 1):.3f} per release; window has {len(rel)} releases")


def evaluate() -> str:
    from scipy.stats import spearmanr

    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    s = pd.DataFrame([json.loads(x) for x in store(settings).read_text().splitlines()])
    s = s.dropna(subset=["tone"])
    n, start = universe_params(Market.US, settings)
    uni = load_pit_universe(Market.US, settings, cache, n, start)
    spy = uni.benchmarks["SPY"]["close"].astype(float).sort_index()
    rows = []
    for r in s.itertuples(index=False):
        px = uni.prices.get(r.entity)
        if px is None or "close" not in px:
            continue
        c = px["close"].astype(float).dropna().sort_index()
        acc = pd.Timestamp(r.accepted_utc)
        acc = acc.tz_localize("UTC") if acc.tzinfo is None else acc
        et = acc.tz_convert("America/New_York")
        day = pd.Timestamp(et.date())
        k = c.index.searchsorted(day, "left")
        if k < len(c) and c.index[k] == day and et.hour >= 16:
            k += 1
        if k < 1 or k + H >= len(c):
            continue
        e0, e1 = c.index[k], c.index[k + H]
        sp0, sp1, spm = (spy.asof(x) for x in (e0, e1, c.index[k - 1]))
        lr = np.log(c.iloc[k - 260 : k + 1]).diff().dropna()
        lm = np.log(spy.reindex(lr.index).ffill()).diff().reindex(lr.index)
        ok = lm.notna()
        beta = float(np.cov(lr[ok], lm[ok])[0, 1] / lm[ok].var()) if ok.sum() > 120 else 1.0
        rows.append(
            {
                "ticker": r.ticker,
                "week": e0 - pd.Timedelta(e0.weekday(), "D"),
                "tone": r.tone,
                "pillar": r.pillar,
                "react": np.log(c.iloc[k] / c.iloc[k - 1]) - beta * np.log(sp0 / spm),
                "post20": np.log(c.iloc[k + H] / c.iloc[k]) - beta * np.log(sp1 / sp0),
            }
        )
    d = pd.DataFrame(rows)
    lines = ["# LLM 공시 필러 소급 검증 — 학습 종료(2026-05) 이후 실적 발표 (docs/MODELING.md 46)", ""]
    lines.append(f"- 점수 낸 발표 {len(s)}건 · 수익률 계산된 발표 {len(d)}건 · 비용 합계 ${s['cost_usd'].sum():.2f}")
    for score in ("tone", "pillar"):
        for lab in ("post20", "react"):
            pooled = spearmanr(d[score], d[lab]).correlation if len(d) > 10 else float("nan")
            wk = [
                spearmanr(g[score], g[lab]).correlation
                for _, g in d.groupby("week")
                if len(g) >= 10 and g[score].nunique() > 1
            ]
            m = float(np.mean(wk)) if wk else float("nan")
            t = m / (np.std(wk, ddof=1) / np.sqrt(len(wk))) if len(wk) > 2 else float("nan")
            lines.append(f"- {score} vs {lab}: 주별 IC 평균 {m:+.4f} (t {t:+.2f}, 주 {len(wk)}개) · 전체 {pooled:+.4f}")
    lines += [
        "",
        "판정(46): 20거래일 이후 수익과의 주별 IC t ≥ 2 이면 '근거 있음'. 표본이 실적 시즌 하나라 검정력이 낮습니다.",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "US_llm_backfill_summary.md").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--limit", type=int, default=0)
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "score":
        asyncio.run(score_all(a.limit))
    else:
        print(evaluate())


if __name__ == "__main__":
    main()
