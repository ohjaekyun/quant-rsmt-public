"""``quantcio alt`` sub-app: alternative data (docs/MODELING.md 16).

- ``fetch --market KR|US``: refresh the alternative data of the modeling universe (+ the KR XSH band names) into the
  caches (OpenDART / NPS / SEC). Weekly cron recommended; OpenDART allows 20,000 calls per day - an interrupted
  run resumes from the cache.
- ``jobs``: append today's job-posting snapshot of the current KR universe to the append-only ledger
  ``alt_ledger/kr_jobs.jsonl`` (forward recording only; needs SARAMIN_API_KEY and/or WORK24_API_KEY). Commit the
  ledger periodically like ``holdout/ledger``.
- ``status``: what is cached.

Registered by the main CLI (``app.add_typer(alt_app, name="alt")``).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import typer

alt_app = typer.Typer(help="Alternative data: OpenDART / NPS / SEC caches and the job-posting forward ledger.")


def _settings(config: Path | None) -> Any:
    from quantcio.config import load_settings

    return load_settings(config)


def _universe_entities(market: Any, settings: Any, cache: Any, with_xsh: bool) -> tuple[list[str], dict]:
    """Entities of the modeling universe (and, KR, every name of the configured XSH rank band) + their names."""
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(market, settings)
    if market is Market.KR and with_xsh:
        n = max(n, int(settings.holdout.xsh_kr_rank_band[1]))
    uni = load_pit_universe(market, settings, cache, n, start)
    mem = uni.membership
    if market is Market.KR and with_xsh:
        lo = int(settings.holdout.xsh_kr_rank_band[0])
        minrank = mem.groupby("entity")["rank"].min()
        keep = set(minrank[(minrank <= universe_params(market, settings)[0]) | (minrank >= lo)].index.astype(str))
        mem = mem[mem["entity"].astype(str).isin(keep)]
    names = {}
    if "name" in mem:
        names = mem.groupby("entity")["name"].agg(lambda x: sorted(set(map(str, x.dropna())))).to_dict()
    return sorted(mem["entity"].astype(str).unique()), names


@alt_app.command("fetch")
def fetch(
    market: str = typer.Option("KR", "--market", help="KR | US"),
    with_xsh: bool = typer.Option(True, "--with-xsh/--no-xsh", help="KR: also the XSH band names"),
    config: Path | None = typer.Option(None, "--config", help="override YAML"),
) -> None:
    """Refresh the alternative-data caches of the modeling universe."""
    from quantcio.data.alt.load import load_alt_data
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market

    settings = _settings(config)
    mk = Market(market.strip().upper())
    cache = DiskCache(settings.cache_path)
    ents, names = _universe_entities(mk, settings, cache, with_xsh)
    typer.echo(f"{mk.value}: {len(ents)} entities")
    alt = load_alt_data(mk, settings, ents, date(2012, 1, 1), names=names, progress=typer.echo)
    typer.echo(json.dumps(alt.coverage, ensure_ascii=False, default=str, indent=1))
    for n in alt.notes:
        typer.echo(f"- {n}")
    typer.echo(f"events {len(alt.events):,} · measures {len(alt.measures):,}")


@alt_app.command("jobs")
def jobs(
    max_calls: int = typer.Option(450, "--max-calls", help="API calls per source (Saramin: 500/day/key)"),
    ledger: Path | None = typer.Option(None, "--ledger", help="append-only JSONL (default alt_ledger/kr_jobs.jsonl)"),
    config: Path | None = typer.Option(None, "--config", help="override YAML"),
) -> None:
    """Append today's job-posting snapshot of the current KR universe (forward recording only)."""
    from quantcio.data.alt import kr_jobs
    from quantcio.data.alt.kr_nps import dart_companies
    from quantcio.data.cache import DiskCache
    from quantcio.models import Market

    settings = _settings(config)
    if not (settings.data.saramin_api_key or settings.data.work24_api_key):
        typer.echo("SARAMIN_API_KEY / WORK24_API_KEY 가 없습니다 (.env) — 채용 공고 기록을 건너뜁니다")
        raise typer.Exit(1)
    cache = DiskCache(settings.cache_path)
    ents, names = _universe_entities(Market.KR, settings, cache, with_xsh=False)
    comps, _cnt, notes = dart_companies(settings, ents)
    companies = {e: [*(comps[e].names() if e in comps else []), *names.get(e, [])] for e in ents}
    bizr = {e: c.bizr_no for e, c in comps.items() if c.bizr_no}
    df = kr_jobs.snapshot_job_postings(
        companies,
        date.today(),
        saramin_key=settings.data.saramin_api_key,
        work24_key=settings.data.work24_api_key,
        bizr_nos=bizr,
        max_calls=max_calls,
    )
    n = kr_jobs.append_snapshot(df, ledger or kr_jobs.DEFAULT_LEDGER)
    for x in [*notes, *df.attrs.get("notes", [])]:
        typer.echo(f"- {x}")
    typer.echo(f"{n} new ledger rows ({len(df)} snapshot rows) -> {ledger or kr_jobs.DEFAULT_LEDGER}")


@alt_app.command("status")
def status(config: Path | None = typer.Option(None, "--config", help="override YAML")) -> None:
    """What the alternative-data caches hold."""
    settings = _settings(config)
    root = settings.cache_path / "alt"
    for sub in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
        files = [f for f in sub.rglob("*") if f.is_file()]
        mb = sum(f.stat().st_size for f in files) / 1e6
        typer.echo(f"{sub.name}: {len(files):,} files, {mb:,.1f} MB")
    from quantcio.data.alt import kr_jobs

    if kr_jobs.DEFAULT_LEDGER.exists():
        led = kr_jobs.read_ledger(kr_jobs.DEFAULT_LEDGER)
        typer.echo(f"jobs ledger: {len(led):,} rows, {led['date'].nunique() if len(led) else 0} snapshot dates")
    else:
        typer.echo("jobs ledger: none yet (quantcio alt jobs)")


__all__ = ["alt_app"]
