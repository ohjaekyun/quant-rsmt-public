"""``quantcio modeling`` sub-app: ``select --market KR|US`` and ``show --market KR|US``.

Registered by the main CLI (``app.add_typer(modeling_app, name="modeling")``).

``select`` builds the PIT modeling panel (cached), runs the nested walk-forward selection (every context is
checkpointed in the disk cache, so an interrupted run resumes), writes ``reports/modeling/<M>_<as_of>.md|json``
and caches the serving artifact that ``quantcio report`` uses. Long: KR ~10 min, US ~25 min (cold); scripts that
call it must guard with ``if __name__ == "__main__":`` (the US PIT loader may spawn a process pool).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import typer

from quantcio.i18n import tr

modeling_app = typer.Typer(help="Stock-selection model: PIT dataset, nested walk-forward selection, serving artifact.")


def _settings(config: Path | None, lang: str | None) -> Any:
    from quantcio.config import load_settings

    over: dict[str, Any] = {}
    if lang:
        over["report"] = {"language": lang}
    return load_settings(config, **over)


def serving_summary(art: Any) -> dict[str, Any]:
    """Build-time serving summary (before any scoring-time XSH confirmation: ``multiplier`` is the unconfirmed
    one; ``multiplier_if_confirmed`` applies only when the gate passed and XSH confirms it)."""
    lat = art.latest
    shrink = float((art.calibration or {}).get("shrink") or 0.0)
    gate_passed = bool(getattr(art, "gate_passed", art.skill_ok))
    return {
        "as_of": str(art.as_of.date()),
        "n": len(lat),
        "selected": art.selected,
        "members": art.members,
        "skill_ok": art.skill_ok,
        "gate_passed": gate_passed,
        "post_hoc": bool(getattr(art, "post_hoc", False)),
        "primary_auc": getattr(art, "primary_auc", "auc"),
        "multiplier": art.multiplier,
        "multiplier_if_confirmed": shrink if gate_passed else art.multiplier,
        "mu_min": float(lat["mu_excess"].min()) if len(lat) else None,
        "mu_max": float(lat["mu_excess"].max()) if len(lat) else None,
        "market_p_up": (art.market_model or {}).get("served_p_up"),
    }


def guard_xsh(ds: Any, settings: Any, say: Any = None) -> Any:
    """The dataset without the registered cross-sectional holdout (XSH) names of its market (docs/HOLDOUT.md).

    Dates are kept: ``select`` also builds the serving artifact. A registry that cannot be read stops the run (the
    held-out names could leak into tuning). When rows are dropped the snapshot changes, so no checkpoint built with
    them is reused.
    """
    from quantcio import holdout

    try:
        frame, n = holdout.drop_xsh_names(ds.frame, ds.market, settings)
    except holdout.HoldoutError as exc:
        raise typer.BadParameter(f"holdout registry: {exc}") from exc
    if not n:
        return ds
    ds.frame = frame.reset_index(drop=True)
    ds.snapshot = f"{ds.snapshot}|xsh-{n}"
    ds.notes = [*ds.notes, {"ko": f"[홀드아웃] XSH 종목 {n}행 제외", "en": f"[holdout] {n} XSH rows excluded"}]
    if say is not None:
        say(f"holdout guard: {n} XSH rows excluded")
    return ds


@modeling_app.command("select")
def select_cmd(
    market: str = typer.Option(..., "--market", help="KR | US"),
    threads: int = typer.Option(8, help="LightGBM threads"),
    no_cache: bool = typer.Option(False, "--no-cache", help="ignore dataset / context checkpoints"),
    out: Path | None = typer.Option(None, "--out", help="report directory (default reports/modeling)"),
    config: Path | None = typer.Option(None, "--config", help="override YAML"),
    lang: str | None = typer.Option(None, "--lang", help="ko | en"),
) -> None:
    """Run the full model selection for one market and cache the serving artifact."""
    from quantcio.data.cache import DiskCache
    from quantcio.modeling.dataset import build_dataset
    from quantcio.modeling.report import summary_lines, write_report
    from quantcio.modeling.selection import RESULT_NS, run_selection
    from quantcio.modeling.serve import build_artifact, save_artifact
    from quantcio.models import Market

    logging.basicConfig(level=logging.WARNING)
    s = _settings(config, lang)
    mk = Market(market.upper())
    cache = DiskCache(s.cache_path)
    t0 = time.monotonic()

    def say(msg: str) -> None:
        typer.echo(f"[{time.monotonic() - t0:7.1f}s] {msg}")

    ds = build_dataset(mk, s, cache, use_cache=not no_cache, progress=say)
    ds = guard_xsh(ds, s, say)
    res = run_selection(ds, s, cache, threads=threads, use_cache=not no_cache, progress=say)
    art = build_artifact(res, ds, s)
    serving = serving_summary(art)
    md, js = write_report(res, s, out, serving)
    art.report_path = str(md)
    save_artifact(art, s, cache)
    light = res.to_dict()
    light["summary_ko"], light["summary_en"] = summary_lines(res)
    light["serving"] = serving
    light["report"] = {"md": str(md), "json": str(js)}
    cache.set(RESULT_NS, mk.value, light)
    ko, en = summary_lines(res)
    typer.echo(tr(s, ko, en))
    typer.echo(tr(s, f"리포트: {md} / {js}", f"Report: {md} / {js}"))


@modeling_app.command("show")
def show_cmd(
    market: str = typer.Option(..., "--market", help="KR | US"),
    config: Path | None = typer.Option(None, "--config", help="override YAML"),
    lang: str | None = typer.Option(None, "--lang", help="ko | en"),
) -> None:
    """Print the cached selection summary and serving artifact of one market."""
    from quantcio.data.cache import DiskCache
    from quantcio.modeling.selection import RESULT_NS
    from quantcio.modeling.serve import load_artifact
    from quantcio.models import Market

    s = _settings(config, lang)
    mk = Market(market.upper())
    cache = DiskCache(s.cache_path)
    res = cache.get(RESULT_NS, mk.value, None)
    if not res:
        typer.echo(
            tr(
                s,
                f"{mk.value}: 선택 결과 없음 — `quantcio modeling select --market {mk.value}` 실행",
                f"{mk.value}: no selection result - run `quantcio modeling select --market {mk.value}`",
            )
        )
        raise typer.Exit(1)
    typer.echo(tr(s, res.get("summary_ko", ""), res.get("summary_en", "")))
    from quantcio.modeling.metrics import SECONDARY_AUC, auc_label

    key = str((res.get("gate") or {}).get("metric") or "auc")
    lang = s.report.language
    typer.echo(
        tr(
            s,
            f"주 지표 = {auc_label(key, lang)}, 보조 = {auc_label(SECONDARY_AUC, lang)}",
            f"primary = {auc_label(key, lang)}, secondary = {auc_label(SECONDARY_AUC, lang)}",
        )
    )
    for c in res.get("candidates", []):
        o, h = c.get("outer") or {}, c.get("holdout") or {}
        oa, oi, ha = o.get(key) or {}, o.get("ic") or {}, h.get(key) or {}
        o2 = o.get(SECONDARY_AUC) or {}

        def f(x: Any, nd: int = 4) -> str:
            return "   –  " if x is None else f"{x:.{nd}f}"

        mark = "*" if c["name"] == res.get("selected") else " "
        typer.echo(
            f"{mark} {c['name']:<11} {key} {f(oa.get('mean'))} [{f(oa.get('lo'))},{f(oa.get('hi'))}] "
            f"{SECONDARY_AUC} {f(o2.get('mean'))} IC {f(oi.get('mean'))} (t {f(oi.get('t'), 2)}) | holdout {key} "
            f"{f(ha.get('mean'))}"
        )
    art = load_artifact(mk, s, cache)
    if art is not None:
        from quantcio.modeling.serve import gate_verdict, served_state

        st = served_state(art, s)  # incl. the scoring-time XSH confirmation lookup
        vko, ven = gate_verdict(st)
        sv = serving_summary(art)
        typer.echo(
            tr(
                s,
                f"서빙 산출물: {sv['as_of']} · {sv['n']}종목 · 게이트 {vko} · 서빙 배수 {st['multiplier']:.2f}",
                f"Serving artifact: {sv['as_of']} · {sv['n']} names · gate {ven} · served multiplier "
                f"{st['multiplier']:.2f}",
            )
        )
    rep = res.get("report") or {}
    if rep:
        typer.echo(tr(s, f"리포트: {rep.get('md')}", f"Report: {rep.get('md')}"))


__all__ = ["guard_xsh", "modeling_app", "serving_summary"]
