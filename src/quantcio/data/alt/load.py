"""Load every enabled alternative-data source of one market into one ``AltData`` (settings ``alt`` block).

KR: OpenDART (``kr_dart_events``) + the National Pension Service file (``kr_nps``) when the pre-declared rule
allows it (``alt.kr_nps``: ``auto`` = only when past monthly files reach back to ``alt.nps_min_month``,
docs/MODELING.md 16.3). US: SEC submissions forms + the Form 3/4/5 insider data sets (``us_sec``).
A source that is disabled, has no key or fails leaves its features NaN and says so in ``notes`` - never silent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from datetime import date

import pandas as pd

from quantcio.config import Settings
from quantcio.data.alt import AltData
from quantcio.models import Market

log = logging.getLogger(__name__)


def nps_allowed(settings: Settings, progress: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """(include NPS features?, reason) under the pre-declared rule (``alt.kr_nps``)."""
    mode = str(settings.alt.kr_nps).lower()
    if mode in ("off", "false", "0"):
        return False, "alt.kr_nps = off"
    if mode in ("on", "true", "1"):
        return True, "alt.kr_nps = on"
    try:
        from quantcio.data.alt.kr_nps import earliest_data_month, nps_available_months
    except ImportError as exc:
        return False, f"NPS loader unavailable ({exc})"
    try:
        earliest_s, verified = earliest_data_month(nps_available_months(settings))
    except Exception as exc:  # noqa: BLE001 - the rule cannot be checked: excluded, said so
        return False, f"NPS month listing failed ({type(exc).__name__}: {exc})"
    if not earliest_s:
        return False, "no past NPS monthly file found"
    earliest = pd.Period(str(earliest_s)[:7], "M")
    need = pd.Period(settings.alt.nps_min_month, "M")
    tag = "verified" if verified else "estimated from the file title"
    if earliest <= need:
        return True, f"NPS history from {earliest} ({tag}; <= {need})"
    return False, f"NPS history starts {earliest} ({tag}; > {need}): forward recording only (docs/MODELING.md 16.3)"


def load_alt_data(
    market: Market | str,
    settings: Settings,
    entities: Iterable[str],
    start: date,
    *,
    names: Mapping[str, Iterable[str]] | None = None,
    refresh_days: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> AltData:
    """All enabled alternative data of ``entities`` (modeling entity ids) published from ``start``."""
    mk = Market(str(getattr(market, "value", market)).upper())
    say = progress or (lambda _m: None)
    ents = sorted({str(e) for e in entities})
    alt = AltData(mk.value)
    cfg = settings.alt
    refresh = cfg.refresh_days if refresh_days is None else refresh_days
    if not cfg.enabled:
        alt.notes.append("대안 데이터 꺼짐(alt.enabled = false)")
        return alt
    if mk is Market.KR:
        if cfg.kr_dart:
            from quantcio.data.alt.kr_dart_events import load_kr_dart_alt

            say("alt: OpenDART")
            try:
                dart = load_kr_dart_alt(settings, ents, start, names=names, refresh_days=refresh, progress=say)
                alt = alt.merged(dart)
            except Exception as exc:  # noqa: BLE001 - the source is skipped loudly
                log.exception("OpenDART alt data failed")
                alt.notes.append(f"OpenDART 대안 데이터 실패: {type(exc).__name__}")
        ok, why = nps_allowed(settings, say)
        alt.coverage["kr_nps_rule"] = {"included": ok, "reason": why}
        if ok:
            from quantcio.data.alt.kr_nps import load_kr_nps

            say("alt: NPS")
            try:
                nps = load_kr_nps(
                    settings,
                    ents,
                    start,
                    data_go_kr_key=settings.data.data_go_kr_key,
                    progress=lambda i, n, msg: say(f"NPS {i}/{n} {msg}"),
                )
                alt = alt.merged(nps)
            except Exception as exc:  # noqa: BLE001
                log.exception("NPS alt data failed")
                alt.notes.append(f"국민연금 대안 데이터 실패: {type(exc).__name__}")
        else:
            alt.notes.append(f"국민연금 특징 제외: {why}")
        return alt
    if cfg.us_sec:
        try:
            from quantcio.data.alt.us_sec import load_us_sec_alt
            from quantcio.data.pit.us_cik import entity_cik
        except ImportError as exc:
            alt.notes.append(f"SEC 대안 데이터 모듈 없음: {exc}")
            return alt
        ciks = []
        for e in ents:
            try:
                ciks.append(entity_cik(e))
            except (ValueError, TypeError):
                continue
        # submissions JSONs hold the newest filings: refreshed daily (the build itself is cached for a week)
        refreshed = refresh_sec_submissions(settings, ciks, min(refresh, 1) if refresh is not None else None)
        if refreshed:
            say(f"alt: SEC submissions older than {refresh} days refreshed: {refreshed}")
        say("alt: SEC")
        try:
            alt = alt.merged(load_us_sec_alt(settings, ciks, start, progress=say))
        except Exception as exc:  # noqa: BLE001
            log.exception("SEC alt data failed")
            alt.notes.append(f"SEC 대안 데이터 실패: {type(exc).__name__}")
    return alt


def refresh_sec_submissions(settings: Settings, ciks: Iterable[int], max_age_days: int | None) -> int:
    """Drop cached SEC submissions main JSONs older than ``max_age_days`` so the next read refetches them
    (the shared ``Submissions`` cache never expires; older ``files`` pages hold closed history and are kept).
    Returns the number dropped."""
    if max_age_days is None:
        return 0
    import time

    from quantcio.data.pit.us_net import PitNet

    net = PitNet(settings)
    limit = time.time() - float(max_age_days) * 86400
    n = 0
    for cik in ciks:
        path = net.path("sec", f"subs/{int(cik)}.json.gz")
        try:
            if path.exists() and path.stat().st_mtime < limit:
                path.unlink()
                n += 1
        except OSError as exc:
            log.warning("SEC submissions refresh %s: %s", cik, exc)
    return n


def nps_included(alt: AltData | None) -> bool:
    """Whether the NPS features belong to the feature set of this dataset (the rule's decision at load time)."""
    if alt is None:
        return False
    return bool((alt.coverage.get("kr_nps_rule") or {}).get("included"))


__all__ = ["load_alt_data", "nps_allowed", "nps_included", "refresh_sec_submissions"]
