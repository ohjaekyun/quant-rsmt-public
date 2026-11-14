"""Minimal bilingual text helper (report language ``settings.report.language``: "ko" | "en").

Usage at the place a user-facing sentence is built::

    from quantcio.i18n import tr
    note = tr(settings, f"매크로 오버레이: 한도 {a:.0%} → {b:.0%}", f"Macro overlay: gross cap {a:.0%} → {b:.0%}")

Keep both variants next to each other (no separate catalogue), so a new sentence cannot silently lack a
translation. Numbers and tickers are formatted identically in both languages.
"""

from __future__ import annotations

from typing import Any

LANGS = ("ko", "en")


def lang_of(obj: Any) -> str:
    """Language code from a Settings, ReportConfig, a plain string, or None (-> "ko")."""
    if obj is None:
        return "ko"
    if isinstance(obj, str):
        return obj if obj in LANGS else "ko"
    report = getattr(obj, "report", None)
    lang = getattr(report, "language", None) if report is not None else getattr(obj, "language", None)
    return lang if lang in LANGS else "ko"


def tr(obj: Any, ko: str, en: str) -> str:
    """Pick the Korean or English variant for the configured report language."""
    return en if lang_of(obj) == "en" else ko


def is_en(obj: Any) -> bool:
    return lang_of(obj) == "en"
