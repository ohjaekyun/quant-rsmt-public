"""A4 CIO report (Korean by default, English with settings.report.language == "en"): HTML/PDF rendering, small
charts and a deterministic sample report.

Typical use (pipeline / CLI)::

    from quantcio.report import fit_to_pages_isolated, price_chart_svg

    charts = {t.instrument.ticker: price_chart_svg(md.prices, t.execution, t.executive.target_price) ...}
    result = fit_to_pages_isolated(report, settings, out_dir / "report.pdf", charts=charts)  # watchdog subprocess
    if result.fallback_html or not result.fits:
        warnings.extend(result.warnings)          # never silent

`fit_to_pages_isolated` (report/worker.py) runs `fit_to_pages` in a subprocess with a timeout, one retry and an HTML
fallback - the production path; `fit_to_pages` itself renders in-process (tests, callers holding a PdfRenderer).
Lower-level: `render_html(report, settings, compact_level, charts)` and `render_pdf(html, out_path)`.
Playwright and matplotlib are imported lazily, so importing this package is cheap.
"""

from quantcio.report.charts import fig_to_svg, price_chart_svg, sparkline_svg
from quantcio.report.render import (
    DISCLAIMER,
    DISCLAIMER_EN,
    MISSING,
    MODULE_HEADINGS,
    REPORT_TITLE,
    REPORT_TITLE_EN,
    FitResult,
    NoBrowserError,
    PdfRenderer,
    count_pdf_pages,
    fit_to_pages,
    portfolio_risk_labels,
    render_html,
    render_pdf,
    save_html_fallback,
)
from quantcio.report.sample import SAMPLE_NOTE, SAMPLE_NOTE_EN, build_sample_charts, build_sample_report
from quantcio.report.worker import RENDER_TIMEOUT_S, fit_to_pages_isolated

__all__ = [
    "DISCLAIMER",
    "DISCLAIMER_EN",
    "MISSING",
    "MODULE_HEADINGS",
    "REPORT_TITLE",
    "REPORT_TITLE_EN",
    "RENDER_TIMEOUT_S",
    "SAMPLE_NOTE",
    "SAMPLE_NOTE_EN",
    "FitResult",
    "NoBrowserError",
    "PdfRenderer",
    "build_sample_charts",
    "build_sample_report",
    "count_pdf_pages",
    "fig_to_svg",
    "fit_to_pages",
    "fit_to_pages_isolated",
    "portfolio_risk_labels",
    "price_chart_svg",
    "render_html",
    "render_pdf",
    "save_html_fallback",
    "sparkline_svg",
]
