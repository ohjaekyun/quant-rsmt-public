"""quantcio PDF rendering: Jinja2 HTML -> A4 PDF via Playwright + system Chrome.

Verified on macOS (Darwin 25.6), Python 3.12.13, playwright 1.63.0,
Google Chrome 153.0.8010.53, pypdf 6.19.0, matplotlib 3.11.2.
"""
from __future__ import annotations

import io
import itertools
import os
import re
import tempfile
from pathlib import Path

from playwright.sync_api import Browser, Error as PlaywrightError, sync_playwright

# ---------------------------------------------------------------- page counting
_PAGE_RE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def count_pdf_pages(pdf: Path | bytes) -> int:
    """Page count. pypdf if available, else regex over raw bytes (Chrome/Skia PDFs
    do not use object streams, so '/Type /Page' objects are visible in plain text)."""
    data = pdf if isinstance(pdf, (bytes, bytearray)) else Path(pdf).read_bytes()
    try:
        from pypdf import PdfReader

        return len(PdfReader(io.BytesIO(data)).pages)
    except ImportError:
        return len(_PAGE_RE.findall(data))


# ---------------------------------------------------------------- browser launch
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


class NoBrowserError(RuntimeError):
    pass


def _launch(p) -> Browser:
    """Primary: installed Google Chrome (channel='chrome'; override with env
    QUANTCIO_BROWSER_CHANNEL, e.g. 'msedge' on Windows).
    Fallback 1: explicit executable (env QUANTCIO_CHROME, else the macOS default path).
    Fallback 2: Playwright's bundled headless shell
    (`python -m playwright install --only-shell chromium`, ~100 MB, not run here)."""
    channel = os.environ.get("QUANTCIO_BROWSER_CHANNEL", "chrome")
    attempts: list[tuple[str, dict]] = [(f"channel={channel}", {"channel": channel})]
    exe = os.environ.get("QUANTCIO_CHROME") or MAC_CHROME
    if Path(exe).exists():
        attempts.append((f"executable_path={exe}", {"executable_path": exe}))
    attempts.append(("bundled chromium-headless-shell", {}))
    errors = []
    for label, kw in attempts:
        try:
            return p.chromium.launch(headless=True, **kw)
        except PlaywrightError as e:  # browser missing / failed to start
            errors.append(f"{label}: {str(e).splitlines()[0]}")
    raise NoBrowserError(
        "No Chromium-family browser could be launched. Install Google Chrome, or run "
        "`python -m playwright install --only-shell chromium`.\n  " + "\n  ".join(errors)
    )


FOOTER_TEMPLATE = (
    "<div style=\"width:100%;font-size:7px;color:#8a8a8a;padding:0 12mm;"
    "font-family:'Apple SD Gothic Neo','Noto Sans KR',sans-serif;"
    "display:flex;justify-content:space-between;\">"
    "<span>quantcio · 투자 참고용 · 투자 권유 아님</span>"
    "<span><span class=\"pageNumber\"></span> / <span class=\"totalPages\"></span></span></div>"
)


class PdfRenderer:
    """Re-usable renderer: one Chrome process for many reports. Measured: Chrome
    launch+close ~0.25 s, ~0.3 s per 2-page report when reused; a one-shot
    render_pdf() is ~1.4 s warm (driver start + launch + render), ~5 s on the
    very first cold run after boot."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None

    def __enter__(self) -> "PdfRenderer":
        self._pw = sync_playwright().start()
        try:
            self._browser = _launch(self._pw)
        except BaseException:
            # __exit__ is NOT called when __enter__ raises; without this stop() the
            # next sync_playwright() in the process fails with "It looks like you are
            # using Playwright Sync API inside the asyncio loop".
            self._pw.stop()
            self._pw = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        # VERIFY-FIX: stop the driver even if browser.close() raises (e.g. Chrome
        # crashed), otherwise later sync_playwright() calls in the process fail.
        try:
            if self._browser:
                self._browser.close()
        finally:
            self._browser = None
            if self._pw:
                self._pw.stop()
                self._pw = None

    def pdf_bytes(
        self,
        html: str,
        *,
        scale: float = 1.0,
        footer: bool = True,
        base_dir: Path | None = None,
        timeout_ms: int = 30_000,
    ) -> bytes:
        assert self._browser is not None, "use `with PdfRenderer() as r:`"
        page = self._browser.new_page()
        try:
            page.set_default_timeout(timeout_ms)
            page.emulate_media(media="print")
            if base_dir is None:
                # self-contained HTML (inline CSS/SVG/data-URIs): no file needed
                page.set_content(html, wait_until="load")
            else:
                # relative <img src="x.png"> etc. need a real file:// URL
                base_dir = Path(base_dir)
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".html", dir=base_dir, delete=False, encoding="utf-8"
                ) as f:
                    f.write(html)
                    tmp = Path(f.name)
                try:
                    page.goto(tmp.as_uri(), wait_until="load")
                finally:
                    tmp.unlink(missing_ok=True)
            page.evaluate("document.fonts.ready.then(() => true)")  # wait for webfonts
            return page.pdf(
                format="A4",
                print_background=True,           # keep badge / table background colors
                prefer_css_page_size=True,       # @page { size: A4 } wins
                # @page margin in CSS overrides these when present; they are the
                # fallback and also the area where the footer template is drawn.
                margin={"top": "12mm", "bottom": "14mm", "left": "12mm", "right": "12mm"},
                scale=scale,                     # 0.1..2.0 ; used by fit_to_pages()
                display_header_footer=footer,
                header_template="<span></span>",
                footer_template=FOOTER_TEMPLATE if footer else "<span></span>",
                outline=True,                    # PDF bookmarks from <h1>-<h6>
                tagged=True,                     # accessible/tagged PDF
            )
        finally:
            page.close()


def render_pdf(
    html: str,
    out_path: Path,
    *,
    footer: bool = True,
    base_dir: Path | None = None,
    html_fallback: bool = True,
) -> int:
    """Render HTML to an A4 PDF at out_path. Returns the page count.
    If no browser can be launched and html_fallback=True, writes
    out_path.with_suffix('.html') instead and returns 0."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PdfRenderer() as r:
            data = r.pdf_bytes(html, footer=footer, base_dir=base_dir)
    except NoBrowserError as e:
        if not html_fallback:
            raise
        p = save_html_fallback(html, out_path)
        print(f"[quantcio] PDF skipped ({str(e).splitlines()[0]}); wrote {p}")
        return 0
    out_path.write_bytes(data)
    return count_pdf_pages(data)


def fit_to_pages(
    html: str,
    out_path: Path,
    *,
    max_pages: int = 3,
    scales: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8),
    footer: bool = True,
    base_dir: Path | None = None,
    html_fallback: bool = True,
) -> tuple[int, float]:
    """Enforce the 2-3 page limit: shrink print scale until pages <= max_pages.
    Returns (page_count, scale_used). If even the smallest scale overflows, the
    smallest-scale PDF is written and the caller should trim content (e.g. drop
    the lowest-priority news items) and retry (check `page_count > max_pages`).
    VERIFY-FIX: same no-browser HTML fallback and base_dir as render_pdf();
    returns (0, 0.0) when it fell back to HTML."""
    if not scales:
        raise ValueError("scales must not be empty")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PdfRenderer() as r:
            for s in scales:
                data = r.pdf_bytes(html, scale=s, footer=footer, base_dir=base_dir)
                n = count_pdf_pages(data)
                if n <= max_pages:
                    break
    except NoBrowserError as e:
        if not html_fallback:
            raise
        p = save_html_fallback(html, out_path)
        print(f"[quantcio] PDF skipped ({str(e).splitlines()[0]}); wrote {p}")
        return 0, 0.0
    out_path.write_bytes(data)
    return n, s


def save_html_fallback(html: str, out_path: Path) -> Path:
    """Last resort when no browser is available: ship the HTML (open in any
    browser -> Print -> Save as PDF; @page CSS keeps it A4)."""
    p = Path(out_path).with_suffix(".html")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------- matplotlib -> inline SVG
def setup_matplotlib_korean() -> str:
    """Korean-capable font + real minus sign. Returns the font family chosen."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt

    have = {f.name for f in fm.fontManager.ttflist}
    for fam in ("Apple SD Gothic Neo", "AppleGothic", "NanumGothic", "Noto Sans KR", "Noto Sans CJK KR", "Malgun Gothic"):
        if fam in have:
            break
    else:
        fam = "DejaVu Sans"  # Korean will be tofu; log a warning in the app
    plt.rcParams.update(
        {
            # VERIFY-FIX: a fallback list (matplotlib >= 3.6) so glyphs the Korean font
            # lacks (U+2212 inside label/title *strings*, which axes.unicode_minus does
            # not cover; also some symbols) come from DejaVu Sans instead of tofu.
            "font.family": [fam, "DejaVu Sans"] if fam != "DejaVu Sans" else fam,
            "axes.unicode_minus": False,   # otherwise U+2212 minus -> missing glyph box
            "svg.fonttype": "path",        # glyphs as vector paths: PDF independent of fonts
            "svg.hashsalt": "quantcio",    # deterministic ids in the SVG
        }
    )
    return fam


_SVG_HEAD = re.compile(r"^.*?(?=<svg)", re.S)
_svg_seq = itertools.count(1)


def fig_to_svg(fig, *, css_width: str = "100%") -> str:
    """Matplotlib figure -> inline <svg> string (XML prolog/DOCTYPE stripped,
    fixed pt width/height replaced so CSS controls the size, viewBox kept,
    every id prefixed per chart so many inline SVGs never share ids)."""
    import matplotlib.pyplot as plt

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    svg = _SVG_HEAD.sub("", buf.getvalue(), count=1)
    svg = re.sub(r'<svg([^>]*?)\swidth="[^"]+"\s+height="[^"]+"', rf'<svg\1 width="{css_width}"', svg, count=1)
    pre = f"c{next(_svg_seq)}-"
    ids = set(re.findall(r'\sid="([^"]+)"', svg))
    svg = re.sub(r'(\sid=")([^"]+)"', lambda m: f'{m.group(1)}{pre}{m.group(2)}"', svg)
    svg = re.sub(r'(href="#|url\(#)([^")]+)',
                 lambda m: m.group(1) + pre + m.group(2) if m.group(2) in ids else m.group(0), svg)
    return svg


# ---------------------------------------------------------------- Jinja env (VERIFY-FIX)
MISSING = "—"


def _is_missing(v) -> bool:
    """None, float/np.float32/np.float64 NaN, pd.NA/pd.NaT."""
    if v is None:
        return True
    try:
        return bool(v != v)
    except (TypeError, ValueError):  # bool(pd.NA) raises TypeError
        return True


def _safe_format(fmt: str, *args, **kw) -> str:
    """Drop-in for Jinja's `format` filter: '%+.1f'|format(nan) would print '+nan'."""
    if any(_is_missing(a) for a in args) or any(_is_missing(v) for v in kw.values()):
        return MISSING
    return fmt % (kw or args)


def none_to_nan(obj):
    """Recursively turn None into float('nan') so template arithmetic such as
    `r.upside*100` or `r.upside >= 0` never raises TypeError (it did with None)."""
    if obj is None:
        return float("nan")
    if isinstance(obj, dict):
        return {k: none_to_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [none_to_nan(v) for v in obj]
    return obj


def make_jinja_env(template_dir: Path):
    """Autoescaped env (.html/.j2) whose `format` filter renders None/NaN as '—'.
    Call template.render(rows=[none_to_nan(r) for r in rows], ...)."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(loader=FileSystemLoader(str(template_dir)),
                      autoescape=select_autoescape(["html", "j2"]))
    env.filters["format"] = _safe_format
    return env
