"""Small inline-SVG charts for the A4 report.

- `price_chart_svg`: 1-year close with MA50/MA200, the entry zone and the stop-loss / take-profit /
  target levels of the execution plan (matplotlib, rendered to vector SVG).
- `sparkline_svg`: dependency-free polyline sparkline (pure SVG string).

matplotlib is used through the object-oriented `Figure` API only (no pyplot global state, no backend
switching), and every rcParam is scoped with `rc_context`, so importing this module never changes the
caller's matplotlib configuration. Chart text is drawn as vector paths (`svg.fonttype="path"`), so the PDF
does not depend on fonts being installed in Chrome.

Gotchas carried over from the verified renderer (docs/sources/reference/render_pdf.py):
- Korean fonts lack U+2212, so `axes.unicode_minus=False` plus a DejaVu Sans fallback in `font.family`.
- Several inline SVGs in one HTML page share glyph ids; `fig_to_svg` prefixes every id per chart.
"""

from __future__ import annotations

import functools
import hashlib
import io
import itertools
import logging
import math
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from quantcio.models import ExecutionPlan

log = logging.getLogger(__name__)

__all__ = [
    "CHART_COLORS",
    "korean_font_family",
    "fig_to_svg",
    "price_chart_svg",
    "sparkline_svg",
]

KOREAN_FONT_CANDIDATES = (
    "Apple SD Gothic Neo",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans KR",
    "Noto Sans CJK KR",
    "Malgun Gothic",
)

# One palette for the whole report (report.css uses the same values).
CHART_COLORS = {
    "close": "#1b1b1a",
    "ma50": "#2a78d6",
    "ma200": "#eb6834",
    "tp": "#1a7f3c",
    "sl": "#c62828",
    "target": "#0b5e2f",
    "entry": "#2a78d6",
    "grid": "#e6e5e0",
    "axis": "#8a8985",
    "text": "#52514e",
}


@functools.lru_cache(maxsize=1)
def korean_font_family() -> str | None:
    """First installed Korean-capable font family, or None (logged once) when none is available."""
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for fam in KOREAN_FONT_CANDIDATES:
        if fam in have:
            return fam
    log.warning(
        "No Korean font found for matplotlib (tried %s); chart labels fall back to ASCII.",
        ", ".join(KOREAN_FONT_CANDIDATES),
    )
    return None


def _rc(font_size: float) -> dict:
    fam = korean_font_family()
    return {
        "font.family": [fam, "DejaVu Sans"] if fam else ["DejaVu Sans"],
        "font.size": font_size,
        "axes.unicode_minus": False,  # U+2212 is missing from Korean fonts -> tofu box
        "svg.fonttype": "path",  # glyphs as vector paths: PDF independent of installed fonts
        "svg.hashsalt": "quantcio",  # deterministic ids (then prefixed per chart by fig_to_svg)
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 0.0,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
    }


_SVG_HEAD = re.compile(r"^.*?(?=<svg)", re.S)
_svg_seq = itertools.count(1)


def fig_to_svg(fig, *, css_width: str = "100%") -> str:
    """Matplotlib figure -> inline `<svg>` string.

    The XML prolog/DOCTYPE is stripped, the fixed pt width/height is replaced so CSS controls the size
    (the viewBox is kept), and every id plus its `href="#.."`/`url(#..)` references gets a per-chart
    prefix so many inline SVGs never share ids.
    """
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.01, transparent=True)
    svg = _SVG_HEAD.sub("", buf.getvalue(), count=1)
    svg = re.sub(r'<svg([^>]*?)\swidth="[^"]+"\s+height="[^"]+"', rf'<svg\1 width="{css_width}"', svg, count=1)
    digest = hashlib.sha1(svg.encode("utf-8")).hexdigest()[:6]
    pre = f"q{digest}{next(_svg_seq)}-"
    ids = set(re.findall(r'\sid="([^"]+)"', svg))
    svg = re.sub(r'(\sid=")([^"]+)"', lambda m: f'{m.group(1)}{pre}{m.group(2)}"', svg)
    svg = re.sub(
        r'(href="#|url\(#)([^")]+)',
        lambda m: m.group(1) + pre + m.group(2) if m.group(2) in ids else m.group(0),
        svg,
    )
    return svg


def _tick_label(v: float, scale: float) -> str:
    """Compact axis label: 286500 -> '287k' for large prices, 225.5 -> '226', 12.34 -> '12.3'."""
    if scale >= 100_000:
        return f"{v / 1000:,.0f}k"
    if scale >= 10:
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def price_chart_svg(
    prices: pd.DataFrame,
    plan: ExecutionPlan | None = None,
    target: float | None = None,
    *,
    lookback: int = 252,
    width_in: float = 2.05,
    height_in: float = 1.05,
    font_size: float = 5.6,
) -> str:
    """Small 1-year price chart as an inline SVG string.

    Draws the close, MA50 and MA200 (computed on the full history, so they exist over the whole window
    when enough data is present - a moving average without enough history is simply not drawn), the
    entry zone as a light band, and horizontal SL / TP / target levels labelled at the right edge.
    The y-range always includes the plotted levels so a stop-loss far below the window stays visible.

    Raises ValueError when `prices` has no usable close series (callers show "차트 없음" instead).
    """
    from matplotlib import dates as mdates
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if prices is None or "close" not in getattr(prices, "columns", []):
        raise ValueError("price_chart_svg: prices must be a DataFrame with a 'close' column")
    close = pd.to_numeric(prices["close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < 2:
        raise ValueError(f"price_chart_svg: need >= 2 valid closes, got {len(close)}")
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close = pd.Series(close.to_numpy(dtype=float), index=idx)

    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    view = close.iloc[-lookback:]
    x = view.index
    korean = korean_font_family() is not None
    c = CHART_COLORS

    def _valid(p: float | None) -> bool:
        try:
            return p is not None and math.isfinite(float(p)) and float(p) > 0
        except (TypeError, ValueError):
            return False

    # (price, label, color, linestyle); an invalid level (NaN / <= 0, e.g. an ATR-less plan) is skipped and
    # logged - it must neither crash the chart nor distort the y-range
    levels: list[tuple[float, str, str, object]] = []
    tp = plan.take_profit if plan is not None else None
    if plan is not None:
        for price, label, col in ((tp, "TP", c["tp"]), (plan.stop_loss, "SL", c["sl"])):
            if _valid(price):
                levels.append((float(price), label, col, "-"))
            else:
                log.warning("price_chart_svg: invalid %s level %r not drawn", label, price)
    if _valid(target):
        if _valid(tp) and abs(float(target) / float(tp) - 1.0) < 0.004:
            levels[0] = (float(tp), "TP·목표" if korean else "TP/TGT", c["tp"], "-")
        else:
            levels.append((float(target), "목표" if korean else "TGT", c["target"], (0, (2.2, 1.4))))
    entry_band = (
        plan is not None and _valid(plan.entry_low) and _valid(plan.entry_high) and plan.entry_high > plan.entry_low
    )

    with rc_context(_rc(font_size)):
        fig = Figure(figsize=(width_in, height_in))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        if entry_band:
            ax.axhspan(plan.entry_low, plan.entry_high, color=c["entry"], alpha=0.12, lw=0, zorder=1)
        for s, col in ((ma200, c["ma200"]), (ma50, c["ma50"])):
            seg = s.reindex(x)
            if seg.notna().any():
                ax.plot(x, seg.to_numpy(), color=col, lw=0.75, zorder=2, solid_capstyle="round")
        ax.plot(x, view.to_numpy(), color=c["close"], lw=0.95, zorder=3, solid_joinstyle="round")
        ax.plot([x[-1]], [view.iloc[-1]], marker="o", ms=2.2, color=c["close"], mec="white", mew=0.5, zorder=4)

        for price, _, col, ls in levels:
            ax.axhline(price, color=col, lw=0.7, ls=ls, zorder=2.5)

        lo = min([float(view.min())] + [p for p, *_ in levels])
        hi = max([float(view.max())] + [p for p, *_ in levels])
        pad = (hi - lo) * 0.07 or hi * 0.05
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(x[0], x[-1])

        # right-edge labels; nudge apart when levels are within ~9% of the axis height
        span = (hi + pad) - (lo - pad)
        placed: list[float] = []
        for price, label, col, _ in sorted(levels, key=lambda t: t[0], reverse=True):
            y = price
            for p in placed:
                if abs(y - p) < 0.09 * span:
                    y = p - 0.09 * span
            placed.append(y)
            ax.annotate(
                label,
                xy=(1.0, y),
                xycoords=("axes fraction", "data"),
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                color=col,
                fontsize=font_size * 0.95,
                fontweight="bold",
                annotation_clip=False,
            )

        scale = float(view.median())
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _tick_label(v, scale)))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
        ax.grid(axis="y", color=c["grid"], lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(c["axis"])
        ax.tick_params(colors=c["text"], labelsize=font_size)
        return fig_to_svg(fig)


def sparkline_svg(
    values: Sequence[float] | pd.Series,
    *,
    width: int = 90,
    height: int = 22,
    stroke: str = "#52514e",
    stroke_width: float = 1.2,
    end_dot: bool = True,
) -> str:
    """Tiny line chart as a self-contained SVG string (no ids, no fonts, scales with CSS width).

    NaNs are dropped. Raises ValueError with fewer than 2 finite values.
    """
    arr = np.asarray(pd.Series(values, dtype="float64").dropna().to_numpy(), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        raise ValueError(f"sparkline_svg: need >= 2 finite values, got {arr.size}")
    pad = stroke_width + (1.5 if end_dot else 0.5)
    lo, hi = float(arr.min()), float(arr.max())
    rng = hi - lo or 1.0
    xs = np.linspace(pad, width - pad, arr.size)
    ys = height - pad - (arr - lo) / rng * (height - 2 * pad)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dot = (
        f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="{stroke_width * 1.3:.1f}" fill="{stroke}"/>' if end_dot else ""
    )
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" '
        f'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{stroke_width}" '
        f'stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>{dot}</svg>'
    )
