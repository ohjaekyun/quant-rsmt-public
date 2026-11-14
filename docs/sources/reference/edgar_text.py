"""EDGAR (inline-XBRL) HTML -> clean text, plus 10-K/10-Q section extractor.

Verified on AAPL / NVDA / MSFT latest 10-K + 10-Q (fetched 2026-09-24) and
8-K EX-99.1 earnings releases.  Only deps: lxml (+ stdlib).
"""
from __future__ import annotations

import re
from collections import Counter
from lxml import html as LH

_BLOCK = {
    "p", "div", "br", "tr", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "hr", "title", "center", "blockquote", "pre", "dt", "dd",
    "dl", "table", "body", "page",
}
_DROP_TAGS = {"script", "style", "head", "noscript", "object", "embed", "svg"}
_HIDDEN_RE = re.compile(r"display\s*:\s*none", re.I)
# unicode spaces / zero-width chars built with chr() so the source stays pure ASCII
_SPACES = "".join(map(chr, [0xA0, 0x2002, 0x2003, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000]))
_WS_RE = re.compile("[ \t\r\f\v" + _SPACES + "]+")
_ZW_RE = re.compile("[" + "".join(map(chr, [0x200B, 0x200C, 0x200D, 0xFEFF, 0xAD])) + "]")
_QUOTES = str.maketrans({0x2019: "'", 0x2018: "'", 0x201C: '"', 0x201D: '"', 0x2013: "-", 0x2014: "-"})
_PAGENUM_RE = re.compile(r"(page\s*)?-?\s*\d{1,3}\s*-?", re.I)
_FOOTER_RE = re.compile(r"[^|]{2,80}\|[^|]{0,40}Form\s+10-[KQ]\s*\|\s*\d{1,3}", re.I)
_RUNHDR_RE = re.compile(r"(part\s+[ivx]+(\s*,\s*[ivx]+)*\s*)?(item\s+\d{1,2}[a-c]?(\s*,\s*\d{1,2}[a-c]?)*)?", re.I)


def _local(tag) -> str:
    if not isinstance(tag, str):
        return ""  # comments / processing instructions
    t = tag.lower()
    return t.split("}", 1)[1] if "}" in t else t


def _walk(el, out: list) -> None:
    tag = _local(el.tag)
    if tag == "table":
        out.append("\n" + _table_to_text(el) + "\n")
    else:
        block = tag in _BLOCK
        if block:
            out.append("\n")
        if el.text and isinstance(el.tag, str):
            out.append(el.text)
        for ch in el:
            _walk(ch, out)
        if block:
            out.append("\n")
    if el.tail:
        out.append(el.tail)


def _cell_text(td) -> str:
    parts: list = [td.text] if td.text else []
    for ch in td:  # <br>/<div>/<p> inside a cell -> spaces ("June 27,<br>2026")
        _walk(ch, parts)
    return _WS_RE.sub(" ", _ZW_RE.sub("", "".join(parts)).replace("\n", " ")).strip()


def _table_to_text(tbl) -> str:
    """Render a table as 'a | b | c' rows; merge EDGAR's split '$', ')' and '%' cells."""
    lines = []
    for tr in tbl.iter():
        if _local(tr.tag) != "tr":
            continue
        merged: list = []
        for td in tr:
            if _local(td.tag) not in ("td", "th"):
                continue
            c = _cell_text(td)
            if not c:
                continue
            if merged and (c in (")", "%", ")%", "%)") or c.startswith(")")):
                merged[-1] += c
            elif merged and merged[-1] in ("$", "(", "$(", "($"):
                merged[-1] += c
            else:
                merged.append(c)
        if merged:
            lines.append(" | ".join(merged))
    return "\n".join(lines)


def edgar_html_to_text(raw, drop_page_furniture: bool = True) -> str:
    """Convert an EDGAR .htm (incl. inline XBRL, Workiva/DFIN/Toppan output) to plain text.

    - drops <ix:header> hidden XBRL facts, display:none blocks, <script>/<style>/<head>
    - tables -> one row per line, cells joined by ' | ' (with '$', ')' and '%' merged)
    - normalises unicode spaces/quotes, removes page numbers, 'X | Form 10-Q | 12'
      footers and MSFT-style running headers ('Item 1A', 'PART II') that repeat >= 3x
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw = re.sub(r"^\s*<\?xml[^>]*\?>", "", raw)  # lxml refuses str input with an encoding decl
    m = re.search(r"<html", raw[:5000], re.I)  # non-iXBRL exhibits keep an SGML <DOCUMENT><TYPE>.. wrapper
    if m and m.start() > 0:
        raw = raw[m.start():]
    if not re.search(r"<(html|body|div|p|table)\b", raw[:20000], re.I):  # plain-text (.txt) filing
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"<[^>]+>", "", raw)).strip()
    doc = LH.fromstring(raw)
    to_drop = []
    for el in doc.iter():
        if not isinstance(el.tag, str):
            to_drop.append(el)  # comments / PIs
            continue
        lt = el.tag.lower()
        if lt in _DROP_TAGS or lt == "ix:header" or (lt.endswith("}header") and "inlinexbrl" in lt):
            to_drop.append(el)
        else:
            st = el.get("style")
            if st and _HIDDEN_RE.search(st):
                to_drop.append(el)
    for el in to_drop:
        parent = el.getparent()
        if parent is None:
            continue
        if el.tail:  # keep the text that follows the removed element
            prev = el.getprevious()
            if prev is not None:
                prev.tail = (prev.tail or "") + el.tail
            else:
                parent.text = (parent.text or "") + el.tail
        parent.remove(el)

    out: list = []
    _walk(doc, out)
    text = _ZW_RE.sub("", "".join(out)).translate(_QUOTES)
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    if drop_page_furniture:
        cnt = Counter(lines)
        keep = []
        for ln in lines:
            if ln and (_PAGENUM_RE.fullmatch(ln) or _FOOTER_RE.fullmatch(ln)):
                continue
            if ln and _RUNHDR_RE.fullmatch(ln) and cnt[ln] >= 3:
                continue
            if ln.lower() in ("table of contents", "index"):
                continue
            keep.append(ln)
        lines = keep
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --------------------------------------------------------------------------- sections
_SEP = r"[\s|.:\-]*"
_SECTION_PATTERNS = {
    # (start regexes, end regexes); all anchored at line start (re.M)
    "10-K": {
        "risk_factors": ([rf"^item{_SEP}1a{_SEP}risk\s+factors"],
                         [rf"^item{_SEP}1b\b", rf"^item{_SEP}1c\b", rf"^item{_SEP}2{_SEP}propert"]),
        "mdna": ([rf"^item{_SEP}7{_SEP}management'?s\s+discussion"],
                 [rf"^item{_SEP}7a\b", rf"^item{_SEP}8{_SEP}financial\s+statements"]),
    },
    "10-Q": {
        "mdna": ([rf"^item{_SEP}2{_SEP}management'?s\s+discussion"],
                 [rf"^item{_SEP}3{_SEP}quantitative", rf"^item{_SEP}4{_SEP}controls"]),
        "risk_factors": ([rf"^item{_SEP}1a{_SEP}risk\s+factors"],
                         [rf"^item{_SEP}2{_SEP}unregistered", rf"^item{_SEP}3{_SEP}defaults",
                          rf"^item{_SEP}4{_SEP}mine", rf"^item{_SEP}5{_SEP}other\s+information",
                          rf"^item{_SEP}6{_SEP}exhibits"]),
    },
}


def extract_section(text: str, form: str, section: str, max_chars: int | None = 60_000) -> str:
    """Extract section ('risk_factors' | 'mdna') from edgar_html_to_text() output.

    form: '10-K' / '10-Q' (amendments like '10-K/A' map to the base form).
    For every heading match it takes the text up to the next end-heading and keeps the
    LONGEST span -> skips table-of-contents rows and in-text cross references.
    Returns '' if not found (e.g. 10-K/A Part III-only amendments).
    """
    base = "10-K" if form.upper().startswith("10-K") else "10-Q"
    starts, ends = _SECTION_PATTERNS[base][section]
    flags = re.I | re.M
    s_pos = sorted({m.start() for p in starts for m in re.finditer(p, text, flags)})
    e_pos = sorted({m.start() for p in ends for m in re.finditer(p, text, flags)})
    best = (0, 0)
    for s in s_pos:
        nxt = [e for e in e_pos if e > s + 50]
        e = nxt[0] if nxt else min(len(text), s + 150_000)  # no end heading -> cap
        if e - s > best[1] - best[0]:
            best = (s, e)
    if best == (0, 0):
        return ""
    sec = text[best[0]:best[1]].strip()
    if len(sec) < 3000:  # bank-style filers (JPM/BAC/C): section is only a sub-TOC 'Title | page'
        sec = _expand_sub_toc(text, sec, best[1], ends) or sec
    tail_lines = sec.split("\n")
    while tail_lines and (not tail_lines[-1].strip() or _RUNHDR_RE.fullmatch(tail_lines[-1].strip())):
        tail_lines.pop()  # drop trailing running header like 'Item 3, 4'
    sec = "\n".join(tail_lines)
    if max_chars and len(sec) > max_chars:
        sec = truncate_smart(sec, max_chars)
    return sec


def _expand_sub_toc(text: str, sec: str, after: int, ends: list) -> str:
    entries = [e.strip() for e in re.findall(r"^(.{3,120}?) \| \d{1,3}$", sec, re.M)]
    if len(entries) < 3:
        return ""
    def _find(title: str, start: int):
        m = re.compile(r"^" + re.escape(title) + r"\b.{0,40}$", re.I | re.M).search(text, start)
        return m.start() if m else None
    s = next((p for p in (_find(t, after) for t in entries[:3]) if p is not None), None)
    if s is None:
        return ""
    last = next((p for p in (_find(t, s + 1) for t in reversed(entries)) if p is not None), None) or s
    stops = list(ends) + [r"^consolidated (statements? of|balance sheets?)", r"^report of independent",
                          r"^.{0,60}\bconsolidated statements of (income|operations)\b"]
    e_m = [m.start() for p in stops for m in re.finditer(p, text[last + 50:], re.I | re.M)]
    e_m = [x + 50 for x in e_m]
    e = last + min(min(e_m) if e_m else 12_000, 12_000)  # last sub-section is short (e.g. FLS)
    return text[s:e].strip()


def truncate_smart(s: str, max_chars: int, head_frac: float = 0.7) -> str:
    """Keep head (70%) + tail (30%) cut on line boundaries, with an explicit marker."""
    head_n = int(max_chars * head_frac)
    tail_n = max_chars - head_n
    head = s[:head_n]
    cut = head.rfind("\n")
    head = head[:cut] if cut > head_n * 0.5 else head
    tail = s[-tail_n:]
    cut = tail.find("\n")
    tail = tail[cut + 1:] if 0 <= cut < tail_n * 0.5 else tail
    return f"{head}\n\n[... {len(s) - len(head) - len(tail):,} chars truncated ...]\n\n{tail}"


def risk_factor_headlines(risk_text: str, max_items: int = 60) -> list:
    """Risk-factor titles = standalone one-sentence lines (40-400 chars) ending in '.'.
    Good as a compact 'key risk' list to feed the LLM instead of the full section."""
    out = []
    for ln in risk_text.split("\n"):
        ln = ln.strip().lstrip(chr(0x2022) + chr(0xB7) + "o- ").strip()
        if (40 <= len(ln) <= 400 and ln.endswith(".") and ". " not in ln and "|" not in ln
                and not re.match(r"(other than|this section|the following|for all of these|there have been)", ln, re.I)):
            out.append(ln)
    return out[:max_items]


_KEYWORDS = ("expect", "outlook", "guidance", "anticipate", "demand", "supply", "constrain", "margin",
             "headwind", "tailwind", "decline", "decrease", "increase", "growth", "tariff", "pricing",
             "backlog", "impairment", "restructur", "litigation", "regulat", "export", "china",
             "liquidity", "repurchase", "capital expenditure", "uncertain", "risk")


def select_paragraphs(text: str, budget_chars: int = 12_000, keywords=_KEYWORDS,
                      keep_first: int = 1, min_len: int = 120) -> str:
    """Budgeted digest for LLM input: keep the first `keep_first` paragraphs, then the
    highest keyword-density paragraphs (tables skipped), emitted in original order."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    scored = []
    for i, p in enumerate(paras):
        if len(p) < min_len or p.count("|") > 3:
            continue
        if re.search(r"forward-looking statements|should be read in conjunction|incorporated by reference", p, re.I):
            continue  # boilerplate
        low = p.lower()
        score = sum(low.count(k) for k in keywords) / (len(p) ** 0.5)
        scored.append((i < keep_first, score, i, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    chosen, used = [], 0
    for _, _, i, p in scored:
        if used + len(p) > budget_chars:
            continue
        chosen.append((i, p))
        used += len(p) + 2
    return "\n\n".join(p for _, p in sorted(chosen))
