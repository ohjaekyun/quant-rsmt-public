"""PDF rendering watchdog: `fit_to_pages` in a child process with a hard timeout, so a report can never hang.

Live incident 2026-09-25 23:4x: Playwright's node driver crashed while transferring the PDF ("SyntaxError:
Unterminated string in JSON at position 65536" in coreBundle.js) and the in-process sync API then waited forever -
the report process hung for 20+ minutes. Rendering now runs in a separate process:

Parent side, ``fit_to_pages_isolated`` (the production path of ``pipeline.run_report`` and ``quantcio doctor``):

1. ``payload.pkl`` in a private temp dir: the PortfolioReport, the Settings (secrets blanked), the chart SVGs, the
   output path and the page limit. Pickled, not JSON: a JSON round trip turns NaN metrics (e.g. a CV bound) into
   null, which the float fields reject.
2. ``python -m quantcio.report.worker <payload.pkl> <result.pkl>`` in its own session; stdout / stderr go to
   ``worker.log`` and every compaction attempt (level, scale, pages) is streamed to ``attempts.jsonl``.
3. The worker may run for ``RENDER_TIMEOUT_S`` (180 s; a normal 6-name report takes 5-15 s). A result written but a
   worker that does not exit within ``EXIT_GRACE_S`` (e.g. stuck closing Chrome) -> the result is used and the
   process tree is killed.
4. Timeout, crash (no result / non-zero exit) or an exception in the worker -> the whole process tree is killed
   (the worker, Playwright's node driver and the Chrome process group Playwright detaches) and the render is
   retried ONCE (``RENDER_ATTEMPTS = 2``).
5. Both attempts failed -> HTML fallback written in THIS process (Jinja only, no browser) at the compaction level
   the worker had chosen (its first attempt that fitted the page limit) or ``MAX_COMPACT_LEVEL``, with a page-1
   note; ``FitResult(pdf_path=None, pages=0, fallback_html=True)`` and a Korean / English warning that
   ``run_report`` puts in ``ReportArtifacts.warnings``. A retry that succeeded is reported as a warning too.

Worker side, ``main``: rebuilds the report, starts ONE Chrome, runs ``fit_to_pages(renderer=...)``, writes the
result atomically BEFORE closing Chrome, then exits (exit code 1 with ``{"ok": False, "error": ...}`` on any
exception).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pickle
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from quantcio.config import Settings
from quantcio.i18n import tr
from quantcio.models import PortfolioReport

log = logging.getLogger(__name__)

RENDER_TIMEOUT_S = 180.0  # per attempt; a normal report renders in 5-15 s (incl. ~2 s worker start-up)
RENDER_ATTEMPTS = 2  # the first try + ONE retry
EXIT_GRACE_S = 10.0  # result on disk but the worker still running -> use the result, kill the tree
POLL_S = 0.2
KILL_WAIT_S = 10.0
LOG_TAIL_LINES = 20
PAYLOAD_VERSION = 1


# ============================================================================ process tree


def _process_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, pgid) of every process (POSIX ``ps``)."""
    out = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,pgid="], capture_output=True, text=True, timeout=10, check=True
    ).stdout
    table: dict[int, tuple[int, int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.lstrip("-").isdigit() for p in parts):
            table[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return table


def descendants(pid: int, table: dict[int, tuple[int, int]]) -> list[int]:
    """Every descendant pid of ``pid`` in ``table`` (children first, breadth-first)."""
    children: dict[int, list[int]] = {}
    for child, (parent, _pgid) in table.items():
        children.setdefault(parent, []).append(child)
    out: list[int] = []
    queue = list(children.get(pid, []))
    while queue:
        p = queue.pop(0)
        if p in out or p == pid:
            continue
        out.append(p)
        queue.extend(children.get(p, []))
    return out


def kill_process_tree(proc: subprocess.Popen) -> list[int]:
    """SIGKILL ``proc``, all its descendants and their process groups (Playwright starts Chrome detached, in a
    group of its own); returns the pids signalled. The caller's own process group is never signalled."""
    if os.name == "nt":  # pragma: no cover - POSIX is the supported platform
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=KILL_WAIT_S)
        return [proc.pid]
    pids = [proc.pid]
    groups = {proc.pid}  # the worker leads its own session / group (start_new_session=True)
    try:
        table = _process_table()
        pids += descendants(proc.pid, table)
        groups |= {table[p][1] for p in pids if p in table}
    except Exception as exc:  # noqa: BLE001 - still kill the worker's own group
        log.warning("PDF worker: process table unavailable (%s) - killing the worker's process group only", exc)
    groups.discard(os.getpgid(0))
    for g in sorted(groups):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(g, signal.SIGKILL)
    for p in pids:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, signal.SIGKILL)
    try:
        proc.wait(timeout=KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        log.error("PDF worker %d still alive %.0f s after SIGKILL", proc.pid, KILL_WAIT_S)
    return pids


# ============================================================================ parent


def _safe_settings(settings: Settings) -> Settings:
    """A deep copy without credentials (the renderer never needs them; the payload file lives briefly on disk)."""
    s = settings.model_copy(deep=True)
    with contextlib.suppress(Exception):
        s.data.dart_api_key = ""
    return s


def _worker_cmd() -> list[str]:
    return [sys.executable, "-m", "quantcio.report.worker"]


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2])  # .../src: the worker imports this very quantcio
    env["PYTHONPATH"] = os.pathsep.join([src, *(p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p)])
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _tail(path: Path, n: int = LOG_TAIL_LINES) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [ln for ln in lines if ln.strip()][-n:]


def _read_attempts(path: Path) -> list[tuple[int, float, int]]:
    out: list[tuple[int, float, int]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            level, scale, pages = json.loads(line)
            out.append((int(level), float(scale), int(pages)))
        except (ValueError, TypeError):
            continue  # a line cut by the kill
    return out


def _fit_from_dict(d: dict[str, Any]) -> Any:
    from quantcio.report.render import FitResult

    return FitResult(
        pdf_path=Path(d["pdf_path"]) if d.get("pdf_path") else None,
        html_path=Path(d["html_path"]) if d.get("html_path") else None,
        pages=int(d["pages"]),
        compact_level=int(d["compact_level"]),
        scale=float(d["scale"]),
        fits=bool(d["fits"]),
        fallback_html=bool(d["fallback_html"]),
        attempts=[(int(a), float(b), int(c)) for a, b, c in d.get("attempts", [])],
        warnings=[str(w) for w in d.get("warnings", [])],
        extended=bool(d.get("extended", False)),
    )


def _run_once(
    cmd: Sequence[str], result_path: Path, log_path: Path, timeout: float, settings: Settings
) -> tuple[Any | None, str]:
    """One worker run -> (FitResult, "") or (None, reason in the report language)."""
    t0 = time.monotonic()
    with log_path.open("ab") as logf:
        proc = subprocess.Popen(
            [*cmd],
            stdin=subprocess.DEVNULL,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=_worker_env(),
            close_fds=True,
            start_new_session=os.name != "nt",
        )
    killed = False
    try:
        seen_result: float | None = None
        while True:
            try:
                proc.wait(timeout=POLL_S)
                break
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            if seen_result is None and result_path.exists():
                seen_result = now
            if seen_result is not None and now - seen_result > EXIT_GRACE_S:
                log.warning(
                    "PDF worker %d wrote its result but did not exit within %.0f s - killing its process tree",
                    proc.pid,
                    EXIT_GRACE_S,
                )
                kill_process_tree(proc)
                killed = True
                break
            if seen_result is None and now - t0 > timeout:
                pids = kill_process_tree(proc)
                log.error(
                    "PDF worker %d timed out after %.0f s - process tree killed (%d processes)",
                    proc.pid,
                    timeout,
                    len(pids),
                )
                return None, tr(settings, f"시간 초과 {timeout:.0f}초", f"timed out after {timeout:.0f} s")
    finally:
        if proc.poll() is None:  # interrupted (KeyboardInterrupt ...): never leave a renderer behind
            kill_process_tree(proc)
            killed = True
    rc = proc.returncode
    res: dict[str, Any] | None = None
    if result_path.exists():
        try:
            res = pickle.loads(result_path.read_bytes())
        except Exception as exc:  # noqa: BLE001 - reported below as a crash
            log.warning("PDF worker result unreadable: %s", exc)
    if isinstance(res, dict) and res.get("ok"):
        fit = _fit_from_dict(res["fit"])
        log.info("PDF worker finished in %.1fs (exit %s%s)", time.monotonic() - t0, rc, ", killed" if killed else "")
        return fit, ""
    if isinstance(res, dict) and res.get("error"):
        err = " ".join(str(res["error"]).split())[:200]
        log.warning("PDF worker error: %s\n%s", err, res.get("traceback", ""))
        return None, tr(settings, f"렌더러 오류: {err}", f"renderer error: {err}")
    last = (_tail(log_path, 1) or [""])[0][:160]
    detail = f": {last}" if last else ""
    return None, tr(settings, f"비정상 종료(코드 {rc}){detail}", f"crashed (exit code {rc}){detail}")


def fit_to_pages_isolated(
    report: PortfolioReport,
    settings: Settings,
    out_path: Path | str,
    *,
    charts: dict[str, str] | None = None,
    max_pages: int | None = None,
    timeout: float = RENDER_TIMEOUT_S,
    attempts: int = RENDER_ATTEMPTS,
    worker_cmd: Sequence[str] | None = None,
) -> Any:
    """`render.fit_to_pages` in a watchdog subprocess (see the module docstring); never hangs.

    Returns the worker's FitResult, or after ``attempts`` failed runs the HTML fallback FitResult
    (``pdf_path=None``, ``pages=0``, ``fallback_html=True``, one warning naming every failure).
    ``worker_cmd`` replaces ``python -m quantcio.report.worker`` (tests); the payload and result paths are appended.

    Raises:
        ValueError: max_pages < 1 or attempts < 1 (checked before any process starts).
    """
    from quantcio.report.render import MAX_COMPACT_LEVEL, FitResult, render_html, save_html_fallback

    limit = int(max_pages if max_pages is not None else settings.report.max_pages)
    if limit < 1:
        raise ValueError(f"max_pages must be >= 1, got {limit}")
    if int(attempts) < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="quantcio_render_"))
    reasons: list[str] = []
    try:
        payload = tmp / "payload.pkl"
        attempts_path = tmp / "attempts.jsonl"
        log_path = tmp / "worker.log"
        payload.write_bytes(
            pickle.dumps(
                {
                    "version": PAYLOAD_VERSION,
                    "report": report,
                    "settings": _safe_settings(settings),
                    "charts": dict(charts) if charts is not None else None,
                    "out_path": str(out_path),
                    "max_pages": limit,
                    "attempts_path": str(attempts_path),
                }
            )
        )
        base = list(worker_cmd) if worker_cmd is not None else _worker_cmd()
        for i in range(1, int(attempts) + 1):
            result_path = tmp / f"result_{i}.pkl"
            fit, reason = _run_once([*base, str(payload), str(result_path)], result_path, log_path, timeout, settings)
            if fit is not None:
                if reasons:
                    fit.warnings.insert(
                        0,
                        tr(
                            settings,
                            f"PDF 렌더러 실패 {len(reasons)}회({'; '.join(reasons)}) → 재시도 {i}회차에 성공",
                            f"PDF renderer failed {len(reasons)}x ({'; '.join(reasons)}) → succeeded on attempt {i}",
                        ),
                    )
                return fit
            reasons.append(reason)
            log.warning("PDF worker attempt %d/%d failed: %s", i, attempts, reason)
            for line in _tail(log_path):
                log.warning("PDF worker log: %s", line)

        # every attempt failed: HTML in this process (no browser), at the level the worker had chosen
        tried = _read_attempts(attempts_path)
        level = next((lv for lv, _s, n in tried if n <= limit), MAX_COMPACT_LEVEL)
        why = "; ".join(reasons)
        note = tr(
            settings,
            f"PDF 생성 실패(렌더러 {len(reasons)}회: {why}) — HTML 대체 출력입니다. 브라우저에서 열어 인쇄 → PDF로 "
            "저장하세요",
            f"PDF not generated (renderer failed {len(reasons)}x: {why}) — this is the HTML fallback; open it in a "
            "browser and print to PDF",
        )
        html = render_html(report, settings, level, charts, overflow_note=note)
        p = save_html_fallback(html, out_path)
        with contextlib.suppress(OSError):
            if out_path.exists() and out_path.stat().st_mtime >= t_start:
                out_path.unlink()  # a PDF from a killed attempt is unverified: the fallback is the output
                log.warning("removed the unverified PDF %s written by a killed renderer", out_path)
        msg = tr(
            settings,
            f"PDF 렌더러 {len(reasons)}회 실패({why}) — HTML 대체 파일 저장: {p} (압축 레벨 {level}; 브라우저에서 "
            "인쇄 → PDF로 저장)",
            f"PDF renderer failed {len(reasons)}x ({why}) - HTML fallback saved: {p} (compaction level {level}; "
            "open it in a browser and print to PDF)",
        )
        log.error(msg)
        return FitResult(
            pdf_path=None,
            html_path=p,
            pages=0,
            compact_level=level,
            scale=0.0,
            fits=False,
            fallback_html=True,
            attempts=tried,
            warnings=[msg],
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================ worker


def _write_result(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(value))
    os.replace(tmp, path)  # atomic: the parent treats existence as "complete"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m quantcio.report.worker <payload.pkl> <result.pkl>", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    payload_path, result_path = Path(args[0]), Path(args[1])
    renderer = None
    try:
        payload = pickle.loads(payload_path.read_bytes())
        if payload.get("version") != PAYLOAD_VERSION:
            raise ValueError(f"payload version {payload.get('version')!r} != {PAYLOAD_VERSION}")
        from quantcio.report.render import NoBrowserError, PdfRenderer, fit_to_pages

        report = payload["report"]
        if not isinstance(report, PortfolioReport):
            raise TypeError(f"payload report is {type(report).__name__}, not PortfolioReport")
        settings = payload["settings"]
        with open(payload["attempts_path"], "a", encoding="utf-8") as attempts_f:

            def on_attempt(level: int, scale: float, pages: int) -> None:
                attempts_f.write(json.dumps([level, scale, pages]) + "\n")
                attempts_f.flush()

            try:
                renderer = PdfRenderer().__enter__()
            except NoBrowserError:
                renderer = None  # fit_to_pages tries the launch itself and writes its HTML fallback
            res = fit_to_pages(
                report,
                settings,
                Path(payload["out_path"]),
                charts=payload["charts"],
                max_pages=payload["max_pages"],
                renderer=renderer,
                on_attempt=on_attempt,
            )
        fit = {
            "pdf_path": str(res.pdf_path) if res.pdf_path else None,
            "html_path": str(res.html_path) if res.html_path else None,
            "pages": res.pages,
            "compact_level": res.compact_level,
            "scale": res.scale,
            "fits": res.fits,
            "fallback_html": res.fallback_html,
            "attempts": [list(a) for a in res.attempts],
            "warnings": list(res.warnings),
            "extended": res.extended,
        }
        _write_result(result_path, {"ok": True, "fit": fit})
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which retries / falls back
        with contextlib.suppress(Exception):
            _write_result(
                result_path,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
            )
        traceback.print_exc()
        return 1
    finally:
        # Chrome closes AFTER the result is on disk: a hang here only costs the parent's EXIT_GRACE_S
        if renderer is not None:
            with contextlib.suppress(Exception):
                renderer.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
