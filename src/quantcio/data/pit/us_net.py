"""Network access + on-disk raw cache for the US point-in-time (PIT) builders.

Every download is cached under ``settings.cache_path / "pit" / "us" / "<source>_v<k>/"`` (gitignored), so a warm
rebuild needs no network. Hosts and pacing (verified live 2026-09-25, docs/sources/pit/us-universe.json):

- SEC (www.sec.gov, data.sec.gov): <= 5 req/s here, on top of the process-wide SEC budget of ``SecSession``; a
  descriptive ``settings.data.sec_user_agent`` is mandatory (a clear error is raised when it is empty).
- Yahoo v8 chart (query2): 4 req/s with the short UA (a full Chrome UA gets HTTP 429 from a TLS fingerprint check).
- Wikipedia / Wikidata: 0.3 s spacing, HTTP 429 honours ``retry-after`` (bursts are throttled even at a few calls).
- GitHub (fja05680/sp500): 0.2 s spacing. Wayback Machine: 1 req/s.

A failed refresh falls back to the stale cached copy and records a Korean note (never silently); only a cache miss
with a failed download raises.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import requests

from quantcio.config import Settings
from quantcio.data.us import UA_YAHOO_SHORT, SecSession

log = logging.getLogger(__name__)

SOURCE_DIRS = {  # source -> versioned cache directory (bump the version when the stored format changes)
    "sp500": "sp500_v1",
    "sec": "sec_v1",
    "ftd": "ftd_v1",
    "yahoo": "yahoo_v1",
    "xbrl": "xbrl_v1",
    "build": "build_v1",
}
_GAP = {"sec": 0.2, "yahoo": 0.25, "wiki": 0.3, "archive": 1.0, "github": 0.2}
_RETRY_STATUS = (429, 500, 502, 503, 504)


class SecUserAgentError(RuntimeError):
    """``settings.data.sec_user_agent`` is empty: SEC EDGAR refuses anonymous clients."""


def require_sec_user_agent(settings: Settings) -> str:
    ua = (settings.data.sec_user_agent or "").strip()
    if not ua:
        raise SecUserAgentError(
            'SEC_USER_AGENT 미설정 — 미국 PIT 유니버스/재무는 SEC EDGAR가 필수입니다. .env에 SEC_USER_AGENT="회사명 '
            '연락처@도메인" 형식으로 설정하세요 (예: "QuantCIO Research research@example.com")'
        )
    return ua


class _Gap:
    """Thread-safe minimum spacing between calls of one kind."""

    def __init__(self, seconds: float):
        self.seconds = seconds
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.seconds
        if wait > 0:
            time.sleep(wait)


_GAPS = {k: _Gap(v) for k, v in _GAP.items()}


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class PitNet:
    """HTTP (polite, retrying) + raw file cache shared by the US PIT membership / CIK / price / XBRL steps.

    Tests subclass it and override :meth:`http_get` to serve fixtures (no network).
    """

    def __init__(self, settings: Settings, root: Path | None = None):
        self.settings = settings
        self.root = Path(root) if root is not None else settings.cache_path / "pit" / "us"
        self._sec: SecSession | None = None
        self._web = requests.Session()
        self._lock = threading.Lock()
        self.notes: list[str] = []
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ paths

    def path(self, source: str, *parts: str) -> Path:
        return self.root.joinpath(SOURCE_DIRS[source], *parts)

    # ------------------------------------------------------------------ http

    def sec_user_agent(self) -> str:
        return require_sec_user_agent(self.settings)

    def contact_user_agent(self) -> str:
        """Descriptive UA for Wikimedia/GitHub/Wayback (their API etiquette asks for a contact)."""
        ua = (self.settings.data.sec_user_agent or "").strip()
        return f"QuantCIO-PIT/1.0 ({ua or 'no contact configured'}) python-requests"

    def _sec_session(self) -> SecSession:
        ua = self.sec_user_agent()
        with self._lock:
            if self._sec is None or self._sec.headers.get("User-Agent") != ua:
                self._sec = SecSession(ua)
            return self._sec

    def http_get(
        self,
        url: str,
        kind: str,
        params: dict | None = None,
        headers: dict | None = None,
        ok: tuple[int, ...] = (200,),
        tries: int = 5,
    ) -> requests.Response:
        """GET with per-kind pacing and retries on 429/5xx (``retry-after`` honoured). Status not in ``ok`` raises."""
        session: requests.Session = self._sec_session() if kind == "sec" else self._web
        hdrs = dict(headers or {})
        if kind in ("wiki", "github", "archive"):
            hdrs.setdefault("User-Agent", self.contact_user_agent())
        elif kind == "yahoo":
            hdrs.setdefault("User-Agent", UA_YAHOO_SHORT)
        last: BaseException | None = None
        for i in range(tries):
            _GAPS[kind].wait()
            self.stats[f"http_{kind}"] += 1
            try:
                r = session.get(url, params=params, headers=hdrs or None, timeout=60)
            except requests.RequestException as exc:
                last = exc
                time.sleep(min(2**i, 20))
                continue
            if r.status_code in _RETRY_STATUS and r.status_code not in ok:
                last = requests.HTTPError(f"HTTP {r.status_code} {url}", response=r)
                try:
                    wait = float(r.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0.0
                time.sleep(min(wait or 3 * 2**i, 60))
                continue
            if r.status_code == 403 and kind == "sec":
                raise SecUserAgentError(f"SEC EDGAR 403 (User-Agent 거부) — SEC_USER_AGENT 형식 확인: {url}")
            if r.status_code not in ok:
                r.raise_for_status()
                raise requests.HTTPError(f"HTTP {r.status_code} {url}", response=r)
            return r
        raise RuntimeError(f"{url}: {tries}회 재시도 실패 ({type(last).__name__}: {last})")

    # ------------------------------------------------------------------ cached files

    def cached_bytes(
        self,
        source: str,
        rel: str,
        url: str,
        kind: str,
        max_age_h: float | None,
        params: dict | None = None,
        headers: dict | None = None,
    ) -> bytes:
        """Bytes of ``url`` cached at ``<source dir>/<rel>`` (``.gz`` names are stored gzip-compressed).

        ``max_age_h=None`` = cached forever. A failed refresh of a stale file returns the stale copy with a note.
        """
        p = self.path(source, rel)
        fresh = p.exists() and (max_age_h is None or time.time() - p.stat().st_mtime < max_age_h * 3600)
        if fresh:
            self.stats["cache_hit"] += 1
            return self._read(p)
        try:
            body = self.http_get(url, kind, params=params, headers=headers).content
        except SecUserAgentError:
            raise
        except Exception as exc:  # noqa: BLE001 - stale copy is better than nothing, and it is reported
            if p.exists():
                age_d = (time.time() - p.stat().st_mtime) / 86400
                self.note(f"{rel}: 새로 받기 실패({type(exc).__name__}) — {age_d:.1f}일 전 캐시 사용")
                return self._read(p)
            raise
        atomic_write(p, gzip.compress(body, 6) if rel.endswith(".gz") else body)
        return body

    @staticmethod
    def _read(p: Path) -> bytes:
        raw = p.read_bytes()
        return gzip.decompress(raw) if p.name.endswith(".gz") else raw

    def read_pickle(self, source: str, rel: str) -> Any | None:
        p = self.path(source, rel)
        if not p.exists():
            return None
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache entry is rebuilt
            log.warning("pit cache %s unreadable (%s) - rebuilding", p, exc)
            return None

    def write_pickle(self, source: str, rel: str, value: Any) -> None:
        atomic_write(self.path(source, rel), pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def read_json(self, source: str, rel: str) -> Any | None:
        p = self.path(source, rel)
        if not p.exists():
            return None
        try:
            return json.loads(self._read(p))
        except (OSError, ValueError):
            return None

    def write_json(self, source: str, rel: str, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False, default=str).encode()
        atomic_write(self.path(source, rel), gzip.compress(body, 6) if rel.endswith(".gz") else body)

    # ------------------------------------------------------------------ notes

    def note(self, msg: str) -> None:
        with self._lock:
            if msg not in self.notes:
                self.notes.append(msg)
        log.warning("pit-us: %s", msg)
