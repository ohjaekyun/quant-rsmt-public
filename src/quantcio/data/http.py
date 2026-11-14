"""Shared HTTP session: retries with backoff, browser-like UA, per-host politeness delay."""

from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

_MIN_INTERVAL = {  # seconds between requests per host
    "data.sec.gov": 0.15,
    "www.sec.gov": 0.15,
    "finance.naver.com": 0.3,
    "m.stock.naver.com": 0.3,
    "news.google.com": 0.5,
    "opendart.fss.or.kr": 0.2,
    "dart.fss.or.kr": 0.5,
    "fchart.stock.naver.com": 0.1,
    "api.stock.naver.com": 0.3,
    "n.news.naver.com": 0.3,
    "finance.daum.net": 0.2,
    "navercomp.wisereport.co.kr": 0.3,
    "kind.krx.co.kr": 0.5,
    "www.fool.com": 0.5,
}
_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _throttle(url: str) -> None:
    host = urlparse(url).netloc
    gap = _MIN_INTERVAL.get(host, 0.0)
    if not gap:
        return
    with _lock:
        now = time.monotonic()
        wait = _last_call.get(host, 0.0) + gap - now
        _last_call[host] = max(now, _last_call.get(host, 0.0) + gap)
    if wait > 0:
        time.sleep(wait)


class PoliteSession(requests.Session):
    def __init__(self, user_agent: str = BROWSER_UA, timeout: float = 15.0):
        super().__init__()
        self.headers.update({"User-Agent": user_agent, "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8"})
        retry = Retry(
            total=3,
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_maxsize=16)
        self.mount("https://", adapter)
        self.mount("http://", adapter)
        self._timeout = timeout

    def request(self, method, url, *args, **kwargs):  # type: ignore[override]
        kwargs.setdefault("timeout", self._timeout)
        _throttle(url)
        return super().request(method, url, *args, **kwargs)
