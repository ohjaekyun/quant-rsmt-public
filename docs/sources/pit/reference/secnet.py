import gzip, hashlib, json, threading, time
from pathlib import Path
import requests
UA = "QuantCIO Research research@example.com"
CACHE = Path(__file__).resolve().parent / "cache"; CACHE.mkdir(exist_ok=True)
_MIN_GAP = 0.25  # <= 4 req/s (SEC limit is 10/s per IP, shared with other jobs)
_last = [0.0]; _lock = threading.Lock()
_sess = requests.Session(); _sess.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})

def _throttle():
    with _lock:
        wait = _last[0] + _MIN_GAP - time.monotonic()
        if wait > 0: time.sleep(wait)
        _last[0] = time.monotonic()

def get(url: str, *, cache: bool = True, retries: int = 4) -> bytes:
    p = CACHE / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".gz")
    if cache and p.exists():
        return gzip.decompress(p.read_bytes())
    for i in range(retries):
        _throttle()
        r = _sess.get(url, timeout=60, stream=True)
        raw = r.raw.read(decode_content=False)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** i); continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} {url}: {raw[:200]!r}")
        body = gzip.decompress(raw) if r.headers.get("Content-Encoding", "") == "gzip" else raw
        if cache: p.write_bytes(gzip.compress(body, 6))
        return body
    raise RuntimeError(f"gave up after {retries} retries: {url}")

def companyfacts(cik): return json.loads(get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json"))
def submissions(cik): return json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
def get_json(url: str, **kw) -> dict: return json.loads(get(url, **kw))  # FIX: used by the EDGAR autocomplete snippet
