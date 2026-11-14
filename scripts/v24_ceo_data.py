"""CEO and strategy data (docs/MODELING.md 63, pre-declared 2026-10-16). Research only.

    .venv/bin/python scripts/v24_ceo_data.py fetch --what all      # or exctv, business, wikidata

KR: OpenDART executive rosters (exctvSttus; annual reports 2015-2025 + half-year reports 2016-2026) and the
"II. 사업의 내용" section of every annual report (document.xml; only the section text is kept). US: Wikidata CEO terms
(P169 with start / end) and schools / degrees (P69 with P512 / P812). Resumable; personal data stays in the cache.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402
import v13_data as v13  # noqa: E402
import v16_data as v16  # noqa: E402
import v22_tone as tone  # noqa: E402

say = v13.say
DART_DOC = "https://opendart.fss.or.kr/api/document.xml"
WIKIDATA = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "Mozilla/5.0 (compatible; research bot)", "Accept": "application/sparql-results+json"}
_SEC = re.compile(r"<TITLE[^>]*>\s*II\.\s*사업의\s*내용\s*</TITLE>(.*?)<TITLE[^>]*>\s*III\.", re.S)


def root(settings) -> Path:
    return kd.root(settings, "v24")


def _codes(settings, cache) -> list[str]:
    return sorted(v16.universe("KR", settings, cache)["entity"].astype(str))


def fetch_exctv(settings, cache) -> dict:
    k = kd.key("DART_API_KEY")
    cc = v16.kr_corp_codes(settings)
    d = root(settings) / "exctv"
    d.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today()
    reports = [(y, "11011") for y in range(2015, today.year)] + [(y, "11012") for y in range(2016, today.year + 1)]
    sess, calls = requests.Session(), 0
    codes = _codes(settings, cache)
    for i, code in enumerate(codes, 1):
        corp = cc.get(code)
        if not corp:
            continue
        p = d / f"{code}.json"
        have = json.loads(p.read_text()) if p.exists() else {}
        new = False
        for y, rc in reports:
            key = f"{y}_{rc}"
            if key in have:
                continue
            js = v16._dart_get(sess, k, "exctvSttus", {"corp_code": corp, "bsns_year": str(y), "reprt_code": rc})
            have[key] = js.get("list", []) if js.get("status") == "000" else []
            calls, new = calls + 1, True
            time.sleep(0.1)
        if new:
            p.write_text(json.dumps(have, ensure_ascii=False))
        if i % 50 == 0:
            say(f"exctvSttus {i}/{len(codes)} codes, {calls} calls")
    return {"codes": len(codes), "calls": calls}


def _decode(raw: bytes) -> str:
    """DART document XML: UTF-8 from ~2022, EUC-KR (cp949) before - the declaration says which."""
    m = re.search(rb'encoding="([^"]+)"', raw[:200])
    enc = m.group(1).decode().lower() if m else "utf-8"
    for e in dict.fromkeys([enc, "utf-8", "cp949"]):
        try:
            return raw.decode("cp949" if e in ("euc-kr", "euckr", "ks_c_5601-1987") else e)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("cp949", errors="replace")


def section_text(xml: str) -> str:
    m = _SEC.search(xml)
    if not m:
        return ""
    t = re.sub(r"<[^>]+>", " ", m.group(1))
    t = re.sub(r"&[a-zA-Z]+;|&#\d+;", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def fetch_business(settings, cache) -> dict:
    """The business section of every annual report whose receipt number the rosters carry."""
    k = kd.key("DART_API_KEY")
    src, d = root(settings) / "exctv", root(settings) / "business"
    d.mkdir(parents=True, exist_ok=True)
    sess, calls, n = requests.Session(), 0, 0
    codes = _codes(settings, cache)
    for i, code in enumerate(codes, 1):
        p_src = src / f"{code}.json"
        if not p_src.exists():
            continue
        rosters = json.loads(p_src.read_text())
        want = {}
        for key, rows in rosters.items():
            y, rc = key.split("_")
            if rc == "11011" and rows:
                want[str(rows[0].get("rcept_no"))] = int(y)
        p = d / f"{code}.jsonl.gz"
        # done = a section was found, or the receipt was already read with the encoding fix (v2: none in the report)
        have = (
            {r["rcept_no"] for r in tone.read_records(p) if r.get("text") or r.get("v") == 2} if p.exists() else set()
        )
        todo = {r: y for r, y in want.items() if r not in have}
        if not todo:
            continue
        with gzip.open(p, "at") as fh:
            for rno, y in sorted(todo.items()):
                for attempt in range(4):
                    try:
                        r = sess.get(DART_DOC, params={"crtfc_key": k, "rcept_no": rno}, timeout=180)
                        break
                    except requests.RequestException:
                        time.sleep(5 * (attempt + 1))
                else:
                    continue
                calls += 1
                if r.headers.get("Content-Type", "").startswith("application/json") or len(r.content) < 200:
                    msg = r.text[:200]
                    if '"020"' in msg:
                        raise RuntimeError("DART 일일 한도(020) 도달 — 내일 다시 실행")
                    continue
                try:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    main = max(z.infolist(), key=lambda i: i.file_size).filename
                    text = section_text(_decode(z.read(main)))
                except (zipfile.BadZipFile, ValueError):
                    text = ""
                fh.write(json.dumps({"rcept_no": rno, "year": y, "text": text, "v": 2}, ensure_ascii=False) + "\n")
                n += 1
                time.sleep(0.2)
        if i % 25 == 0:
            say(f"business sections {i}/{len(codes)} codes, {n} reports, {calls} calls")
    return {"reports_new": n, "calls": calls}


def _sparql(q: str) -> pd.DataFrame:
    r = requests.post(WIKIDATA, data={"query": q}, headers=UA, timeout=300)
    r.raise_for_status()
    return pd.DataFrame([{k: v["value"] for k, v in x.items()} for x in r.json()["results"]["bindings"]])


def fetch_wikidata(settings) -> dict:
    ceo = _sparql(
        "SELECT ?cik ?ceo ?start ?end WHERE { ?co wdt:P5531 ?cik . ?co p:P169 ?st . ?st ps:P169 ?ceo ."
        " OPTIONAL { ?st pq:P580 ?start . } OPTIONAL { ?st pq:P582 ?end . } }"
    )
    ceo["ceo"] = ceo["ceo"].str.rsplit("/", n=1).str[-1]
    ids = [q for q in ceo["ceo"].unique() if q.startswith("Q")]
    parts = []
    for i in range(0, len(ids), 200):
        vals = " ".join(f"wd:{q}" for q in ids[i : i + 200])
        parts.append(
            _sparql(
                f"SELECT ?ceo ?school ?schoolLabel ?degree ?degreeLabel ?birth WHERE {{ VALUES ?ceo {{ {vals} }}"
                " OPTIONAL { ?ceo p:P69 ?e . ?e ps:P69 ?school . OPTIONAL { ?e pq:P512 ?degree . } }"
                " OPTIONAL { ?ceo wdt:P569 ?birth . }"
                ' SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }'
            )
        )
        time.sleep(2)
    ed = pd.concat(parts, ignore_index=True)
    ed["ceo"] = ed["ceo"].str.rsplit("/", n=1).str[-1]
    ceo.to_parquet(root(settings) / "wd_ceo_terms.parquet", index=False)
    ed.to_parquet(root(settings) / "wd_ceo_people.parquet", index=False)
    return {"terms": len(ceo), "ciks": int(ceo["cik"].nunique()), "people": len(ids)}


def main() -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--what", default="all", choices=["all", "exctv", "business", "wikidata"])
    a = ap.parse_args()
    s = load_settings()
    cache = DiskCache(s.cache_path)
    steps = {
        "wikidata": lambda: fetch_wikidata(s),
        "exctv": lambda: fetch_exctv(s, cache),
        "business": lambda: fetch_business(s, cache),
    }
    for name, fn in steps.items():
        if a.what in ("all", name):
            try:
                say(f"{name}: {fn()}")
            except Exception as exc:  # noqa: BLE001 - the other sources still load
                say(f"{name}: FAILED {type(exc).__name__}: {kd._redact(str(exc), kd.key('DART_API_KEY'))}")


if __name__ == "__main__":
    main()
