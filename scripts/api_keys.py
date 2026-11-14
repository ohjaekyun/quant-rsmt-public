"""Free API keys for the research data (docs/ALTDATA.md section 6): status check and one-command fetch.

    .venv/bin/python scripts/api_keys.py            # which keys are set / valid, and what each one unlocks
    .venv/bin/python scripts/api_keys.py fetch      # download everything the set keys allow (cached, resumable)
    .venv/bin/python scripts/api_keys.py fetch --only ECOS_API_KEY
    .venv/bin/python scripts/api_keys.py ledger     # daily forward ledgers (Finnhub, Naver) - run once a day
    .venv/bin/python scripts/keyed_study.py --market US   # then: the pre-declared research run (MODELING.md 23)

Keys live in ``.env`` only (see ``.env.example``); values are never printed.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keyed_data as kd  # noqa: E402


@dataclass
class KeySpec:
    env: str
    market: str
    service: str
    unlocks: str
    signup: str
    how: str
    limit: str
    check: object = None
    extra_env: list[str] = field(default_factory=list)
    used_by: str = ""


KEYS = [
    # ---------------------------------------------------------------- overseas
    KeySpec(
        "USPTO_API_KEY",
        "미국",
        "미국 특허청 오픈데이터 포털(USPTO ODP)",
        "회사별 특허 등록 수(PatentsView 표)",
        "https://data.uspto.gov/apikey",
        "USPTO.gov 계정 만들기 → ID.me 본인 인증 → 'Manage API Key'에서 발급",
        "무료 · 같은 파일은 키당 연 20회 다운로드",
        kd.uspto_check,
        used_by="keyed_study US: us_pat_12m, us_pat_growth",
    ),
    KeySpec(
        "FRED_API_KEY",
        "미국",
        "세인트루이스 연준 FRED",
        "실업수당 신규 청구·금융여건지수·미시간 소비자심리(최초 발표값), 2년 금리·장단기 금리차",
        "https://fredaccount.stlouisfed.org/apikeys",
        "FRED 계정 가입 → API Keys → Request API Key(즉시)",
        "무료 · 분당 120회",
        kd.fred_check,
        used_by="keyed_study US: us_beta_dgs2, us_impact_dgs2, r_icsa, r_nfci, r_umcsent, r_t10y2y",
    ),
    KeySpec(
        "BLS_API_KEY",
        "미국",
        "미국 노동통계국 BLS API v2",
        "업종별 채용·자발적 퇴사·해고·구인율(JOLTS) — 키 없이도 v1로 받지만 하루 25회 제한",
        "https://data.bls.gov/registrationEngine/",
        "이름·이메일 등록 → 메일로 키 수신 → 메일 링크로 활성화",
        "무료 · 하루 500회, 1회 50계열·20년",
        kd.bls_check,
        used_by="us_v8 JOLTS + keyed_study US: us_sec_openings",
    ),
    KeySpec(
        "EIA_API_KEY",
        "미국",
        "미국 에너지정보청 EIA",
        "천연가스 저장량(주간) — 날씨·에너지 충격",
        "https://www.eia.gov/opendata/register.php",
        "이메일 등록 → 즉시 발급",
        "무료",
        kd.eia_check,
        used_by="keyed_study US: us_impact_ngstor, r_ng_stor",
    ),
    KeySpec(
        "FINNHUB_API_KEY",
        "미국",
        "Finnhub",
        "애널리스트 추천 분포·회사 뉴스 건수(무료는 과거 이력이 짧아 오늘부터 기록)",
        "https://finnhub.io/register",
        "가입 즉시 대시보드에 키 표시",
        "무료 · 분당 60회",
        kd.finnhub_check,
        used_by="ledger: alt_ledger/us_finnhub.jsonl (12개월 이상 쌓인 뒤 시험)",
    ),
    # ---------------------------------------------------------------- domestic
    KeySpec(
        "ECOS_API_KEY",
        "한국",
        "한국은행 경제통계시스템 ECOS",
        "기준금리·소비자물가·경제심리지수·소비자심리·기업경기전망(업종별)·전산업생산",
        "https://ecos.bok.or.kr/api/#/AuthKeyApply",
        "ECOS 회원가입 → 인증키 신청(즉시)",
        "무료",
        kd.ecos_check,
        used_by="keyed_study KR: kr_sec_bsi, r_kr_*",
    ),
    KeySpec(
        "KOSIS_API_KEY",
        "한국",
        "통계청 KOSIS 공유서비스",
        "사업체노동력조사 — 산업별 입직률·이직률(월별)",
        "https://kosis.kr/openapi/",
        "회원가입 → '활용신청' → 인증키 발급(즉시)",
        "무료",
        kd.kosis_check,
        used_by="keyed_study KR: kr_sec_hire, kr_sec_sep",
    ),
    KeySpec(
        "KIPRIS_API_KEY",
        "한국",
        "특허정보원 KIPRIS Plus",
        "회사(출원인)별 특허 등록 수(연도별)",
        "https://plus.kipris.or.kr",
        "회원가입 → Open API 무료 서비스 신청 → 마이페이지에서 키 확인",
        "무료 할당량 있음(가입 화면의 월·일 한도 확인) · 분당 60회",
        kd.kipris_check,
        used_by="keyed_study KR: kr_pat_1y, kr_pat_growth",
    ),
    KeySpec(
        "NAVER_CLIENT_ID",
        "한국",
        "네이버 검색 API",
        "회사별 뉴스 건수(최근 기사만 검색돼 오늘부터 기록)",
        "https://console.ncloud.com (NAVER API HUB)",
        "네이버 클라우드 콘솔 → NAVER API HUB → 이용 신청 → 인증 정보(Client ID / Client Secret)",
        "무료 · 하루 25,000회",
        lambda cid: kd.naver_check(cid, kd.key("NAVER_CLIENT_SECRET")),
        extra_env=["NAVER_CLIENT_SECRET"],
        used_by="ledger: alt_ledger/kr_news_naver.jsonl",
    ),
]


def _reddit_check(cid: str) -> str:
    import requests

    r = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        auth=(cid, kd.key("REDDIT_CLIENT_SECRET")),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": "quantcio-research/1.0 (by quantcio research script)"},
        timeout=60,
    )
    return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"


KEYS.append(
    KeySpec(
        "REDDIT_CLIENT_ID",
        "미국",
        "레딧 API(앱 전용 인증)",
        "r/wallstreetbets·stocks·investing 게시물의 종목 언급 수·강세/약세 단어 수(오늘부터 기록)",
        "https://support.reddithelp.com (Developer Support → Data API 접근 요청)",
        "2025-11부터 사전 승인제(Responsible Builder Policy): 지원 양식으로 접근 요청(역할 'App not supported by "
        "Devvit', 비상업 연구 목적·호출량 기재) → 이메일 승인(1~4주, 거절 가능) → 승인 후 reddit.com/prefs/apps에서 "
        "'create another app' → 종류 'script', redirect uri http://localhost:8080 → 앱 이름 아래 문자열이 Client ID, "
        "'secret'이 Client Secret",
        "무료(비상업), 분당 100회",
        _reddit_check,
        extra_env=["REDDIT_CLIENT_SECRET"],
        used_by="ledger: alt_ledger/us_reddit.jsonl",
    )
)
EXISTING = [  # already wired in src/quantcio (status only)
    ("ANTHROPIC_API_KEY", "LLM 분석(리포트)"),
    ("DART_API_KEY", "OpenDART 공시·지분·직원(한국)"),
    ("SEC_USER_AGENT", "SEC EDGAR 접속 이름(미국, 키가 아니라 '조직명 이메일')"),
    ("SARAMIN_API_KEY", "사람인 채용공고 기록(한국)"),
    ("WORK24_API_KEY", "고용24 채용공고 기록(한국, 기업회원만)"),
    ("DATA_GO_KR_KEY", "공공데이터포털(선택 — 국민연금 파일은 키 없이 받음)"),
]


def status(live: bool = True) -> None:
    print("\n[추가 무료 키]")
    for s in KEYS:
        v = kd.key(s.env)
        extra_ok = all(kd.key(e) for e in s.extra_env)
        if v and extra_ok and live and s.check:
            try:
                res = s.check(v)
            except Exception as exc:  # noqa: BLE001 - shown to the user, key redacted
                res = kd._redact(f"{type(exc).__name__}: {exc}", v)[:120]
            mark = "확인됨" if res == "ok" else f"오류: {res}"
        else:
            mark = "입력됨(확인 생략)" if v and extra_ok else "없음"
        names = " + ".join([s.env, *s.extra_env])
        print(f"- {s.market} | {names}: {mark}")
        print(f"    {s.service} — {s.unlocks}")
        if mark == "없음":
            print(f"    받는 곳: {s.signup}  ({s.how}; {s.limit})")
    print("\n[이미 쓰는 키]")
    for env, what in EXISTING:
        print(f"- {env}: {'입력됨' if kd.key(env) else '없음'} — {what}")


def _kr_companies(settings, cache) -> dict[str, str]:
    from quantcio.data.pit import load_pit_universe
    from quantcio.modeling.dataset import universe_params
    from quantcio.models import Market

    n, start = universe_params(Market.KR, settings)  # the KR modeling universe (same as build_dataset)
    uni = load_pit_universe(Market.KR, settings, cache, n, start)
    m = uni.membership.sort_values("date").drop_duplicates("entity", keep="last")
    return {str(e): str(n) for e, n in zip(m["entity"], m.get("name", m["entity"]), strict=False)}


def _us_tickers(settings, cache) -> list[str]:
    from quantcio.data.pit import load_pit_universe
    from quantcio.models import Market

    uni = load_pit_universe(Market.US, settings, cache, 500, date.fromisoformat(settings.backtest.us_start))
    last = uni.membership["date"].max()
    return sorted(uni.membership.loc[uni.membership["date"] == last, "ticker"].astype(str).unique())


def fetch(only: str | None = None) -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    settings = load_settings()
    cache = DiskCache(settings.cache_path)

    def say(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    jobs = {
        "USPTO_API_KEY": lambda: kd.uspto_fetch(settings, say),
        "FRED_API_KEY": lambda: kd.fred_fetch(settings, say),
        "BLS_API_KEY": lambda: kd.bls_fetch(settings, say),
        "EIA_API_KEY": lambda: kd.eia_fetch(settings, say),
        "ECOS_API_KEY": lambda: kd.ecos_fetch(settings, say),
        "KOSIS_API_KEY": lambda: kd.kosis_fetch(settings, say),
        "KIPRIS_API_KEY": lambda: kd.kipris_fetch(settings, _kr_companies(settings, cache), say),
    }
    for env, job in jobs.items():
        if only and env != only:
            continue
        if not kd.key(env):
            say(f"{env}: 없음 — 건너뜀")
            continue
        try:
            say(f"{env}: {job()}")
        except Exception as exc:  # noqa: BLE001 - one source failing does not stop the others
            say(kd._redact(f"{env}: 실패 {type(exc).__name__}: {exc}", kd.key(env))[:400])


def ledgers(only: str | None = None) -> None:
    from quantcio.config import load_settings
    from quantcio.data.cache import DiskCache

    settings = load_settings()
    cache = DiskCache(settings.cache_path)
    secrets = [kd.key(e) for e in ("FINNHUB_API_KEY", "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET")]
    jobs = []
    if kd.key("FINNHUB_API_KEY"):
        jobs.append(("Finnhub", lambda: kd.finnhub_ledger(settings, _us_tickers(settings, cache), print)))
    if kd.key("NAVER_CLIENT_ID") and kd.key("NAVER_CLIENT_SECRET"):
        jobs.append(("Naver", lambda: kd.naver_ledger(_kr_companies(settings, cache), print)))
    import attention_data as att

    jobs.append(("ApeWisdom", lambda: att.apewisdom_ledger(print)))
    import us_v10_data as v10

    jobs.append(("Options", lambda: v10.options_ledger(_us_tickers(settings, cache), print)))
    jobs.append(("EPS trend", lambda: v10.eps_trend_ledger(_us_tickers(settings, cache), print)))
    if kd.key("REDDIT_CLIENT_ID") and kd.key("REDDIT_CLIENT_SECRET"):
        jobs.append(("Reddit", lambda: att.reddit_ledger(_us_tickers(settings, cache), print)))
    secrets += [kd.key("REDDIT_CLIENT_ID"), kd.key("REDDIT_CLIENT_SECRET")]
    if only:
        jobs = [(n, j) for n, j in jobs if n.lower() == only.lower()]
        if not jobs:
            print(f"--only {only}: 해당 기록 없음 (Finnhub, Naver, ApeWisdom, Options, EPS trend, Reddit)")
    for name, job in jobs:
        try:
            print(f"{name}: {job()}")
        except Exception as exc:  # noqa: BLE001 - one ledger failing does not stop the other; keys redacted
            print(kd._redact(f"{name}: 실패 {type(exc).__name__}: {exc}", *secrets)[:400])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="status", choices=["status", "fetch", "ledger"])
    ap.add_argument("--only", default=None, help="fetch: one key name (ECOS_API_KEY); ledger: one ledger (Naver)")
    ap.add_argument("--no-live", action="store_true", help="status without test calls")
    a = ap.parse_args()
    if a.cmd == "status":
        status(not a.no_live)
    elif a.cmd == "fetch":
        fetch(a.only)
    else:
        ledgers(a.only)


if __name__ == "__main__":
    main()
