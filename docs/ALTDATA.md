# 대안 데이터 — DART·국민연금·SEC·채용 공고 (2026-10-03)

종목 선택 모델 v3([MODELING.md](MODELING.md) 16절)의 특징으로 쓰는 대안 데이터의 출처, 시점 규칙, 수집 방법, 한계입니다.
코드는 `src/quantcio/data/alt/`(수집)와 `src/quantcio/modeling/altfeatures.py`(특징)에 있습니다.

## 1. 한 가지 시점 규칙

모든 대안 데이터는 두 표로 정리됩니다(`data/alt/__init__.py`의 `AltData`).

| 표 | 행 | 열 |
|---|---|---|
| `events` | 날짜가 있는 사건 1건 | entity, published, kind, value, source |
| `measures` | 기간 측정값 1개 | entity, published, period_end, item, value, source |

- **`published < t`일 때만** 격자일 t에서 씁니다. 장중이나 장 마감 뒤에 나온 공시는 그날이 아니라 다음 격자일부터 씁니다.
- 공개일을 모르면 늦은 쪽(보수적)으로 정하고 `notes`에 적습니다.
- `covered`: 출처별로 데이터를 **완전히** 받은 엔티티 목록입니다. 받은 엔티티의 "사건 없음"은 0, 받지 못한 엔티티는 NaN입니다.

## 2. 출처

### 2.1 KR — OpenDART (`kr_dart_events.py`, `DART_API_KEY`)

| API | 내용 | 이력 | 모델 입력 |
|---|---|---|---|
| `list.json` (공시 유형 B·I·E) | 주요사항·거래소·기타 공시 목록 → 사건 분류(자사주 취득·처분·소각, 유상증자·CB/BW/EB, 공급계약, M&A, 최대주주·대표 변경, 불성실·관리종목·소송, 조회공시, 정정 등 26종) | 2013년~ | 예 (특징 10개) |
| `empSttus.json` (사업보고서) | 직원 수, 기간제 근로자, 평균 근속, 급여 총액 | 2014 사업연도~ | 예 (특징 4개) |
| `elestock.json` | 임원·주요주주 소유 변동 | **최근 약 2년만**(2024-10~) | 아니요 — 전진 기록 |
| `majorstock.json` | 5% 대량보유 변동 | **최근 약 2년만** | 아니요 — 전진 기록 |

- 호출 한도: 키당 하루 20,000회(상태 020). 모든 응답은 `.cache/quantcio/alt/kr_dart/`에 캐시되고, 한도에 걸리면 깔끔하게 멈춘 뒤
  다음 실행이 캐시에서 이어 받습니다. 네트워크가 끊기면 최대 15분까지 기다렸다가 멈춥니다. 로그와 오류 메시지에서 키는 `***`로 가립니다.
- 직원 현황 집계: `성별합계` 같은 합계 행이 있으면 합계 행만 씁니다(상세 행과 함께 더하면 두 배가 됩니다).
- `elestock`·`majorstock`은 응답을 캐시에 **합쳐** 쌓으므로, 오늘부터 이력이 계속 늘어납니다.

### 2.2 KR — 국민연금 가입 사업장 내역 (`kr_nps.py`, 키 불필요)

- 공공데이터포털 파일 데이터 15083277. "주기성 과거 데이터" 목록(130개 파일, 가장 이른 자료 월 2015-11)을 키 없이 받습니다.
  파일당 80~116MB(CSV, cp949), 전체 약 13GB, 약 95분. 법인 행만 남겨 월별 parquet(약 10MB)로 저장합니다.
- 상장사 매칭: OpenDART `company.json`의 사업자등록번호 앞 6자리 + 상호 정규화 일치. 같은 접두어의 다른 상호는 제외하고 셉니다.
- **공개일**: 2019-08 이후 파일은 등록일(자료 월 다음 달 17~31일), 수정됐으면 수정일. 2015-11~2019-06 자료 월은 2019-08-05~06에 한꺼번에
  등록돼 원래 공개일을 알 수 없으므로 등록일을 씁니다. 그래서 국민연금 특징은 2019-08-07부터 값이 있습니다.
- 한계: 본사 사업자번호로 신고한 인원만 셉니다(별도 사업자번호의 공장은 빠짐). 상호 변경 전 월은 매칭되지 않을 수 있습니다.

### 2.3 US — SEC (`us_sec.py`, `SEC_USER_AGENT`)

| 출처 | 내용 | 이력 |
|---|---|---|
| 제출 목록(submissions JSON + 이전 페이지) | Form 4·144·SC 13D/G·NT·424B·S-3 건수, 8-K 항목별 건수 | 상장 이후 전부 |
| Form 3/4/5 내부자 거래 데이터셋(분기 zip) | 장내 매수(P)·매도(S) 금액과 보고자 수 | 2006년~ |

- `published` = SEC 접수 시각의 뉴욕 날짜(없으면 filingDate). 내부자 데이터셋은 FILING_DATE입니다.
- 다른 회사에 관한 13D/G·144(회사가 신고자인 경우)는 제외합니다. Form 4는 구분이 안 돼 약 0.1%가 섞입니다.
- 데이터셋은 분기 단위로 늦게 나옵니다(현재 2026-06-30까지). 제출 목록 캐시는 `alt.refresh_days`(7일)보다 오래되면 다시 받습니다.

### 2.4 매크로 노출 (두 시장)

종목의 일간 로그수익을 [벤치마크, 요인]에 252거래일(126일 이상) 회귀한 요인 계수: 환율(KR USD/KRW, US 달러인덱스), WTI, 미 10년물 금리,
HYG/LQD. 요인 수준은 t − 매크로 지연(KR 1일)까지만 씁니다. 레짐 특징으로 VIX/VIX3M(기간 구조)을 추가했습니다.

### 2.5 채용 공고 — 전진 기록만 (`kr_jobs.py`)

- 과거 공고는 어디서도 받을 수 없으므로, 오늘부터 하루 한 번 스냅숏을 **추가 전용 원장** `alt_ledger/kr_jobs.jsonl`에 남깁니다.
  12개월 이상 쌓이기 전에는 모델에 넣지 않습니다.
- 사람인 Open API(`SARAMIN_API_KEY`, https://oapi.saramin.co.kr/ 에서 신청, 하루 500회): 상장사 공고 전체를 훑어 회사명·사업자번호로
  엄격하게 매칭합니다.
- 고용24 Open API(`WORK24_API_KEY`): **기업회원만** 키를 받을 수 있습니다(개인회원 키는 거부).
- 기업 리뷰(잡플래닛·블라인드)는 공개 API가 없고 약관이 자동 수집을 금지하며 과거 시점 기록도 없어 **수집하지 않습니다**.
- 미국 Glassdoor(2026-10-04 검토): 공개 API는 2021~2022년에 닫혔고(현재는 고용주 대상 영업 계약만), 약관이 자동 수집을 금지하므로
  **수집하지 않습니다**. 학계 근거는 있습니다(Green·Huang·Wen·Zhou 2019, JFE: 평점이 오른 회사가 내린 회사보다 수익이 높음, 2008~2016 자료).
  쓰려면 라이선스 데이터(리뷰 평점 이력을 파는 대안 데이터 업체)를 사야 하며, 날짜별 평점 이력이 있어야 시점 규칙을 지킬 수 있습니다.

## 3. 명령

```bash
quantcio alt fetch --market KR      # DART·국민연금 캐시 갱신(모델링 유니버스 + XSH 구간 종목), 주 1회 권장
quantcio alt fetch --market US      # SEC 제출 목록·내부자 데이터셋 갱신
quantcio alt jobs                   # 오늘의 채용 공고 스냅숏을 원장에 추가 (키 필요, 하루 1회)
quantcio alt status                 # 캐시 현황
```

`quantcio modeling select`는 데이터셋을 만들 때 같은 함수로 대안 데이터를 읽습니다(`alt` 설정 블록: `enabled`, `kr_dart`,
`kr_nps: auto`, `us_sec`, `refresh_days`, `macro_betas`).

## 4. 특징 목록

[MODELING.md](MODELING.md) 16.3절의 표와 같습니다(16.8절의 수정 반영: KR 내부자·5% 특징 제외).
리포트의 "주요 기여 특징"에는 대안 데이터 특징도 한국어 이름으로 나옵니다(`modeling/features.py`의 `FEATURE_DOCS`).

## 5. 무료 API 키로 여는 연구 데이터 (2026-10-04)

키는 `.env`에만 넣습니다(빈 칸이 이미 있음, 설명은 `.env.example`). 값은 어떤 로그에도 출력하지 않습니다. 넣은 뒤:

```bash
.venv/bin/python scripts/api_keys.py            # 키 확인(실제 호출로 유효성 점검) + 무엇이 열리는지
.venv/bin/python scripts/api_keys.py fetch      # 키가 있는 데이터 전부 받기(캐시, 이어받기 가능)
.venv/bin/python scripts/api_keys.py ledger     # 매일 1회: Finnhub·네이버 기록(과거 이력이 없어 오늘부터 쌓음)
.venv/bin/python scripts/keyed_study.py --market US   # 미리 정한 시험(MODELING.md 23)
.venv/bin/python scripts/keyed_study.py --market KR
```

| 키 | 시장 | 여는 데이터 | 받는 곳 | 비용·한도 |
|---|---|---|---|---|
| `USPTO_API_KEY` | 미국 | 회사별 특허 등록 수(PatentsView) | data.uspto.gov/apikey (USPTO.gov 계정 + ID.me 본인 인증) | 무료, 같은 파일 키당 연 20회 |
| `FRED_API_KEY` | 미국 | 실업수당 신규 청구·금융여건지수·소비자심리(최초 발표값), 2년 금리·장단기 금리차 | fredaccount.stlouisfed.org/apikeys | 무료, 분당 120회 |
| `BLS_API_KEY` | 미국 | JOLTS 업종별 채용·퇴사·해고·구인율(키 없으면 하루 25회) | data.bls.gov/registrationEngine | 무료, 하루 500회 |
| `EIA_API_KEY` | 미국 | 천연가스 저장량(주간) | eia.gov/opendata/register.php | 무료 |
| `FINNHUB_API_KEY` | 미국 | 애널리스트 추천 분포·뉴스 건수(전진 기록) | finnhub.io/register | 무료, 분당 60회 |
| `ECOS_API_KEY` | 한국 | 기준금리·물가·경제심리·소비자심리·업종별 기업경기전망·산업생산 | ecos.bok.or.kr/api (인증키 신청) | 무료 |
| `KOSIS_API_KEY` | 한국 | 사업체노동력조사 산업별 입직률·이직률 | kosis.kr/openapi (활용신청) | 무료 |
| `KIPRIS_API_KEY` | 한국 | 회사별 특허 등록 수(연도별) | plus.kipris.or.kr | 무료 할당량(가입 화면 확인), 분당 60회 |
| `NAVER_CLIENT_ID` + `NAVER_CLIENT_SECRET` | 한국 | 회사별 뉴스 건수(전진 기록) | developers.naver.com/apps (검색 API) | 무료, 하루 25,000회 |
| `SARAMIN_API_KEY` / `WORK24_API_KEY` | 한국 | 채용공고 전진 기록(`quantcio alt jobs`) | oapi.saramin.co.kr / 고용24(기업회원만) | 무료 |

- 키 없이 이미 쓰는 무료 출처: SEC EDGAR(`SEC_USER_AGENT`만 필요), 이민국 H-1B, NOAA·Open-Meteo 날씨, 정책 불확실성(EPU·TPU·GPR),
  샌프란시스코 연준 뉴스 심리, 연방 관보, FINRA 공매도, OpenDART(키 있음), 국민연금 사업장 파일.
- 무료로는 안 되는 것: Glassdoor(API 종료·수집 금지), 회사별 뉴스 과거 이력(GDELT는 요청 제한), 애널리스트 추정치 변경 이력(유료).
- 특허 시점 규칙: 미국은 **등록일**(공개되는 날) 기준 직전 12개 완결 월, 한국은 **등록연도** 기준 전년(1월 1일부터 사용). 출원일 기준은 공개 전이라 쓰지 않습니다.
- KIPRIS 출원인 검색은 이름 부분 일치라 두 글자 이름(예: "LG")은 계열사 전체가 잡혀 제외합니다.
