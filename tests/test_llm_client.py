"""LLMClient against a real AsyncAnthropic over httpx2.MockTransport (no key, no network), plus FakeLLMClient."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import transform_schema
from conftest import make_market_data

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.llm import run_llm_analysis
from quantcio.llm.client import (
    FALLBACK_BETA,
    TASK_LABELS,
    TASK_LABELS_EN,
    FakeLLMClient,
    LLMClient,
    LLMConfigError,
    LLMRecorder,
    api_error_message,
    estimate_cost,
    retry_delay,
    schema_digest,
    schema_for,
    task_label,
    usage_dict,
)
from quantcio.llm.prompts import (
    CRITIC_SYSTEM,
    CRITIC_SYSTEM_EN,
    FILINGS_SYSTEM,
    FILINGS_SYSTEM_EN,
    NEWS_SYSTEM,
    NEWS_SYSTEM_EN,
)
from quantcio.llm.schemas import (
    CriticWire,
    CriticWireEN,
    FilingsWire,
    FilingsWireEN,
    NewsWire,
    NewsWireEN,
    wire_model,
)
from quantcio.models import ExecutiveSignal, Market, QuantSignal, Signal, Technicals

# ----------------------------------------------------------------------------- mock helpers

NEWS_OK = {
    "articles": [
        {
            "news_id": "a1",
            "sentiment": 0.5,
            "relevance": 0.9,
            "event_type": "earnings",
            "evidence": "영업이익 10조원 전망",
        }
    ],
    "key_positives": ["실적 개선"],
    "key_negatives": [],
    "key_events": [],
}
DEFAULT_USAGE = {
    "input_tokens": 1000,
    "output_tokens": 500,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}

Step = tuple[int, dict, dict] | BaseException | Callable[[httpx2.Request], Any]


def mock_message(
    content: list[dict],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict | None = None,
    model: str = "claude-opus-5",
    usage: dict | None = None,
) -> dict:
    return {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": usage or dict(DEFAULT_USAGE),
    }


def ok(payload: dict | str = NEWS_OK, **kw: Any) -> tuple[int, dict, dict]:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return (
        200,
        mock_message([{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}], **kw),
        {},
    )


def err(status: int, etype: str, msg: str, headers: dict | None = None) -> tuple[int, dict, dict]:
    return status, {"type": "error", "error": {"type": etype, "message": msg}}, headers or {}


class Transport:
    """Scripted async handler that records every request (url, headers, json body)."""

    def __init__(self, steps: list[Step], delay: float = 0.0):
        self.steps = list(steps)
        self.sent: list[dict[str, Any]] = []
        self.delay = delay
        self.inflight = 0
        self.peak = 0

    async def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.sent.append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "raw": bytes(request.content),
                "json": json.loads(request.content),
            }
        )
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
            if isinstance(step, BaseException):
                raise step
            if callable(step):
                step = step(request)
            status, body, headers = step
            return httpx2.Response(status, json=body, headers={"request-id": f"req_{len(self.sent)}", **headers})
        finally:
            self.inflight -= 1


def llm_settings(base: Settings, **llm: Any) -> Settings:
    s = base.model_copy(deep=True)
    s.llm.response_cache = False
    for k, v in llm.items():
        setattr(s.llm, k, v)
    return s


def make_llm(settings: Settings, transport: Transport, **kw: Any) -> LLMClient:
    ac = anthropic.AsyncAnthropic(
        api_key="sk-ant-test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(transport)),
    )
    return LLMClient(settings, anthropic_client=ac, **kw)


def call(llm: LLMClient, task: str = "news", wire=NewsWire, user: str = "USER", **kw: Any):
    return asyncio.run(llm.call_structured(task, NEWS_SYSTEM, user, wire, **kw))


# ------------------------------------------------------------------------------ wire shape


VERIFIED_BODY_KEYS = {"model", "max_tokens", "system", "messages", "thinking", "output_config", "fallbacks"}


def assert_verified_wire(
    req: dict,
    *,
    system: str,
    wire_model,
    effort: str,
    max_tokens: int,
    model: str = "claude-opus-5",
    fallbacks: bool = True,
) -> None:
    """The request matches the live-probed reference (docs/sources/anthropic-sdk.json, route C "RECOMMENDED")."""
    assert req["method"] == "POST"
    assert req["url"] == "https://api.anthropic.com/v1/messages?beta=true"
    h = req["headers"]
    assert h["anthropic-version"] == "2023-06-01" and h["x-api-key"] == "sk-ant-test"
    assert h["content-type"] == "application/json"
    body = req["json"]
    assert set(body) == (VERIFIED_BODY_KEYS if fallbacks else VERIFIED_BODY_KEYS - {"fallbacks"})
    if fallbacks:
        # exactly the one beta (no structured-outputs beta appended, which the parse() route would add)
        assert h["anthropic-beta"] == FALLBACK_BETA == "server-side-fallback-2026-07-01"
        assert body["fallbacks"] == "default"
    else:
        assert "anthropic-beta" not in h
    assert body["model"] == model
    assert body["max_tokens"] == max_tokens
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {
        "effort": effort,
        "format": {"type": "json_schema", "schema": transform_schema(wire_model)},
    }
    assert body["system"] == [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    assert [m["role"] for m in body["messages"]] == ["user"] and isinstance(body["messages"][0]["content"], str)


def test_wire_request_shape(settings):
    s = llm_settings(settings)
    t = Transport([ok()])
    obj, usage, error = call(make_llm(s, t), user="삼성전자 뉴스 USER")
    assert error is None and isinstance(obj, NewsWire) and obj.articles[0].news_id == "a1"
    req = t.sent[0]
    assert_verified_wire(req, system=NEWS_SYSTEM, wire_model=NewsWire, effort="low", max_tokens=8000)
    assert s.llm.model == "claude-opus-5" and s.llm.effort["news"] == "low" and s.llm.max_tokens["news"] == 8000
    assert req["json"]["messages"] == [{"role": "user", "content": "삼성전자 뉴스 USER"}]
    assert "삼성전자".encode() in req["raw"]  # Hangul goes on the wire as raw UTF-8
    assert not {"temperature", "top_p", "top_k", "betas", "output_format"} & set(req["json"])
    assert usage["attempts"] == 1 and usage["model_served"] == "claude-opus-5" and usage["notices"] == []


def _mini_quant() -> QuantSignal:
    tech = Technicals(
        last_close=60000.0,
        atr=1500.0,
        atr_pct=0.025,
        rsi14=58.3,
        high_52w=65000.0,
        low_52w=45000.0,
        realized_vol_60d=0.28,
        max_drawdown_1y=-0.18,
    )
    return QuantSignal(technicals=tech, factors=[], factor_pillar=0.2, ml_pillar=0.1)


def _mini_prelim() -> ExecutiveSignal:
    return ExecutiveSignal(
        signal=Signal.BUY,
        composite=0.21,
        confidence=62,
        target_price=67200.0,
        upside=0.12,
        horizon_days=60,
        horizon_label="1~3개월",
        pillars={"factor": 0.2, "ml": 0.1, "llm": 0.3},
    )


class Router:
    """MockTransport handler that answers each task by its (byte-stable) system prompt, like the real API would.

    ``reject_beta``: every request carrying the fallback beta gets the org-level 400 rejection.
    ``fallback_for``: that task is served by ``claude-sonnet-5`` via a server-side fallback.
    """

    def __init__(self, md, *, reject_beta: bool = False, fallback_for: str | None = None):
        self.reject_beta, self.fallback_for = reject_beta, fallback_for
        self.payloads = {
            NEWS_SYSTEM: (
                "news",
                {
                    "articles": [
                        {
                            "news_id": n.id,
                            "sentiment": 0.4,
                            "relevance": 1.0,
                            "event_type": "product",
                            "evidence": n.title,
                        }
                        for n in md.news
                    ],
                    "key_positives": [],
                    "key_negatives": [],
                    "key_events": [],
                },
            ),
            FILINGS_SYSTEM: (
                "filings",
                {
                    "summary": [],
                    "earnings_highlights": ["매출액 79조 1,000억원"],
                    "guidance": "none",
                    "tone": 0.2,
                    "risk_factors": [],
                },
            ),
            CRITIC_SYSTEM: (
                "critic",
                {
                    "thesis": "추세가 견조하다.",
                    "bull_points": [],
                    "bear_points": [],
                    "contradictions": [],
                    "confidence_adjustment": 0,
                    "critical_flag": False,
                    "critical_category": None,
                    "critical_reason": None,
                    "critical_evidence": None,
                    "critical_source_ref": None,
                },
            ),
        }
        self.by_task: dict[str, list[dict]] = {}

    def __call__(self, request: httpx2.Request):
        body = json.loads(request.content)
        task, payload = self.payloads[body["system"][0]["text"]]
        self.by_task.setdefault(task, []).append(
            {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "raw": bytes(request.content),
                "json": body,
            }
        )
        if self.reject_beta and "anthropic-beta" in request.headers:
            return err(
                400, "invalid_request_error", f"Unexpected value(s) `{FALLBACK_BETA}` for the `anthropic-beta` header."
            )
        if task == self.fallback_for:
            content = [
                {
                    "type": "fallback",
                    "from": {"model": "claude-opus-5"},
                    "to": {"model": "claude-sonnet-5"},
                    "trigger": {"type": "refusal", "category": "cyber"},
                },
                {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
            ]
            return 200, mock_message(content, model="claude-sonnet-5"), {}
        return ok(payload)


SYSTEMS = {
    "news": (NEWS_SYSTEM, NewsWire),
    "filings": (FILINGS_SYSTEM, FilingsWire),
    "critic": (CRITIC_SYSTEM, CriticWire),
}


@pytest.mark.parametrize("scenario", ["clean", "fallback_served", "beta_rejected"])
def test_run_llm_analysis_wire_and_notices_end_to_end(settings, scenario):
    """Every task's request through a REAL AsyncAnthropic matches the verified reference, and degraded-but-served
    paths (fallback model, rejected beta) land in LLMIntelligence.notices, never in errors."""
    s = llm_settings(settings)
    md = make_market_data(Market.KR)
    router = Router(
        md, reject_beta=scenario == "beta_rejected", fallback_for="filings" if scenario == "fallback_served" else None
    )
    llm = make_llm(s, Transport([router]))
    out = asyncio.run(
        run_llm_analysis(llm, md, _mini_quant(), s, lambda p: _mini_prelim(), now=datetime(2026, 9, 24, tzinfo=UTC))
    )

    assert out.news is not None and out.filings is not None and out.critic is not None
    assert out.errors == [], out.errors
    assert set(router.by_task) == {"news", "filings", "critic"}
    for task, reqs in router.by_task.items():
        system, wire = SYSTEMS[task]
        kw = dict(system=system, wire_model=wire, effort=s.llm.effort[task], max_tokens=s.llm.max_tokens[task])
        if scenario == "beta_rejected":
            # news/filings: the first attempt carries the beta and is rejected, then a retry without it;
            # the critic runs after the client switched fallbacks off, so it never sends the beta at all
            assert_verified_wire(reqs[-1], fallbacks=False, **kw)
            assert all("anthropic-beta" not in r["headers"] for r in reqs[1:])
        else:
            assert len(reqs) == 1
            assert_verified_wire(reqs[0], **kw)
    assert [s.llm.effort[t] for t in ("news", "filings", "critic")] == ["low", "medium", "high"]

    if scenario == "clean":
        assert out.notices == []
    elif scenario == "fallback_served":
        assert out.notices == ["공시 분석: 요청 모델 claude-opus-5 대신 서버 폴백 모델 claude-sonnet-5이(가) 응답"]
    else:
        assert len(router.by_task["critic"]) == 1 and not llm.fallbacks_active
        assert any(
            n.startswith("뉴스 감성: 서버 폴백 베타 거부") or n.startswith("공시 분석: 서버 폴백 베타 거부")
            for n in out.notices
        )
        assert any(n.startswith("비평: 서버 폴백 비활성(베타 헤더 거부") for n in out.notices)
        # one line per call: the rejecting call does not ALSO say "비활성"; no raw SDK dict dumps
        for label in ("뉴스 감성", "공시 분석"):
            rejected = [n for n in out.notices if n.startswith(f"{label}: 서버 폴백 베타 거부")]
            if rejected:
                assert not any(n.startswith(f"{label}: 서버 폴백 비활성") for n in out.notices)
        assert all("{'type'" not in n and "Error code" not in n for n in out.notices)
        assert any("for the `anthropic-beta` header" in n for n in out.notices)
    # the notices are also logged per call and deduplicated
    assert len(out.notices) == len(set(out.notices))


def test_effort_and_max_tokens_follow_task(settings):
    s = llm_settings(settings)
    t = Transport([ok({"summary": [], "earnings_highlights": [], "guidance": "none", "tone": 0.0, "risk_factors": []})])
    obj, _, error = call(make_llm(s, t), task="filings", wire=FilingsWire)
    assert error is None and obj.guidance == "none"
    body = t.sent[0]["json"]
    assert body["output_config"]["effort"] == s.llm.effort["filings"] == "medium"
    assert body["max_tokens"] == 12000
    assert body["output_config"]["format"]["schema"] == transform_schema(FilingsWire)


def test_no_fallbacks_when_disabled(settings):
    t = Transport([ok()])
    _, usage, error = call(make_llm(llm_settings(settings, server_fallbacks=False), t))
    assert error is None
    assert_verified_wire(
        t.sent[0], system=NEWS_SYSTEM, wire_model=NewsWire, effort="low", max_tokens=8000, fallbacks=False
    )
    assert usage["notices"] == []


def test_beta_rejection_retries_without_fallbacks_and_is_observable(settings):
    rej = err(400, "invalid_request_error", f"Unexpected value(s) `{FALLBACK_BETA}` for the `anthropic-beta` header.")
    t = Transport([rej, ok(), ok()])
    llm = make_llm(llm_settings(settings), t)

    async def two_calls():
        a = await llm.call_structured("news", NEWS_SYSTEM, "U1", NewsWire)
        b = await llm.call_structured("news", NEWS_SYSTEM, "U2", NewsWire)
        return a, b

    (obj1, u1, e1), (obj2, u2, e2) = asyncio.run(two_calls())
    assert obj1 is not None and obj2 is not None and e1 is None and e2 is None
    kw = dict(system=NEWS_SYSTEM, wire_model=NewsWire, effort="low", max_tokens=8000)
    assert_verified_wire(t.sent[0], fallbacks=True, **kw)
    assert_verified_wire(t.sent[1], fallbacks=False, **kw)  # same request minus the beta header + body field
    assert_verified_wire(t.sent[2], fallbacks=False, **kw)  # stays off for the client's life
    assert len(t.sent) == 3 and not llm.fallbacks_active
    assert u1["notices"] == [
        "서버 폴백 베타 거부 → 폴백 없이 재시도(Unexpected value(s) "
        f"`{FALLBACK_BETA}` for the `anthropic-beta` header.)"
    ]
    assert any("서버 폴백 비활성" in n for n in u2["notices"])  # still visible on later calls
    assert any("거부" in n for n in llm.notices)


def test_generic_400_is_an_error_not_a_retry(settings):
    t = Transport([err(400, "invalid_request_error", "max_tokens: must be positive")])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and len(t.sent) == 1 and usage["cost_usd"] == 0.0
    # the server's own message, not the SDK's "Error code: 400 - {'type': 'error', ...}" dict dump
    assert error == "잘못된 요청(400): max_tokens: must be positive"


def test_api_error_message_prefers_the_server_message():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "not_found_error", "message": "model: claude-x"}}
    e = anthropic.NotFoundError("Error code: 404 - {...}", response=httpx2.Response(404, request=req), body=body)
    assert api_error_message(e) == "model: claude-x"
    html = anthropic.InternalServerError(
        "<html>bad gateway</html>", response=httpx2.Response(502, request=req), body="<html>bad gateway</html>"
    )
    assert api_error_message(html) == "<html>bad gateway</html>"  # non-JSON body: SDK message as is


# ---------------------------------------------------------------------- stop reasons / validation


def test_refusal_returns_error_and_zero_cost(settings):
    ref = (
        200,
        mock_message(
            [], stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"}
        ),
        {},
    )
    llm = make_llm(llm_settings(settings), Transport([ref]))
    obj, usage, error = call(llm)
    assert obj is None and "refusal" in error and "cyber" in error
    assert usage["cost_usd"] == 0.0 and usage["stop_details"]["category"] == "cyber"
    assert llm.usage_log[-1]["error"] == error


def test_refusal_after_partial_output_is_an_error_but_billed(settings):
    """A refusal that already produced text is billed, and its text is never parsed as the answer."""
    ref = (
        200,
        mock_message(
            [{"type": "text", "text": json.dumps(NEWS_OK)}],
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "bio", "explanation": "x"},
        ),
        {},
    )
    t = Transport([ref])
    llm = make_llm(llm_settings(settings), t)
    obj, usage, error = call(llm)
    assert obj is None and "refusal" in error and len(t.sent) == 1  # no retry, no parse
    assert usage["cost_usd"] == pytest.approx((1000 * 5 + 500 * 25) / 1e6) and llm.total_cost_usd == usage["cost_usd"]


def test_max_tokens_escalates_once_then_succeeds(settings):
    trunc = (200, mock_message([{"type": "text", "text": json.dumps(NEWS_OK)[:40]}], stop_reason="max_tokens"), {})
    t = Transport([trunc, ok()])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert error is None and obj is not None
    assert [r["json"]["max_tokens"] for r in t.sent] == [8000, 16000]
    assert usage["attempts"] == 2 and usage["output_tokens"] == 1000  # both attempts are billed


def test_max_tokens_twice_is_an_error(settings):
    trunc = (200, mock_message([{"type": "text", "text": "{"}], stop_reason="max_tokens"), {})
    t = Transport([trunc, trunc])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and "max_tokens" in error and len(t.sent) == 2 and usage["cost_usd"] > 0


def test_validation_failure_resamples_once(settings):
    bad = {**NEWS_OK, "articles": [{**NEWS_OK["articles"][0], "event_type": "not-an-event"}]}
    t = Transport([ok(bad), ok()])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert error is None and obj is not None and usage["attempts"] == 2 and "validation_errors" not in usage


def test_validation_failure_twice_is_an_error(settings):
    t = Transport([ok({"articles": []}), ok("not json")])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and "스키마 검증 실패" in error and len(t.sent) == 2
    assert usage["validation_errors"]


def test_last_text_block_is_parsed(settings):
    body = mock_message([{"type": "text", "text": "Here is the JSON:"}, {"type": "text", "text": json.dumps(NEWS_OK)}])
    obj, _, error = call(make_llm(llm_settings(settings), Transport([(200, body, {})])))
    assert error is None and obj.articles[0].news_id == "a1"


def test_out_of_range_numbers_do_not_cost_a_resample(settings):
    """Ranges are clamped in code after validation, so a sentiment of 1.7 still validates on the wire."""
    wild = {**NEWS_OK, "articles": [{**NEWS_OK["articles"][0], "sentiment": 1.7, "relevance": -0.2}]}
    t = Transport([ok(wild)])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert error is None and obj.articles[0].sentiment == 1.7 and usage["attempts"] == 1


# ------------------------------------------------------------------------------- typed errors


def test_rate_limit_gets_an_outer_retry(settings):
    t = Transport([err(429, "rate_limit_error", "slow down", {"retry-after": "0"}), ok()])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert error is None and obj is not None and len(t.sent) == 2


def test_rate_limit_exhausted_is_an_error(settings):
    r429 = err(429, "rate_limit_error", "slow down", {"retry-after": "0"})
    t = Transport([r429, r429])
    obj, _, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and "429" in error and len(t.sent) == 2


def test_timeout_is_classified_before_connection_error(settings):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    t = Transport([httpx2.ReadTimeout("read timed out", request=req)])
    obj, _, error = call(make_llm(llm_settings(settings), t, outer_retries=0))
    assert obj is None and "timeout" in error


def test_connection_error_after_timeout_retry(settings):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    t = Transport([httpx2.ReadTimeout("slow", request=req), httpx2.ConnectError("refused", request=req)])
    obj, _, error = call(make_llm(llm_settings(settings), t, outer_retries=1))
    assert obj is None and "네트워크" in error and len(t.sent) == 2


def test_server_error_then_success(settings):
    t = Transport([err(529, "overloaded_error", "overloaded", {"retry-after": "0"}), ok()])
    obj, _, error = call(make_llm(llm_settings(settings), t))
    assert error is None and obj is not None


def test_auth_error_is_not_retried(settings):
    t = Transport([err(401, "authentication_error", "invalid x-api-key")])
    obj, usage, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and "401" in error and len(t.sent) == 1 and usage["cost_usd"] == 0.0


def test_billing_402_is_a_generic_status_error(settings):
    t = Transport([err(402, "billing_error", "credit balance too low")])
    obj, _, error = call(make_llm(llm_settings(settings), t))
    assert obj is None and "402" in error


def test_retry_delay_honours_headers_and_clamps():
    def e(headers):
        resp = httpx2.Response(429, headers=headers, request=httpx2.Request("POST", "https://x"))
        return anthropic.RateLimitError("x", response=resp, body=None)

    assert retry_delay(e({"retry-after-ms": "250"}), 0) == pytest.approx(0.25)
    assert retry_delay(e({"retry-after": "-1"}), 0) == 0.0
    assert retry_delay(e({"retry-after": "999"}), 0) == 60.0
    assert 2.0 <= retry_delay(e({}), 0) <= 3.0


# ------------------------------------------------------------------------------------ cost


def test_cost_accounting_with_cache_fields(settings):
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_creation_input_tokens": 3000,
        "cache_read_input_tokens": 2000,
        "cache_creation": {"ephemeral_5m_input_tokens": 3000, "ephemeral_1h_input_tokens": 0},
    }
    t = Transport([ok(usage=usage)])
    llm = make_llm(llm_settings(settings), t)

    async def twice():
        a = await llm.call_structured("news", NEWS_SYSTEM, "U1", NewsWire)
        b = await llm.call_structured("news", NEWS_SYSTEM, "U2", NewsWire)
        return a, b

    (_, u1, _), (_, u2, _) = asyncio.run(twice())
    # $5/M in, $25/M out, 5m cache write x1.25, cache read x0.1
    expected = (1000 * 5 + 3000 * 5 * 1.25 + 2000 * 5 * 0.1 + 500 * 25) / 1e6
    assert u1["cost_usd"] == pytest.approx(expected) == pytest.approx(0.03725)
    assert llm.total_cost_usd == pytest.approx(2 * expected)
    assert u2["cache_read_input_tokens"] == 2000


def test_fallback_served_model_priced_at_its_rates_and_noticed(settings):
    body = mock_message(
        [
            {
                "type": "fallback",
                "from": {"model": "claude-opus-5"},
                "to": {"model": "claude-sonnet-5"},
                "trigger": {"type": "refusal", "category": "cyber"},
            },
            {"type": "text", "text": json.dumps(NEWS_OK)},
        ],
        model="claude-sonnet-5",
        usage={
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "iterations": [
                {
                    "type": "message",
                    "model": "claude-opus-5",
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                {
                    "type": "fallback_message",
                    "model": "claude-sonnet-5",
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
            ],
        },
    )
    obj, usage, error = call(make_llm(llm_settings(settings), Transport([(200, body, {})])))
    assert error is None and obj is not None
    assert usage["model_served"] == "claude-sonnet-5" and usage["fallback_used"]
    # served at sonnet-5 ($2/$10) + the opus-5 hop that declined after 100 output tokens ($5/$25)
    assert usage["cost_usd"] == pytest.approx((1000 * 2 + 500 * 10) / 1e6 + (1000 * 5 + 100 * 25) / 1e6)
    assert any("claude-sonnet-5" in n for n in usage["notices"])


def test_unknown_model_price_is_flagged():
    class R:  # minimal duck-typed response
        model = "claude-mystery-9"
        stop_reason = "end_turn"
        stop_details = None
        content = []

        class usage:  # noqa: N801
            input_tokens, output_tokens = 1_000_000, 0
            cache_read_input_tokens = cache_creation_input_tokens = 0

    d = usage_dict(R(), "claude-opus-5")
    assert d["pricing_assumed"] and d["cost_usd"] == pytest.approx(5.0)
    assert estimate_cost({**d, "stop_reason": "refusal"}, has_output=False) == 0.0


# ------------------------------------------------------------------------ cache / concurrency


def test_response_cache_hit_skips_the_api(settings, tmp_path):
    s = llm_settings(settings, response_cache=True)
    t = Transport([ok()])
    llm = make_llm(s, t, cache=DiskCache(tmp_path))

    async def run():
        a = await llm.call_structured("news", NEWS_SYSTEM, "SAME", NewsWire)
        b = await llm.call_structured("news", NEWS_SYSTEM, "SAME", NewsWire)
        c = await llm.call_structured("news", NEWS_SYSTEM, "SAME", NewsWire, effort="high")  # new key
        return a, b, c

    (o1, u1, _), (o2, u2, _), (_, u3, _) = asyncio.run(run())
    assert o1 == o2 and len(t.sent) == 2
    assert not u1["cached"] and u2["cached"] and u2["cost_usd"] == 0.0 and not u3["cached"]
    assert t.sent[1]["json"]["output_config"]["effort"] == "high"


def test_failed_calls_are_not_cached(settings, tmp_path):
    s = llm_settings(settings, response_cache=True)
    t = Transport([err(400, "invalid_request_error", "bad"), ok()])
    llm = make_llm(s, t, cache=DiskCache(tmp_path))

    async def run():
        return (
            await llm.call_structured("news", NEWS_SYSTEM, "K", NewsWire),
            await llm.call_structured("news", NEWS_SYSTEM, "K", NewsWire),
        )

    (o1, _, e1), (o2, u2, e2) = asyncio.run(run())
    assert o1 is None and e1 and o2 is not None and not u2["cached"] and len(t.sent) == 2


def test_concurrency_semaphore_caps_inflight_requests(settings):
    t = Transport([ok()], delay=0.05)
    llm = make_llm(llm_settings(settings, concurrency=2), t)

    async def run():
        return await asyncio.gather(*(llm.call_structured("news", NEWS_SYSTEM, f"U{i}", NewsWire) for i in range(6)))

    results = asyncio.run(run())
    assert all(r[2] is None for r in results)
    assert t.peak == 2 and len(t.sent) == 6


# ------------------------------------------------------------------------------- preflight


def _no_credentials_env(monkeypatch, home):
    for k in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(home))  # no ~/.config/anthropic profile


def test_credential_preflight_raises_early(settings, monkeypatch, tmp_path):
    _no_credentials_env(monkeypatch, tmp_path)
    s = llm_settings(settings)
    with pytest.raises(LLMConfigError):
        LLMClient(s)
    with pytest.raises(LLMConfigError):
        LLMClient(s, api_key="")  # an explicit empty key is caught too
    with pytest.raises(LLMConfigError):
        LLMClient(s, anthropic_client=anthropic.AsyncAnthropic())


def test_broken_profile_config_is_a_config_error(settings, monkeypatch, tmp_path):
    _no_credentials_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ANTHROPIC_PROFILE", "default")
    with pytest.raises(LLMConfigError):
        LLMClient(llm_settings(settings))


def test_max_tokens_above_nonstreaming_limit_is_a_config_error(settings):
    s = llm_settings(settings)
    s.llm.max_tokens = {**s.llm.max_tokens, "critic": 30000}
    with pytest.raises(LLMConfigError):
        make_llm(s, Transport([ok()]))


# ------------------------------------------------------------------------------ web search


def test_web_search_pause_turn_sources_and_cost(settings):
    paused = mock_message(
        [
            {"type": "text", "text": "검색 중. "},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "AAPL earnings call"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://www.fool.com/a",
                        "title": "Apple Q3 call",
                        "encrypted_content": "Eq..",
                        "page_age": "2026-07-31",
                    }
                ],
            },
        ],
        stop_reason="pause_turn",
        usage={
            "input_tokens": 5000,
            "output_tokens": 100,
            "server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0},
        },
    )
    done = mock_message(
        [
            {
                "type": "text",
                "text": "매출은 $94,930 million이었다.",
                "citations": [
                    {
                        "type": "web_search_result_location",
                        "url": "https://www.fool.com/a",
                        "title": "Apple Q3 call",
                        "encrypted_index": "Eo..",
                        "cited_text": "Revenue was $94,930 million, up 6% year over year.",
                    }
                ],
            },
        ],
        usage={
            "input_tokens": 6000,
            "output_tokens": 300,
            "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
        },
    )
    t = Transport([(200, paused, {}), (200, done, {})])
    llm = make_llm(llm_settings(settings), t)
    text, sources, usage, error = asyncio.run(
        llm.web_search("web_research", "SYS", "Q", allowed_domains=["fool.com", "reuters.com"], max_uses=3)
    )
    assert error is None and "94,930" in text
    tool = t.sent[0]["json"]["tools"][0]
    assert tool == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 3,
        "allowed_domains": ["fool.com", "reuters.com"],
    }
    assert t.sent[1]["json"]["tools"] == [tool]  # same tools on continuation
    assert t.sent[1]["json"]["messages"][-1]["role"] == "assistant"
    assert t.sent[0]["headers"]["anthropic-beta"] == FALLBACK_BETA
    assert len(sources) == 1 and sources[0]["cited_text"] == ["Revenue was $94,930 million, up 6% year over year."]
    tokens = (5000 + 6000) * 5 + (100 + 300) * 25
    assert usage["cost_usd"] == pytest.approx(tokens / 1e6 + 0.01)
    assert usage["web_search_requests"] == 1


def test_web_search_tool_error_object_is_a_notice(settings):
    body = mock_message(
        [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "s1",
                "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
            },
            {"type": "text", "text": "unknown"},
        ]
    )
    text, sources, usage, error = asyncio.run(
        make_llm(llm_settings(settings), Transport([(200, body, {})])).web_search("web_research", "S", "Q")
    )
    assert error is None and sources == [] and any("max_uses_exceeded" in n for n in usage["notices"])


# ------------------------------------------------------------------------------ fake client


def test_fake_client_modes():
    fake = FakeLLMClient(
        {"news": [NEWS_OK, "boom"], "filings": {"bad": True}, "critic": RuntimeError("x")}, cost_per_call=0.01
    )

    async def run():
        a = await fake.call_structured("news", "S", "U", NewsWire)
        b = await fake.call_structured("news", "S", "U", NewsWire)
        c = await fake.call_structured("news", "S", "U", NewsWire)  # list exhausted
        d = await fake.call_structured("filings", "S", "U", FilingsWire)
        with pytest.raises(RuntimeError):
            await fake.call_structured("critic", "S", "U", NewsWire)
        return a, b, c, d

    a, b, c, d = asyncio.run(run())
    assert isinstance(a[0], NewsWire) and a[2] is None
    assert b[0] is None and b[2] == "boom"
    assert c[0] is None and "응답" in c[2]
    assert d[0] is None and "스키마 검증 실패" in d[2]
    assert [x.task for x in fake.calls] == ["news", "news", "news", "filings", "critic"]
    assert fake.total_cost_usd == pytest.approx(0.04)


def test_fake_client_callable_and_web():
    seen = []

    def responder(task, system, user, wire_model):
        seen.append((task, user))
        return (
            ("요약", [{"url": "u", "title": "t", "cited_text": ["c"]}]) if wire_model is None else NewsWire(**NEWS_OK)
        )

    fake = FakeLLMClient(responder)
    obj, _, e1 = asyncio.run(fake.call_structured("news", "S", "U1", NewsWire))
    text, sources, _, e2 = asyncio.run(fake.web_search("web_research", "S", "U2"))
    assert obj.articles[0].news_id == "a1" and e1 is None and e2 is None
    assert text == "요약" and sources[0]["cited_text"] == ["c"] and seen == [("news", "U1"), ("web_research", "U2")]


def test_recorder_dedups_and_accumulates():
    rec = LLMRecorder()
    rec.record("news", {"cost_usd": 0.1, "notices": ["n1"]}, None)
    rec.record("news", {"cost_usd": 0.2, "notices": ["n1"]}, "실패 사유")
    assert rec.cost_usd == pytest.approx(0.3)
    assert rec.notices == ["뉴스 감성: n1"]  # degraded-but-served -> notices
    assert rec.errors == ["뉴스 감성 실패: 실패 사유"]  # hard failure -> errors


def test_recorder_channels_are_separate_and_logged(caplog):
    rec = LLMRecorder()
    with caplog.at_level("WARNING", logger="quantcio.llm.client"):
        rec.notice("n")
        rec.notice("n")
        rec.error("e")
        rec.error("")  # empty messages are ignored, not recorded
    assert rec.notices == ["n"] and rec.errors == ["e"]
    assert [r.levelname for r in caplog.records] == ["WARNING", "ERROR"]  # never silent


# ------------------------------------------------------------------------------------ live


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
def test_live_structured_news_call(settings):
    s = llm_settings(settings)

    async def run():
        async with LLMClient(s) as llm:
            user = (
                '대상 기업: 삼성전자 (티커 005930)\n\n<document id="a1" published="2026-09-23">\n'
                "삼성전자 3분기 영업이익 10조원 전망\n증권가는 3분기 영업이익이 10조원을 넘길 것으로 본다.\n</document>"
            )
            return await llm.call_structured("news", NEWS_SYSTEM, user, NewsWire)

    obj, usage, error = asyncio.run(run())
    assert error is None, error
    assert obj.articles and obj.articles[0].news_id == "a1"
    assert usage["cost_usd"] > 0


def test_workspace_header_from_env(monkeypatch):
    """Org-level keys need anthropic-workspace-id on every request; it comes from ANTHROPIC_WORKSPACE_ID."""
    from quantcio.llm.client import WORKSPACE_HEADER, make_async_client, with_workspace_header

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    assert with_workspace_header({}) == {}
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test123")
    assert with_workspace_header({})["default_headers"] == {WORKSPACE_HEADER: "wrkspc_test123"}
    explicit = {"default_headers": {"Anthropic-Workspace-Id": "wrkspc_explicit"}}
    assert with_workspace_header(explicit)["default_headers"] == {"Anthropic-Workspace-Id": "wrkspc_explicit"}
    client = make_async_client(api_key="sk-ant-test")
    assert client.default_headers.get(WORKSPACE_HEADER) == "wrkspc_test123"


# ------------------------------------------------------------------------ English output (report.language)

HANGUL = re.compile("[가-힣]")
FAILURE = re.compile(r"error|fail|refus|max_tokens|timeout|invalid", re.I)  # report/render.py's LLM failure words
# sha1 of the {json path: description} map of each Korean wire schema (unchanged since before the English twins)
KO_SCHEMA_DESCRIPTIONS_SHA1 = {
    NewsWire: "831ea349621260fd95af4caf5c1bfda084d313aa",
    FilingsWire: "f8aed9d948d3afcbac1308bb1c48d576bec544a2",
    CriticWire: "fe0ef1a20660f7b17136d09b32c038a6cbdffc11",
}


def en_settings(base: Settings, **llm: Any) -> Settings:
    s = llm_settings(base, **llm)
    s.report.language = "en"
    return s


def _descriptions(node: Any, path: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "description" and isinstance(v, str):
                out[path] = v
            else:
                out.update(_descriptions(v, f"{path}/{k}"))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.update(_descriptions(v, f"{path}/{i}"))
    return out


def _shape(node: Any) -> Any:
    """The schema without descriptions / titles and with the EN model names folded onto the Korean ones."""
    if isinstance(node, dict):
        return {k.replace("WireEN", "Wire"): _shape(v) for k, v in node.items() if k not in ("description", "title")}
    if isinstance(node, list):
        return [_shape(v) for v in node]
    return node.replace("WireEN", "Wire") if isinstance(node, str) else node


@pytest.mark.parametrize("ko,en", [(NewsWire, NewsWireEN), (FilingsWire, FilingsWireEN), (CriticWire, CriticWireEN)])
def test_english_wire_schemas_differ_from_korean_only_in_descriptions(ko, en):
    desc = _descriptions(schema_for(ko))
    assert (
        hashlib.sha1(json.dumps(desc, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        == (KO_SCHEMA_DESCRIPTIONS_SHA1[ko])
    )
    assert any(HANGUL.search(d) for d in desc.values())  # the Korean schema still asks for Korean
    en_schema = schema_for(en)
    assert not HANGUL.search(json.dumps(en_schema, ensure_ascii=False))  # never asks for Korean
    assert "English" in json.dumps(en_schema)
    assert _shape(en_schema) == _shape(schema_for(ko))  # same fields, order, types and required keys
    assert list(en.model_fields) == list(ko.model_fields) and issubclass(en, ko)
    task = {NewsWire: "news", FilingsWire: "filings", CriticWire: "critic"}[ko]
    assert wire_model(task) is ko and wire_model(task, "ko") is ko and wire_model(task, "en") is en
    assert schema_digest(en) != schema_digest(ko)  # distinct response-cache keys per language


class EnglishRouter(Router):
    """``Router`` keyed by the English system prompts, with English descriptive output."""

    def __init__(self, md, **kw: Any):
        super().__init__(md, **kw)
        news, filings, critic = (self.payloads[p] for p in (NEWS_SYSTEM, FILINGS_SYSTEM, CRITIC_SYSTEM))
        filings = ("filings", {**filings[1], "earnings_highlights": ["Revenue KRW 79.1 trillion, up 12.3% YoY"]})
        critic = ("critic", {**critic[1], "thesis": "The trend is firm."})
        self.payloads = {NEWS_SYSTEM_EN: news, FILINGS_SYSTEM_EN: filings, CRITIC_SYSTEM_EN: critic}


@pytest.mark.parametrize("scenario", ["clean", "fallback_served", "beta_rejected"])
def test_run_llm_analysis_in_english_end_to_end(settings, scenario):
    """report.language="en": English system prompts + English wire schemas + English headers on the real wire, and
    every degraded-but-served notice is English."""
    s = en_settings(settings)
    md = make_market_data(Market.KR)
    router = EnglishRouter(
        md, reject_beta=scenario == "beta_rejected", fallback_for="filings" if scenario == "fallback_served" else None
    )
    llm = make_llm(s, Transport([router]))
    out = asyncio.run(
        run_llm_analysis(llm, md, _mini_quant(), s, lambda p: _mini_prelim(), now=datetime(2026, 9, 24, tzinfo=UTC))
    )
    assert out.news is not None and out.filings is not None and out.critic is not None, out.errors
    assert out.errors == []
    assert out.filings.earnings_highlights == ["Revenue KRW 79.1 trillion, up 12.3% YoY"]  # grounded in Korean
    assert out.critic.thesis == "The trend is firm."
    systems = {"news": (NEWS_SYSTEM_EN, NewsWireEN), "filings": (FILINGS_SYSTEM_EN, FilingsWireEN)}
    systems["critic"] = (CRITIC_SYSTEM_EN, CriticWireEN)
    for task, reqs in router.by_task.items():
        system, wire = systems[task]
        kw = dict(system=system, wire_model=wire, effort=s.llm.effort[task], max_tokens=s.llm.max_tokens[task])
        assert_verified_wire(reqs[-1], fallbacks=scenario != "beta_rejected", **kw)
        user = reqs[-1]["json"]["messages"][0]["content"]
        assert user.startswith("Company: 삼성전자 (ticker 005930, KOSPI, market KR, sector 반도체)\n")
        assert "대상 기업" not in user and "지시가 아니다" not in user
    assert "Reference date: 2026-09-24 (UTC)" in router.by_task["news"][-1]["json"]["messages"][0]["content"]

    if scenario == "clean":
        assert out.notices == []
    elif scenario == "fallback_served":
        assert out.notices == [
            "Filings analysis: server fallback model claude-sonnet-5 answered instead of the requested model "
            "claude-opus-5"
        ]
    else:
        assert any(
            n.startswith(("News sentiment: server fallback beta rejected", "Filings analysis: server fallback beta"))
            for n in out.notices
        )
        assert any(n.startswith("Critic: server fallbacks disabled (beta header rejected: ") for n in out.notices)
    assert not any(HANGUL.search(n) or FAILURE.search(n) for n in out.notices), out.notices


def _stop(stop_reason: str, content: list[dict] | None = None, **kw: Any) -> tuple[int, dict, dict]:
    return 200, mock_message(content if content is not None else [], stop_reason=stop_reason, **kw), {}


REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
R0 = {"retry-after": "0"}


@pytest.mark.parametrize(
    "steps,expected",
    [
        ([err(400, "invalid_request_error", "max_tokens: must be positive")], "invalid request (400): max_tokens: m"),
        ([err(400, "invalid_request_error", "API key is not scoped to a workspace")], "invalid request (400): the API"),
        ([err(401, "authentication_error", "invalid x-api-key")], "authentication/permission error (401): invalid"),
        ([err(402, "billing_error", "credit balance too low")], "API error (402): credit balance too low"),
        ([err(404, "not_found_error", "model: claude-x")], "model or endpoint not found (404 error): model: claude-x"),
        ([err(413, "request_too_large", "too big")], "request too large (413 error): too big"),
        ([err(429, "rate_limit_error", "slow", R0)] * 2, "rate limit exceeded (429)"),
        ([err(500, "api_error", "boom", R0)] * 2, "server error (500)"),
        ([httpx2.ReadTimeout("read timed out", request=REQ)], "response timed out (timeout)"),
        ([httpx2.ConnectError("refused", request=REQ)], "network connection error: "),
        ([_stop("refusal", stop_details={"type": "refusal", "category": "cyber"})], "model refused (refusal, cyber)"),
        ([_stop("max_tokens", [{"type": "text", "text": "{"}])] * 2, "output token limit exceeded (max_tokens=16000)"),
        ([ok({"articles": []}), ok("not json")], "schema validation failed: "),
        ([_stop("model_context_window_exceeded")], "context window error: input too long"),
        ([_stop("end_turn", [{"type": "thinking", "thinking": "", "signature": "s"}])], "invalid response: no text"),
    ],
)
def test_client_errors_in_english(settings, steps, expected):
    # outer_retries=0: a transient error is final at once (no backoff sleep between attempts)
    obj, usage, error = call(make_llm(en_settings(settings), Transport(list(steps)), outer_retries=0))
    assert obj is None and error.startswith(expected) and not HANGUL.search(error), error
    rec = LLMRecorder(lang="en")
    rec.record("news", usage, error)
    assert rec.errors == [f"News sentiment failed: {error}"] and FAILURE.search(rec.errors[0])
    if "429" not in error:  # the one exact text given by the spec ("rate limit exceeded (429)")
        assert FAILURE.search(error)


def test_client_errors_stay_korean_by_default(settings):
    _, _, e429 = call(make_llm(llm_settings(settings), Transport([err(429, "rate_limit_error", "slow", R0)] * 2)))
    _, _, e500 = call(make_llm(llm_settings(settings), Transport([err(500, "api_error", "x", R0)] * 2)))
    assert (e429, e500) == ("요청 한도 초과(429)", "서버 오류(500)")


def test_client_notices_in_english(settings, tmp_path):
    s = en_settings(settings)
    rej = err(400, "invalid_request_error", f"Unexpected value(s) `{FALLBACK_BETA}` for the `anthropic-beta` header.")
    llm = make_llm(s, Transport([rej, ok(), ok()]))

    async def two_calls():
        return (
            await llm.call_structured("news", NEWS_SYSTEM_EN, "U1", NewsWireEN),
            await llm.call_structured("news", NEWS_SYSTEM_EN, "U2", NewsWireEN),
        )

    (_, u1, e1), (_, u2, e2) = asyncio.run(two_calls())
    assert e1 is None and e2 is None
    assert u1["notices"] == [
        f"server fallback beta rejected → retried without fallbacks (Unexpected value(s) `{FALLBACK_BETA}` for the "
        "`anthropic-beta` header.)"
    ]
    assert u2["notices"] == [
        f"server fallbacks disabled (beta header rejected: Unexpected value(s) `{FALLBACK_BETA}` for the "
        "`anthropic-beta` header.)"
    ]
    assert llm.notices == [f"server fallback beta ({FALLBACK_BETA}) rejected — retrying without fallbacks"]

    # a fallback model served: noticed, and the notice survives the response cache
    fb = [
        {
            "type": "fallback",
            "from": {"model": "claude-opus-5"},
            "to": {"model": "claude-sonnet-5"},
            "trigger": {"type": "refusal", "category": "cyber"},
        },
        {"type": "text", "text": json.dumps(NEWS_OK)},
    ]
    fb_step = _stop("end_turn", fb, model="claude-sonnet-5")
    cached_llm = make_llm(en_settings(settings, response_cache=True), Transport([fb_step]), cache=DiskCache(tmp_path))
    served = "server fallback model claude-sonnet-5 answered instead of the requested model claude-opus-5"
    first = call(cached_llm, user="SAME")[1]
    second = call(cached_llm, user="SAME")[1]
    assert first["notices"] == [served] and second["cached"] and second["notices"] == [served]

    # a fallback path on the requested model, an unpriced served model, and an unpriced configured model
    same_model = _stop("end_turn", fb, model="claude-opus-5")
    assert call(make_llm(s, Transport([same_model])))[1]["notices"] == [
        "server fallback path used (see usage.iterations)"
    ]
    mystery = call(make_llm(s, Transport([ok(model="claude-mystery-9")])))[1]
    assert mystery["notices"] == [
        "no price for response model claude-mystery-9 → cost estimated at claude-opus-5 rates"
    ]
    unpriced = en_settings(settings, model="claude-mystery-9")
    assert make_llm(unpriced, Transport([ok()])).notices == [
        "no price table for model claude-mystery-9 — estimating cost at claude-opus-5 rates"
    ]
    for n in (*u1["notices"], *u2["notices"], *llm.notices, served, *mystery["notices"]):
        assert not HANGUL.search(n) and not FAILURE.search(n), n


def test_cache_write_failure_notice_in_english(settings, tmp_path):
    class BrokenCache(DiskCache):
        def set(self, *a: Any, **kw: Any) -> None:
            raise OSError("disk full")

    llm = make_llm(en_settings(settings, response_cache=True), Transport([ok()]), cache=BrokenCache(tmp_path))
    obj, usage, error = call(llm)
    assert obj is not None and error is None
    assert usage["notices"] == ["response cache write failed: OSError: disk full"]


def test_web_search_messages_in_english(settings):
    tool_err = mock_message(
        [
            {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "q"}},
            {
                "type": "web_search_tool_result",
                "tool_use_id": "s1",
                "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"},
            },
            {"type": "text", "text": "unknown"},
        ]
    )
    s = en_settings(settings)
    _, _, usage, error = asyncio.run(make_llm(s, Transport([(200, tool_err, {})])).web_search("web_research", "S", "Q"))
    assert error is None and usage["notices"] == ["web search tool error: max_uses_exceeded"]
    paused = _stop("pause_turn", [{"type": "text", "text": "searching"}])
    _, _, _, error = asyncio.run(
        make_llm(s, Transport([paused])).web_search("web_research", "S", "Q", max_continuations=0)
    )
    assert error == "web search failed to finish after 0 continuations (pause_turn)"


def test_fake_client_and_recorder_in_english(settings):
    s = en_settings(settings)
    fake = FakeLLMClient({"filings": {"bad": True}}, settings=s)

    async def run():
        return (
            await fake.call_structured("news", "S", "U", NewsWireEN),
            await fake.call_structured("filings", "S", "U", FilingsWireEN),
            await fake.web_search("web_research", "S", "U"),
        )

    (_, _, e1), (_, _, e2), (_, _, _, e3) = asyncio.run(run())
    assert e1 == "FakeLLMClient error: no canned response for 'news'"
    assert e2.startswith("schema validation failed: missing at ")
    assert e3 == "FakeLLMClient error: no canned response for 'web_research'"

    rec = LLMRecorder(lang=s)
    assert rec.lang == "en" and LLMRecorder().lang == "ko" and LLMRecorder(lang="fr").lang == "ko"
    for task, label in (
        ("news", "News sentiment"),
        ("filings", "Filings analysis"),
        ("critic", "Critic"),
        ("web_research", "Web research"),
    ):
        rec.record(task, {"cost_usd": 0.1, "notices": ["n"]}, "e")
        assert f"{label}: n" in rec.notices and f"{label} failed: e" in rec.errors
        assert task_label(task, "en") == label == TASK_LABELS_EN[task] and task_label(task) == TASK_LABELS[task]
    assert rec.cost_usd == pytest.approx(0.4)


def test_usage_summary_sums_tokens_per_task():
    from types import SimpleNamespace

    log_rows = [
        {
            "task": "news",
            "cached": False,
            "input_tokens": 1000,
            "cache_read_input_tokens": 2000,
            "output_tokens": 300,
            "thinking_tokens": 100,
            "cost_usd": 0.0135,
        },
        {"task": "news", "cached": True, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        {
            "task": "critic",
            "cached": False,
            "input_tokens": 5000,
            "output_tokens": 800,
            "thinking_tokens": 500,
            "cost_usd": 0.045,
        },
    ]
    s = LLMClient.usage_summary(SimpleNamespace(usage_log=log_rows))
    assert s["news"]["calls"] == 2 and s["news"]["cached"] == 1 and s["news"]["input_tokens"] == 1000
    assert s["total"]["input_tokens"] == 6000 and s["total"]["thinking_tokens"] == 600
    assert s["total"]["cost_usd"] == pytest.approx(0.0585)


def test_cli_prints_llm_usage_line(capsys):
    import quantcio.cli as cli

    usage = {
        "news": {
            "calls": 2,
            "cached": 1,
            "input_tokens": 1000,
            "cache_read_input_tokens": 0,
            "output_tokens": 300,
            "thinking_tokens": 0,
            "cost_usd": 0.01,
        },
        "total": {
            "calls": 2,
            "cached": 1,
            "input_tokens": 1000,
            "cache_read_input_tokens": 0,
            "output_tokens": 300,
            "thinking_tokens": 0,
            "cost_usd": 0.01,
        },
    }
    cli._print_llm_usage(usage)
    cli._print_llm_usage({})
    out = capsys.readouterr().out
    assert "LLM 토큰" in out and "1,000" in out and "$0.010" in out and out.count("LLM 토큰") == 1
