"""Async Claude client for the QuantCIO LLM layer: structured outputs, refusal fallbacks, cost, cache.

Ported from the verified ``docs/sources/reference/llm_client_reference.py`` (anthropic 1.8 on httpx2).

- ``client.beta.messages.create`` with ``output_config={"effort", "format": json_schema(transform_schema(W))}``,
  adaptive thinking, and a single ``cache_control: ephemeral`` system block. It deliberately avoids
  ``messages.parse()``, which raises (and loses usage and cost) on a refusal with text, on truncated
  JSON, or on more than one text block.
- Server-side refusal fallback: ``betas=["server-side-fallback-2026-07-01"]`` plus
  ``fallbacks="default"`` when ``settings.llm.server_fallbacks``. If a 400 rejects the beta, the call
  is retried once without it, and fallbacks stay off for the rest of the client's life. This shows
  up as a notice.
- ``stop_reason`` is checked before any content: ``refusal`` and ``model_context_window_exceeded`` are
  errors, and ``max_tokens`` gets one escalation. Only then is the LAST text block validated with
  pydantic, with one re-sample on ``ValidationError``.
- Typed exception chain: BadRequest -> Auth/Permission -> NotFound -> RequestTooLarge -> RateLimit ->
  Overloaded/InternalServer -> APIStatusError -> APITimeoutError -> APIConnectionError. Transient
  errors get ``outer_retries`` extra attempts after the SDK's own retries.
- Cost: per-model $/MTok, cache write x1.25 (5m) / x2 (1h), cache read x0.1, priced at ``resp.model``
  (the model that actually served a fallback), plus billed hops that declined after output.
- Disk response cache (``DiskCache`` namespace ``"llm"``). The key hashes model, system, user, schema
  name (+ schema digest) and effort.

Failures never raise out of ``call_structured`` / ``web_search`` for API problems. They return an
error string (user-facing, in ``settings.report.language``: Korean by default, English for "en") together
with the usage accumulated so far. That string becomes ``LLMIntelligence.errors`` through ``LLMRecorder``.
A degraded path that still produced an answer is listed in ``usage["notices"]`` instead, and
``LLMRecorder`` routes it to ``LLMIntelligence.notices``.
Examples: a fallback model served, the fallback beta was rejected, pricing was assumed, or a cache
write failed.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

import anthropic
import pydantic
from anthropic import AsyncAnthropic, transform_schema

from quantcio.config import Settings
from quantcio.data.cache import DiskCache
from quantcio.i18n import lang_of, tr

log = logging.getLogger(__name__)

W = TypeVar("W", bound=pydantic.BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"  # pairs ONLY with fallbacks="default"
BETA_REJECTED_NOTICE = "서버 폴백 베타 거부 → 폴백 없이 재시도"
BETA_REJECTED_NOTICE_EN = "server fallback beta rejected → retried without fallbacks"
MAX_NONSTREAMING_TOKENS = 21_333  # the SDK demands streaming above this with the default timeout
CACHE_NAMESPACE = "llm"
WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
WEB_SEARCH_USD_PER_CALL = 10.0 / 1000

# USD per 1M tokens (input, output). Cache write 5m = 1.25x input, 1h = 2x input, cache read = 0.1x input.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
DEFAULT_PRICED_MODEL = "claude-opus-5"
CACHE_WRITE_5M_MULT = 1.25
CACHE_WRITE_1H_MULT = 2.0
CACHE_READ_MULT = 0.10

TASK_LABELS: dict[str, str] = {
    "news": "뉴스 감성",
    "filings": "공시 분석",
    "critic": "비평",
    "web_research": "웹 리서치",
}
TASK_LABELS_EN: dict[str, str] = {
    "news": "News sentiment",
    "filings": "Filings analysis",
    "critic": "Critic",
    "web_research": "Web research",
}
_DEFAULT_EFFORT = "medium"
_DEFAULT_MAX_TOKENS = 8000


def task_label(task: str, lang: Any = "ko") -> str:
    """User-facing label of an LLM task (``TASK_LABELS`` or ``TASK_LABELS_EN``)."""
    return (TASK_LABELS_EN if lang_of(lang) == "en" else TASK_LABELS).get(task, task)


def en_count(n: int, noun: str, plural: str | None = None) -> str:
    """English count phrase for notices: "1 article" / "3 articles"."""
    return f"{n} {noun if n == 1 else (plural or noun + 's')}"


class LLMConfigError(RuntimeError):
    """The LLM layer cannot run (no credentials, bad limits). Raised early, before any ticker is processed."""


# ================================================================================ clients


def assert_credentials(client: AsyncAnthropic | anthropic.Anthropic) -> None:
    """Preflight. With nothing configured, ``AsyncAnthropic()`` still constructs, and the first request
    raises a builtin TypeError that is not an AnthropicError. ``api_key=""`` stays "", so check falsiness."""
    if not client.api_key and not client.auth_token and client.credentials is None:
        raise LLMConfigError(
            "Anthropic 자격 증명이 없습니다: ANTHROPIC_API_KEY(또는 ANTHROPIC_AUTH_TOKEN / `ant auth login` 프로필)를 "
            "설정하거나 --no-llm으로 실행하세요."
        )


WORKSPACE_ENV = "ANTHROPIC_WORKSPACE_ID"
WORKSPACE_HEADER = "anthropic-workspace-id"
WORKSPACE_HINT = (
    "API 키가 워크스페이스에 속하지 않습니다 — .env에 ANTHROPIC_WORKSPACE_ID=<워크스페이스 ID>를 추가하거나 "
    "Console에서 워크스페이스 전용 API 키를 발급해 ANTHROPIC_API_KEY를 교체하세요"
)
WORKSPACE_HINT_EN = (
    "the API key is not scoped to a workspace — add ANTHROPIC_WORKSPACE_ID=<workspace ID> to .env, or issue a "
    "workspace-scoped API key in the Console and replace ANTHROPIC_API_KEY"
)


def with_workspace_header(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Add ``anthropic-workspace-id`` from ANTHROPIC_WORKSPACE_ID (org-level keys not scoped to a workspace
    are rejected with 400 unless every request names the workspace). An explicit header in kwargs wins."""
    workspace = os.environ.get(WORKSPACE_ENV, "").strip()
    if not workspace:
        return kwargs
    headers = dict(kwargs.get("default_headers") or {})
    if not any(k.lower() == WORKSPACE_HEADER for k in headers):
        headers[WORKSPACE_HEADER] = workspace
    return {**kwargs, "default_headers": headers}


def is_workspace_error(e: anthropic.BadRequestError) -> bool:
    return "not scoped to a workspace" in api_error_message(e)


def make_async_client(*, max_retries: int = 3, **kwargs: Any) -> AsyncAnthropic:
    """Build an ``AsyncAnthropic`` and fail fast with ``LLMConfigError`` when no credential resolves.

    A broken credential setup (e.g. ``ANTHROPIC_PROFILE`` pointing at a missing config file) makes the SDK
    raise ``CredentialsError`` at construction. That is converted into ``LLMConfigError`` as well.
    """
    kwargs = with_workspace_header(kwargs)
    try:
        client = AsyncAnthropic(max_retries=max_retries, **kwargs)
    except anthropic.CredentialsError as exc:
        raise LLMConfigError(f"Anthropic 자격 증명 설정 오류: {exc}") from exc
    assert_credentials(client)
    return client


# ================================================================================ helpers


@functools.lru_cache(maxsize=64)
def _schema_cached(model_cls: type[pydantic.BaseModel]) -> str:
    return json.dumps(transform_schema(model_cls), sort_keys=True, ensure_ascii=False)


def schema_for(model_cls: type[pydantic.BaseModel]) -> dict[str, Any]:
    """Wire JSON schema (exactly ``transform_schema``), as a fresh copy on every call.

    Deterministic, so it is cache-stable.
    """
    return json.loads(_schema_cached(model_cls))


def schema_digest(model_cls: type[pydantic.BaseModel]) -> str:
    return hashlib.sha1(_schema_cached(model_cls).encode("utf-8")).hexdigest()[:10]


def cached_system(system: str) -> list[dict[str, Any]]:
    """The single, byte-stable system block, marked for prompt caching."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def fallback_kwargs(enabled: bool) -> dict[str, Any]:
    return {"betas": [FALLBACK_BETA], "fallbacks": "default"} if enabled else {}


def is_beta_rejection(e: anthropic.BadRequestError) -> bool:
    msg = (e.message or "").lower()
    return "anthropic-beta" in msg or "fallback" in msg


def api_error_message(e: anthropic.APIError) -> str:
    """The server's own message (``body.error.message``) for display.

    The SDK's ``e.message`` is ``"Error code: 400 - {'type': 'error', ...}"``, a raw dict that must not
    reach the Korean report. Falls back to ``e.message`` for non-JSON bodies (e.g. an HTML 502 page).
    """
    body = getattr(e, "body", None)
    if isinstance(body, Mapping):
        inner = body.get("error")
        for cand in (inner.get("message") if isinstance(inner, Mapping) else None, body.get("message")):
            if isinstance(cand, str) and cand.strip():
                return cand
    return e.message or type(e).__name__


def retry_delay(e: BaseException, attempt: int) -> float:
    """Seconds to wait: honour retry-after-ms / retry-after (clamped to [0, 60]), else exponential backoff."""
    resp = getattr(e, "response", None)
    if resp is not None:
        for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            ra = resp.headers.get(name)
            if ra is None:
                continue
            try:
                return max(0.0, min(float(ra) * scale, 60.0))
            except ValueError:  # HTTP-date form -> fall back to backoff
                continue
    return min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1.0)


def _price(model: str | None) -> tuple[tuple[float, float], bool]:
    """(input, output) $/MTok for ``model`` and whether the default price had to be assumed."""
    if model in PRICING:
        return PRICING[model], False
    return PRICING[DEFAULT_PRICED_MODEL], True


def usage_dict(resp: Any, requested_model: str) -> dict[str, Any]:
    """Flatten a (Beta)Message's usage for cost accounting, including ``cost_usd`` for this one response."""
    u = resp.usage
    cc = getattr(u, "cache_creation", None)
    otd = getattr(u, "output_tokens_details", None)
    stu = getattr(u, "server_tool_use", None)
    iterations = [
        {
            "type": it.type,
            "model": getattr(it, "model", None),
            "input_tokens": it.input_tokens or 0,
            "output_tokens": it.output_tokens or 0,
            "cache_read_input_tokens": getattr(it, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(it, "cache_creation_input_tokens", 0) or 0,
        }
        for it in (getattr(u, "iterations", None) or [])
    ]
    content = list(resp.content or [])
    d: dict[str, Any] = {
        "model_requested": requested_model,
        "model_served": resp.model,
        "fallback_used": any(it["type"] == "fallback_message" for it in iterations)
        or any(getattr(b, "type", None) == "fallback" for b in content),
        "iterations": iterations,
        "input_tokens": u.input_tokens or 0,
        "output_tokens": u.output_tokens or 0,
        "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
        "cache_creation_5m": (getattr(cc, "ephemeral_5m_input_tokens", 0) or 0) if cc else 0,
        "cache_creation_1h": (getattr(cc, "ephemeral_1h_input_tokens", 0) or 0) if cc else 0,
        "thinking_tokens": (getattr(otd, "thinking_tokens", 0) or 0) if otd else 0,
        "web_search_requests": (getattr(stu, "web_search_requests", 0) or 0) if stu else 0,
        "request_id": getattr(resp, "_request_id", None),
        "stop_reason": resp.stop_reason,
    }
    sd = getattr(resp, "stop_details", None)
    if sd is not None:
        d["stop_details"] = {
            "category": getattr(sd, "category", None),
            "explanation": getattr(sd, "explanation", None),
            "recommended_model": getattr(sd, "recommended_model", None),
        }
    d["pricing_assumed"] = _price(resp.model)[1]
    d["cost_usd"] = estimate_cost(
        d,
        has_output=any(getattr(b, "type", None) in ("text", "thinking") for b in content),
    )
    return d


def estimate_cost(d: Mapping[str, Any], *, has_output: bool = True) -> float:
    """USD cost of one response. Top-level usage is billed at the SERVED model's rates, and a refusal before
    any output is free. A fallback hop that declined AFTER producing output is billed but is missing from
    the top-level usage, so it is added here from ``iterations`` at that model's rates."""
    if d["stop_reason"] == "refusal" and not has_output:
        return 0.0
    (pin, pout), _ = _price(d["model_served"])
    five_m = d["cache_creation_5m"] or max(0, d["cache_creation_input_tokens"] - d["cache_creation_1h"])
    usd = (
        d["input_tokens"] * pin
        + five_m * pin * CACHE_WRITE_5M_MULT
        + d["cache_creation_1h"] * pin * CACHE_WRITE_1H_MULT
        + d["cache_read_input_tokens"] * pin * CACHE_READ_MULT
        + d["output_tokens"] * pout
    ) / 1e6
    for it in d.get("iterations", []):
        if it["type"] == "message" and it["model"] and it["model"] != d["model_served"] and it["output_tokens"] > 0:
            (ipin, ipout), _ = _price(it["model"])
            usd += (
                it["input_tokens"] * ipin
                + it["cache_creation_input_tokens"] * ipin * CACHE_WRITE_5M_MULT
                + it["cache_read_input_tokens"] * ipin * CACHE_READ_MULT
                + it["output_tokens"] * ipout
            ) / 1e6
    return round(usd + d["web_search_requests"] * WEB_SEARCH_USD_PER_CALL, 6)


_SUM_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "cache_creation_5m",
    "cache_creation_1h",
    "thinking_tokens",
    "web_search_requests",
)


def empty_usage(model: str) -> dict[str, Any]:
    """Usage record for a call that produced no billable response (error before any reply, or a cache hit)."""
    return {
        "model_requested": model,
        "model_served": None,
        "fallback_used": False,
        "iterations": [],
        **{k: 0 for k in _SUM_KEYS},
        "request_id": None,
        "stop_reason": None,
        "pricing_assumed": False,
        "cost_usd": 0.0,
        "attempts": 0,
        "request_ids": [],
    }


def merge_usage(total: dict[str, Any] | None, d: dict[str, Any]) -> dict[str, Any]:
    """Accumulate one response's usage into the running total of a logical call (retries included)."""
    if total is None or total.get("attempts", 0) == 0:
        return {**d, "attempts": 1, "request_ids": [d.get("request_id")]}
    out = {
        **d,
        "attempts": total["attempts"] + 1,
        "request_ids": total["request_ids"] + [d.get("request_id")],
    }
    for k in _SUM_KEYS:
        out[k] = total.get(k, 0) + d.get(k, 0)
    out["cost_usd"] = round(total.get("cost_usd", 0.0) + d.get("cost_usd", 0.0), 6)
    out["fallback_used"] = bool(total.get("fallback_used")) or bool(d.get("fallback_used"))
    out["pricing_assumed"] = bool(total.get("pricing_assumed")) or bool(d.get("pricing_assumed"))
    return out


def _echoable(content: Sequence[Any]) -> list[Any]:
    """Blocks that are safe to echo back to continue a paused turn. Before the last ``fallback`` block, keep
    only text and paired server-tool blocks (the fallback echo rules)."""
    idx = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    if idx < 0:
        return list(content)
    result_ids = {getattr(b, "tool_use_id", None) for b in content if b.type.endswith("_tool_result")}
    return [
        b
        for i, b in enumerate(content)
        if i > idx
        or b.type == "text"
        or b.type.endswith("_tool_result")
        or (b.type == "server_tool_use" and b.id in result_ids)
    ]


def collect_web_sources(
    blocks: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """One entry per URL: search results merged with citations (``cited_text`` list). Also returns tool errors."""
    by_url: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for b in blocks:
        if b.type == "web_search_tool_result" and isinstance(b.content, list):
            for r in b.content:
                by_url.setdefault(
                    r.url,
                    {
                        "url": r.url,
                        "title": r.title,
                        "page_age": getattr(r, "page_age", None),
                        "cited_text": [],
                    },
                )
        elif b.type == "web_search_tool_result":  # HTTP 200 with an error OBJECT instead of a result list
            errors.append(str(getattr(b.content, "error_code", b.content)))
        elif b.type == "text" and getattr(b, "citations", None):
            for c in b.citations:
                url = getattr(c, "url", None)
                if not url:
                    continue
                entry = by_url.setdefault(
                    url,
                    {
                        "url": url,
                        "title": getattr(c, "title", None),
                        "page_age": None,
                        "cited_text": [],
                    },
                )
                cited = getattr(c, "cited_text", None)
                if cited and cited not in entry["cited_text"]:
                    entry["cited_text"].append(cited)
    return list(by_url.values()), errors


def _short(msg: Any, n: int = 160) -> str:
    return " ".join(str(msg or "").split())[:n]


def schema_error(ve: pydantic.ValidationError, lang: Any = "ko") -> str:
    """User-facing text of a wire-schema validation failure (after the re-sample)."""
    where = f"{ve.errors()[0]['type']} at {ve.errors()[0]['loc']}"
    return tr(lang, f"스키마 검증 실패: {where}", f"schema validation failed: {where}")


# ============================================================================ the protocol


StructuredResult = tuple[W | None, dict[str, Any], str | None]
WebResult = tuple[str, list[dict[str, Any]], dict[str, Any], str | None]


class LLMLike(Protocol):
    """What analysis functions need from a client. Implemented by ``LLMClient`` and ``FakeLLMClient``."""

    async def call_structured(
        self,
        task: str,
        system: str,
        user: str,
        wire_model: type[W],
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> StructuredResult: ...

    async def web_search(
        self,
        task: str,
        system: str,
        user: str,
        *,
        allowed_domains: Sequence[str] | None = None,
        max_uses: int = 5,
        effort: str | None = None,
        max_tokens: int | None = None,
        max_continuations: int = 2,
    ) -> WebResult: ...


# ================================================================================ LLMClient


class LLMClient:
    """Async wrapper around ``AsyncAnthropic`` (see the module docstring for the full behaviour).

    One instance serves every ticker in a run. ``settings.llm.concurrency`` caps the number of HTTP
    requests in flight across all tickers.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        anthropic_client: AsyncAnthropic | None = None,
        cache: DiskCache | None = None,
        outer_retries: int = 1,
        validation_retries: int = 1,
        max_retries: int = 3,
        **client_kwargs: Any,
    ) -> None:
        self.settings = settings
        self.model = settings.llm.model
        too_big = {k: v for k, v in settings.llm.max_tokens.items() if v > MAX_NONSTREAMING_TOKENS}
        if too_big:
            raise LLMConfigError(f"llm.max_tokens {too_big} > {MAX_NONSTREAMING_TOKENS}: 비스트리밍 호출 한도 초과")
        self._client = anthropic_client or make_async_client(max_retries=max_retries, **client_kwargs)
        assert_credentials(self._client)
        if cache is None and settings.llm.response_cache:
            cache = DiskCache(settings.cache_path)
        self.cache = cache if settings.llm.response_cache else None
        self.outer_retries = max(0, outer_retries)
        self.validation_retries = max(0, validation_retries)
        self.concurrency = max(1, settings.llm.concurrency)
        self._sem: asyncio.Semaphore | None = None
        self._sem_loop: asyncio.AbstractEventLoop | None = None
        self._fallbacks_on = settings.llm.server_fallbacks
        self.fallbacks_disabled_reason: str | None = None
        self.total_cost_usd = 0.0
        self.usage_log: list[dict[str, Any]] = []
        self.notices: list[str] = []
        if _price(self.model)[1]:
            self._notice(
                self._t(
                    f"모델 {self.model}의 단가표가 없어 {DEFAULT_PRICED_MODEL} 단가로 비용을 추정합니다",
                    f"no price table for model {self.model} — estimating cost at {DEFAULT_PRICED_MODEL} rates",
                )
            )

    # ------------------------------------------------------------------ lifecycle
    async def aclose(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ small helpers
    @property
    def fallbacks_active(self) -> bool:
        return self._fallbacks_on

    def _t(self, ko: str, en: str) -> str:
        """The Korean or English variant of a user-facing message (``settings.report.language``)."""
        return tr(self.settings, ko, en)

    def _notice(self, msg: str) -> None:
        if msg not in self.notices:
            self.notices.append(msg)
            log.warning(msg)

    def _refusal_error(self, usage: Mapping[str, Any]) -> str:
        cat = (usage.get("stop_details") or {}).get("category")
        detail = f"refusal{', ' + str(cat) if cat else ''}"
        return self._t(f"모델 거절({detail})", f"model refused ({detail})")

    def _max_tokens_error(self, cur_max: int) -> str:
        return self._t(
            f"출력 토큰 한도 초과(max_tokens={cur_max})", f"output token limit exceeded (max_tokens={cur_max})"
        )

    def _semaphore(self) -> asyncio.Semaphore:
        """One semaphore per running event loop (a semaphore bound to a closed loop cannot be reused)."""
        loop = asyncio.get_running_loop()
        if self._sem is None or self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self.concurrency), loop
        return self._sem

    def _effort(self, task: str) -> str:
        return self.settings.llm.effort.get(task, _DEFAULT_EFFORT)

    def _max_tokens(self, task: str) -> int:
        return self.settings.llm.max_tokens.get(task, _DEFAULT_MAX_TOKENS)

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        if self.cache is None:
            return None
        hit = self.cache.get(CACHE_NAMESPACE, key, ttl_hours=None)
        return hit if isinstance(hit, dict) else None

    def _cache_set(self, key: str, value: dict[str, Any], notices: list[str]) -> None:
        if self.cache is None:
            return
        try:
            self.cache.set(CACHE_NAMESPACE, key, value)
        except Exception as exc:  # noqa: BLE001 - a cache write must never fail the analysis
            notices.append(
                self._t(
                    f"응답 캐시 저장 실패: {type(exc).__name__}: {_short(exc, 80)}",
                    f"response cache write failed: {type(exc).__name__}: {_short(exc, 80)}",
                )
            )

    def _finish(
        self,
        task: str,
        usage: dict[str, Any] | None,
        notices: list[str],
        error: str | None,
        cached: bool = False,
    ) -> dict[str, Any]:
        """Final bookkeeping for one logical call: notices, totals, log. Returns the usage dict."""
        u = dict(usage) if usage else empty_usage(self.model)
        if u.get("fallback_used") and u.get("model_served") and u["model_served"] != self.model:
            notices.append(
                self._t(
                    f"요청 모델 {self.model} 대신 서버 폴백 모델 {u['model_served']}이(가) 응답",
                    f"server fallback model {u['model_served']} answered instead of the requested model {self.model}",
                )
            )
        elif u.get("fallback_used"):
            notices.append(
                self._t(
                    "서버 폴백 경로가 사용됨(usage.iterations 참조)", "server fallback path used (see usage.iterations)"
                )
            )
        if u.get("pricing_assumed"):
            notices.append(
                self._t(
                    f"응답 모델 {u.get('model_served')} 단가 미등록 → {DEFAULT_PRICED_MODEL} 단가로 비용 추정",
                    f"no price for response model {u.get('model_served')} → cost estimated at "
                    f"{DEFAULT_PRICED_MODEL} rates",
                )
            )
        if (
            self.settings.llm.server_fallbacks
            and not self._fallbacks_on
            and self.fallbacks_disabled_reason
            and not any(n.startswith((BETA_REJECTED_NOTICE, BETA_REJECTED_NOTICE_EN)) for n in notices)  # already said
        ):
            # every later call (and so every later ticker) still shows that it ran without refusal fallbacks
            notices.append(
                self._t(
                    f"서버 폴백 비활성({self.fallbacks_disabled_reason})",
                    f"server fallbacks disabled ({self.fallbacks_disabled_reason})",
                )
            )
        u["notices"] = list(dict.fromkeys(notices))
        u["cached"] = cached
        u["task"] = task
        u["error"] = error
        self.total_cost_usd = round(self.total_cost_usd + float(u.get("cost_usd", 0.0) or 0.0), 6)
        self.usage_log.append(
            {
                k: u.get(k)
                for k in (
                    "task",
                    "model_requested",
                    "model_served",
                    "attempts",
                    "cost_usd",
                    "fallback_used",
                    "cached",
                    "stop_reason",
                    "error",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "thinking_tokens",
                )
            }
        )
        return u

    def usage_summary(self) -> dict[str, dict[str, float]]:
        """Per task (and ``"total"``): calls, response-cache hits, input / cache-read / cache-write / output /
        thinking tokens and USD, summed over this client's ``usage_log``."""
        keys = (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "output_tokens",
            "thinking_tokens",
            "cost_usd",
        )
        out: dict[str, dict[str, float]] = {}
        for row in self.usage_log:
            for name in (str(row.get("task") or "?"), "total"):
                agg = out.setdefault(name, {"calls": 0, "cached": 0, **{k: 0 for k in keys}})
                agg["calls"] += 1
                agg["cached"] += int(bool(row.get("cached")))
                for k in keys:
                    agg[k] += float(row.get(k) or 0)
        for agg in out.values():
            agg["cost_usd"] = round(agg["cost_usd"], 6)
        return out

    # ------------------------------------------------------------------ transport + typed errors
    async def _send(self, build: Callable[[bool], dict[str, Any]], notices: list[str]) -> tuple[Any, str | None]:
        """Send one logical request: beta-rejection downgrade, then outer retries for transient errors.

        Returns ``(response, None)`` or ``(None, error_message)``. It never raises for API errors.
        """
        outer_left, attempt = self.outer_retries, 0
        while True:
            use_fb = self._fallbacks_on
            exc: BaseException | None = None
            err, retryable = self._t("알 수 없는 오류", "unknown error"), False
            t = self._t
            try:
                async with self._semaphore():
                    return await self._client.beta.messages.create(**build(use_fb)), None
            except anthropic.BadRequestError as e:
                if use_fb and is_beta_rejection(e):
                    reason = _short(api_error_message(e), 100)
                    self._fallbacks_on = False
                    self.fallbacks_disabled_reason = t(f"베타 헤더 거부: {reason}", f"beta header rejected: {reason}")
                    self._notice(
                        t(
                            f"서버 폴백 베타({FALLBACK_BETA})가 거부되어 폴백 없이 재시도합니다",
                            f"server fallback beta ({FALLBACK_BETA}) rejected — retrying without fallbacks",
                        )
                    )
                    notices.append(t(f"{BETA_REJECTED_NOTICE}({reason})", f"{BETA_REJECTED_NOTICE_EN} ({reason})"))
                    continue
                msg = _short(api_error_message(e))
                if is_workspace_error(e):
                    err, exc = (
                        t(f"잘못된 요청(400): {WORKSPACE_HINT}", f"invalid request (400): {WORKSPACE_HINT_EN}"),
                        e,
                    )
                else:
                    err, exc = t(f"잘못된 요청(400): {msg}", f"invalid request (400): {msg}"), e
            except (
                anthropic.AuthenticationError,
                anthropic.PermissionDeniedError,
            ) as e:
                msg = _short(api_error_message(e))
                err, exc = (
                    t(
                        f"인증/권한 오류({e.status_code}): {msg}",
                        f"authentication/permission error ({e.status_code}): {msg}",
                    ),
                    e,
                )
            except anthropic.NotFoundError as e:
                msg = _short(api_error_message(e))
                err, exc = (
                    t(f"모델 또는 엔드포인트 없음(404): {msg}", f"model or endpoint not found (404 error): {msg}"),
                    e,
                )
            except anthropic.RequestTooLargeError as e:
                msg = _short(api_error_message(e))
                err, exc = t(f"요청 크기 초과(413): {msg}", f"request too large (413 error): {msg}"), e
            except anthropic.RateLimitError as e:
                err, exc, retryable = t("요청 한도 초과(429)", "rate limit exceeded (429)"), e, True
            except (anthropic.OverloadedError, anthropic.InternalServerError) as e:
                err, exc, retryable = t(f"서버 오류({e.status_code})", f"server error ({e.status_code})"), e, True
            except anthropic.APIStatusError as e:  # 402 billing, 409, 422, ...
                msg = _short(api_error_message(e))
                err, exc = t(f"API 오류({e.status_code}): {msg}", f"API error ({e.status_code}): {msg}"), e
            except anthropic.APITimeoutError as e:  # subclass of APIConnectionError: must come first
                err, exc, retryable = t("응답 시간 초과(timeout)", "response timed out (timeout)"), e, True
            except anthropic.APIConnectionError as e:
                msg = _short(e.message, 80)
                err, exc, retryable = (
                    t(f"네트워크 연결 오류: {msg}", f"network connection error: {msg}"),
                    e,
                    True,
                )
            except anthropic.APIError as e:  # e.g. APIResponseValidationError
                msg = f"{type(e).__name__}: {_short(api_error_message(e))}"
                err, exc = t(f"API 응답 오류: {msg}", f"API response error: {msg}"), e
            if retryable and outer_left > 0:
                outer_left -= 1
                delay = retry_delay(exc, attempt)
                attempt += 1
                log.warning("LLM %s (SDK 재시도 이후); %.1fs 후 추가 재시도", err, delay)
                await asyncio.sleep(delay)
                continue
            log.warning("LLM 호출 실패: %s", err)
            return None, err

    # ------------------------------------------------------------------ structured call
    def _structured_request(
        self,
        system: str,
        user: str,
        wire_model: type[pydantic.BaseModel],
        effort: str,
        max_tokens: int,
        use_fallbacks: bool,
    ) -> dict[str, Any]:
        return dict(
            model=self.model,
            max_tokens=max_tokens,
            system=cached_system(system),
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema_for(wire_model)},
            },
            **fallback_kwargs(use_fallbacks),
        )

    async def call_structured(
        self,
        task: str,
        system: str,
        user: str,
        wire_model: type[W],
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        """One structured-output call. Returns ``(wire_obj | None, usage, error | None)``.

        ``effort`` and ``max_tokens`` default to ``settings.llm.effort[task]`` and ``settings.llm.max_tokens[task]``.
        """
        effort = effort or self._effort(task)
        cur_max = int(max_tokens or self._max_tokens(task))
        if cur_max > MAX_NONSTREAMING_TOKENS:
            raise LLMConfigError(f"max_tokens={cur_max} > {MAX_NONSTREAMING_TOKENS}: 비스트리밍 한도 초과")
        key = DiskCache.key(
            self.model,
            system,
            user,
            wire_model.__name__,
            schema_digest(wire_model),
            effort,
        )
        notices: list[str] = []

        hit = self._cache_get(key)
        if hit is not None:
            try:
                obj = wire_model.model_validate_json(hit["text"])
            except (pydantic.ValidationError, KeyError, TypeError):
                log.warning("LLM 응답 캐시 항목이 현재 스키마와 맞지 않아 무시합니다 (%s)", task)
            else:
                u = empty_usage(self.model)
                u["model_served"] = hit.get("model_served")
                notices.extend(hit.get("notices", []))
                return obj, self._finish(task, u, notices, None, cached=True), None

        usage: dict[str, Any] | None = None
        validation_left, escalate = self.validation_retries, True
        while True:
            resp, err = await self._send(
                lambda fb: self._structured_request(system, user, wire_model, effort, cur_max, fb),
                notices,
            )
            if resp is None:
                return None, self._finish(task, usage, notices, err), err
            usage = merge_usage(usage, usage_dict(resp, self.model))
            stop = resp.stop_reason or "unknown"

            if stop == "refusal":  # the whole fallback chain refused (or fallbacks are off)
                err = self._refusal_error(usage)
                return None, self._finish(task, usage, notices, err), err
            if stop == "max_tokens":
                if escalate and cur_max < MAX_NONSTREAMING_TOKENS:
                    escalate = False
                    cur_max = min(cur_max * 2, MAX_NONSTREAMING_TOKENS)
                    log.warning(
                        "LLM %s: max_tokens 도달 → max_tokens=%d로 재시도",
                        task,
                        cur_max,
                    )
                    continue
                err = self._max_tokens_error(cur_max)
                return None, self._finish(task, usage, notices, err), err
            if stop == "model_context_window_exceeded":
                err = self._t(
                    "입력이 컨텍스트 한도를 초과(model_context_window_exceeded)",
                    "context window error: input too long (model_context_window_exceeded)",
                )
                return None, self._finish(task, usage, notices, err), err

            texts = [b.text for b in resp.content if b.type == "text"]
            if not texts:
                err = self._t(
                    f"응답에 텍스트 블록 없음(stop_reason={stop})",
                    f"invalid response: no text block (stop_reason={stop})",
                )
                return None, self._finish(task, usage, notices, err), err
            try:
                obj = wire_model.model_validate_json(texts[-1])
            except pydantic.ValidationError as ve:
                usage["validation_errors"] = ve.errors(include_url=False)[:5]
                if validation_left > 0:
                    validation_left -= 1
                    log.warning(
                        "LLM %s: 스키마 검증 실패(%s) → 1회 재샘플",
                        task,
                        ve.errors()[0]["type"],
                    )
                    continue
                err = schema_error(ve, self.settings)
                return None, self._finish(task, usage, notices, err), err
            usage.pop("validation_errors", None)
            final = self._finish(task, usage, notices, None)
            self._cache_set(
                key,
                {
                    "text": texts[-1],
                    "model_served": usage.get("model_served"),
                    "notices": [n for n in final["notices"] if "폴백 모델" in n or "fallback model" in n],
                },
                final["notices"],
            )
            return obj, final, None

    # ------------------------------------------------------------------ web search (server tool)
    async def web_search(
        self,
        task: str,
        system: str,
        user: str,
        *,
        allowed_domains: Sequence[str] | None = None,
        max_uses: int = 5,
        effort: str | None = None,
        max_tokens: int | None = None,
        max_continuations: int = 2,
    ) -> WebResult:
        """Free-text research with the server ``web_search`` tool. Returns ``(text, sources, usage, error)``.

        ``sources`` has one entry per URL: ``{url, title, page_age, cited_text: [...]}``. Structured outputs are
        documented as incompatible with citations, so this step returns text and the caller structures it.
        """
        effort = effort or self._effort(task)
        cur_max = int(max_tokens or self._max_tokens(task))
        tool: dict[str, Any] = {
            "type": WEB_SEARCH_TOOL_TYPE,
            "name": "web_search",
            "max_uses": int(max_uses),
        }
        if allowed_domains:
            tool["allowed_domains"] = list(allowed_domains)
        key = DiskCache.key(
            self.model,
            system,
            user,
            "web_search",
            json.dumps(tool, sort_keys=True),
            effort,
        )
        notices: list[str] = []
        hit = self._cache_get(key)
        if hit is not None and isinstance(hit.get("text"), str) and isinstance(hit.get("sources"), list):
            u = empty_usage(self.model)
            u["model_served"] = hit.get("model_served")
            return (
                hit["text"],
                hit["sources"],
                self._finish(task, u, notices, None, cached=True),
                None,
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        usage: dict[str, Any] | None = None
        blocks: list[Any] = []
        resp: Any = None
        for _ in range(max(0, max_continuations) + 1):
            resp, err = await self._send(
                lambda fb: dict(
                    model=self.model,
                    max_tokens=cur_max,
                    system=cached_system(system),
                    messages=list(messages),
                    tools=[tool],
                    thinking={"type": "adaptive"},
                    output_config={"effort": effort},
                    **fallback_kwargs(fb),
                ),
                notices,
            )
            if resp is None:
                return "", [], self._finish(task, usage, notices, err), err
            usage = merge_usage(usage, usage_dict(resp, self.model))
            blocks += list(resp.content)
            if resp.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": _echoable(resp.content)})
        sources, tool_errors = collect_web_sources(blocks)
        if tool_errors:
            codes = ", ".join(dict.fromkeys(tool_errors))
            notices.append(self._t(f"웹 검색 도구 오류: {codes}", f"web search tool error: {codes}"))
        text = "".join(b.text for b in blocks if b.type == "text")
        stop = resp.stop_reason or "unknown"
        err = None
        if stop == "refusal":
            err = self._refusal_error(usage)
        elif stop == "pause_turn":
            err = self._t(
                f"웹 검색이 {max_continuations}회 이어받기 후에도 끝나지 않음(pause_turn)",
                f"web search failed to finish after {max_continuations} continuations (pause_turn)",
            )
        elif stop == "max_tokens":
            err = self._max_tokens_error(cur_max)
        final = self._finish(task, usage, notices, err)
        if err is None:
            self._cache_set(
                key,
                {
                    "text": text,
                    "sources": sources,
                    "model_served": usage.get("model_served"),
                },
                final["notices"],
            )
        return text, sources, final, err


# ================================================================================ recorder


@dataclass
class LLMRecorder:
    """Per-ticker accumulator of LLM cost, per-call usage summaries and observable messages.

    There are two channels, which map 1:1 onto ``LLMIntelligence.errors`` / ``.notices``:

    - ``errors`` (``error()``): HARD failures, where a part was NOT produced. Examples: the API call
      failed, the model refused, the output was invalid after the re-sample, an analysis step raised, or
      the critic could not run.
    - ``notices`` (``notice()``): DEGRADED-BUT-SERVED paths, where the part exists but something was
      changed or dropped on the way. Examples: a server fallback model answered, the fallback beta was
      rejected (so fallbacks are off), a value was clamped, ungrounded or unknown items were dropped,
      duplicates were merged, a pricing assumption was made, or a cache write failed.

    Messages are de-duplicated per channel and always logged (errors at ERROR, notices at WARNING),
    so nothing is silent.

    ``lang`` ("ko" default / "en", or anything ``quantcio.i18n.lang_of`` accepts, e.g. Settings) is the language
    of the labels and texts this recorder builds itself (``record()``); the analysis steps read it too.
    """

    errors: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)
    lang: str = "ko"

    def __post_init__(self) -> None:
        self.lang = lang_of(self.lang)

    def label(self, task: str) -> str:
        """User-facing task label in this recorder's language ("뉴스 감성" / "News sentiment", ...)."""
        return task_label(task, self.lang)

    def error(self, msg: str) -> None:
        """Record a hard failure (the affected part is missing)."""
        if msg and msg not in self.errors:
            self.errors.append(msg)
            log.error("LLM 오류: %s", msg)

    def notice(self, msg: str) -> None:
        """Record a degraded-but-served path (the part exists, with something corrected or dropped)."""
        if msg and msg not in self.notices:
            self.notices.append(msg)
            log.warning("LLM 참고: %s", msg)

    def record(self, task: str, usage: Mapping[str, Any], error: str | None) -> None:
        """Add one call's cost. The call's usage notices go to ``notices`` and its error goes to ``errors``,
        each labelled with the task."""
        label = self.label(task)
        self.cost_usd = round(self.cost_usd + float(usage.get("cost_usd", 0.0) or 0.0), 6)
        self.calls.append(
            {
                "task": task,
                **{
                    k: usage.get(k)
                    for k in (
                        "model_served",
                        "attempts",
                        "cost_usd",
                        "cached",
                        "fallback_used",
                        "stop_reason",
                    )
                },
                "error": error,
            }
        )
        for n in usage.get("notices", []) or []:
            self.notice(f"{label}: {n}")
        if error:
            self.error(tr(self.lang, f"{label} 실패: {error}", f"{label} failed: {error}"))


# ================================================================================ fake client


@dataclass
class FakeCall:
    task: str
    system: str
    user: str
    wire_model: type[pydantic.BaseModel] | None
    effort: str | None
    max_tokens: int | None


FakeResponse = pydantic.BaseModel | Mapping[str, Any] | str | BaseException | None


class FakeLLMClient:
    """Test double with ``LLMClient``'s interface. It never touches the network.

    ``responses`` is either a mapping ``task -> response`` (a list value is consumed one item per call) or
    a callable ``(task, system, user, wire_model) -> response``. A response can be:

    - a wire-model instance or a dict (validated against ``wire_model``) -> success
    - a ``str`` -> an error string (as if the API call failed)
    - an exception instance -> raised (to test that callers never propagate it)
    - ``None`` -> "no canned response" error

    For ``web_search``, the ``"web_research"`` response is ``(text, sources)`` or one of the above.
    """

    def __init__(
        self,
        responses: Mapping[str, Any] | Callable[..., Any] | None = None,
        *,
        settings: Settings | None = None,
        cost_per_call: float = 0.0,
    ) -> None:
        self.responses: dict[str, Any] | Callable[..., Any] = (
            responses
            if callable(responses)
            else {k: (list(v) if isinstance(v, list) else v) for k, v in (responses or {}).items()}
        )
        self.settings = settings
        self.model = settings.llm.model if settings else "fake-model"
        self.cost_per_call = cost_per_call
        self.calls: list[FakeCall] = []
        self.total_cost_usd = 0.0
        self.notices: list[str] = []

    def _next(self, task: str, system: str, user: str, wire_model: Any) -> Any:
        if callable(self.responses):
            return self.responses(task, system, user, wire_model)
        value = self.responses.get(task)
        if isinstance(value, list):
            return value.pop(0) if value else None
        return value

    def _no_response(self, task: str) -> str:
        return tr(
            self.settings,
            f"FakeLLMClient: '{task}' 응답이 준비되지 않음",
            f"FakeLLMClient error: no canned response for '{task}'",
        )

    def _usage(self, task: str, error: str | None) -> dict[str, Any]:
        u = empty_usage(self.model)
        u.update(
            cost_usd=self.cost_per_call,
            attempts=1,
            model_served=self.model,
            notices=[],
            cached=False,
            task=task,
            error=error,
            fake=True,
        )
        self.total_cost_usd = round(self.total_cost_usd + self.cost_per_call, 6)
        return u

    async def call_structured(
        self,
        task: str,
        system: str,
        user: str,
        wire_model: type[W],
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> StructuredResult:
        self.calls.append(FakeCall(task, system, user, wire_model, effort, max_tokens))
        await asyncio.sleep(0)  # yield, so concurrent callers interleave like real I/O
        value = self._next(task, system, user, wire_model)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            err = self._no_response(task)
            return None, self._usage(task, err), err
        if isinstance(value, str):
            return None, self._usage(task, value), value
        try:
            if isinstance(value, wire_model):
                obj = value
            elif isinstance(value, pydantic.BaseModel):
                obj = wire_model.model_validate(value.model_dump())
            else:
                obj = wire_model.model_validate(value)
        except pydantic.ValidationError as ve:
            err = schema_error(ve, self.settings)
            return None, self._usage(task, err), err
        return obj, self._usage(task, None), None

    async def web_search(
        self,
        task: str,
        system: str,
        user: str,
        *,
        allowed_domains: Sequence[str] | None = None,
        max_uses: int = 5,
        effort: str | None = None,
        max_tokens: int | None = None,
        max_continuations: int = 2,
    ) -> WebResult:
        self.calls.append(FakeCall(task, system, user, None, effort, max_tokens))
        await asyncio.sleep(0)
        value = self._next(task, system, user, None)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, tuple) and len(value) == 2:
            text, sources = value
            return str(text), list(sources), self._usage(task, None), None
        err = value if isinstance(value, str) else self._no_response(task)
        return "", [], self._usage(task, err), err
