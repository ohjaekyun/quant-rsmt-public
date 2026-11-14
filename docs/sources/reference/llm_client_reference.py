"""llm_client_reference.py - Claude structured-output client for quantcio (sync + async).

Verifier-corrected copy (2026-09-24): fixes the credential preflight for api_key="", honours
retry-after-ms and clamps the retry delay, adds billed declined-after-output fallback hops to the
cost, stops a beta-rejection retry in web_research from using up a continuation, and dedupes sources by URL.
Schema caveat: `dict[str, X]` fields become {"properties": {}, "additionalProperties": false} on the
wire, so the model can only return {}. Model factor maps as list[{name, score}] instead.

Verified offline with httpx2.MockTransport against anthropic==1.8.0, httpx2==2.13.1,
pydantic==2.13.5, Python 3.12.13 (see p09_reference_selftest.py). Live-verified only that
api.anthropic.com is reachable and that an invalid key raises AuthenticationError (401).
NOT live-verified (no API key): model output quality, and whether your org accepts the
`server-side-fallback-2026-07-01` beta. If the org rejects it, the wrapper drops fallbacks
and retries once automatically.

Design decisions (each one comes from a probe):
* Uses `client.beta.messages.create` + `output_config.format` built with
  `anthropic.transform_schema`, and validates the text block itself. It avoids `messages.parse()`
  because parse() raises pydantic.ValidationError on a refusal that has text, on truncated JSON
  (max_tokens), and whenever there is more than one text block. That error loses the response,
  so the usage and cost of the call are lost with it.
* Fallbacks use the typed beta params `betas=[FALLBACK_BETA], fallbacks="default"`. The non-beta
  namespace has no `fallbacks` kwarg (TypeError). It also turns the `fallback` block into a
  `TextBlock` and logs pydantic serializer warnings.
* Prompt caching: `cache_control: ephemeral` on the single system block. Keep that text
  byte-stable (no dates, no tickers). Prefixes below the model minimum (512-4096 tokens) are
  not cached, and no error is raised.
* SDK retries (max_retries) already cover 408/409/429/5xx/529 plus connection errors and
  timeouts. This module adds only: (a) a few outer retries for 429/529/5xx/connection errors
  after the SDK gives up, honouring retry-after; (b) one max_tokens escalation; (c) one
  re-sample when the output fails client-side validation (ge/le/min_length etc. are removed
  from the wire schema, so only the client checks them); (d) one retry without fallbacks
  if the beta header is rejected.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Awaitable, Callable, Iterable, Literal, Sequence, TypeVar, Union

import anthropic
import pydantic
from anthropic import Anthropic, AsyncAnthropic, transform_schema

log = logging.getLogger("quantcio.llm")

T = TypeVar("T", bound=pydantic.BaseModel)
R = TypeVar("R")
Effort = Literal["low", "medium", "high", "xhigh", "max"]
UserContent = Union[str, list]

DEFAULT_MODEL = "claude-opus-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # pairs ONLY with fallbacks="default" (array form uses -2026-06-01)
MAX_NONSTREAMING_TOKENS = 21_333  # SDK raises ValueError above this for non-streaming calls with the default timeout

# USD per 1M tokens: (input, output). Cache write 5m = 1.25x input, 1h = 2x input, cache read = 0.1x input.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
WEB_SEARCH_USD_PER_CALL = 10.0 / 1000

_RETRYABLE = (
    anthropic.RateLimitError,  # 429
    anthropic.OverloadedError,  # 529 (sibling of InternalServerError, not a subclass)
    anthropic.InternalServerError,  # 500/502/503/504 (1.8.0 maps 503/504 here, not to ServiceUnavailable/DeadlineExceeded)
    anthropic.APIConnectionError,  # includes APITimeoutError
)
_FATAL = (
    anthropic.AuthenticationError,  # 401
    anthropic.PermissionDeniedError,  # 403
    anthropic.NotFoundError,  # 404 (bad model id / no access)
    anthropic.RequestTooLargeError,  # 413
)


class LLMConfigError(RuntimeError):
    """No credentials. Raised by make_client() so the problem shows before any ticker is processed."""


# --------------------------------------------------------------------------- clients


def _assert_credentials(client: Anthropic | AsyncAnthropic) -> None:
    # With nothing configured, construction succeeds. The first request then raises a builtin
    # TypeError("Could not resolve authentication method...") that is not an AnthropicError.
    # FIX (verifier): api_key="" (e.g. make_client(api_key=os.getenv("X", ""))) is kept as "" by the
    # SDK, passed the old `is None` check and then hit the same TypeError at request time.
    if not client.api_key and not client.auth_token and client.credentials is None:
        raise LLMConfigError(
            "No Anthropic credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / `ant auth login` profile)."
        )


def make_client(*, max_retries: int = 3, **kwargs: Any) -> Anthropic:
    client = Anthropic(max_retries=max_retries, **kwargs)
    _assert_credentials(client)
    return client


def make_async_client(*, max_retries: int = 3, **kwargs: Any) -> AsyncAnthropic:
    client = AsyncAnthropic(max_retries=max_retries, **kwargs)
    _assert_credentials(client)
    return client


# --------------------------------------------------------------------------- helpers


@functools.lru_cache(maxsize=64)
def schema_for(model_cls: type[pydantic.BaseModel]) -> dict[str, Any]:
    """Wire schema, exactly what messages.parse() would send. Deterministic, so the cache stays stable."""
    return transform_schema(model_cls)


def _cached_system(system: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _fallback_kwargs(enabled: bool) -> dict[str, Any]:
    return {"betas": [FALLBACK_BETA], "fallbacks": "default"} if enabled else {}


def _is_beta_rejection(e: anthropic.BadRequestError) -> bool:
    msg = (e.message or "").lower()
    return "anthropic-beta" in msg or "fallback" in msg


def _retry_delay(e: Exception, attempt: int) -> float:
    # FIX (verifier): also honour retry-after-ms (the SDK does), and clamp to [0, 60] so a
    # negative/garbage header can never reach time.sleep() (negative -> ValueError).
    resp = getattr(e, "response", None)
    if resp is not None:
        for name, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            ra = resp.headers.get(name)
            if ra is None:
                continue
            try:
                return max(0.0, min(float(ra) * scale, 60.0))
            except ValueError:  # HTTP-date form -> fall through to backoff
                continue
    return min(2.0 * (2**attempt), 30.0) + random.uniform(0, 1.0)


def usage_dict(resp: Any, requested_model: str) -> dict[str, Any]:
    """Flatten usage for cost accounting. Works for Message and BetaMessage."""
    u = resp.usage
    cc = getattr(u, "cache_creation", None)
    otd = getattr(u, "output_tokens_details", None)
    stu = getattr(u, "server_tool_use", None)
    iterations = [
        {"type": it.type, "model": getattr(it, "model", None), "input_tokens": it.input_tokens, "output_tokens": it.output_tokens,
         "cache_read_input_tokens": getattr(it, "cache_read_input_tokens", 0) or 0,
         "cache_creation_input_tokens": getattr(it, "cache_creation_input_tokens", 0) or 0}
        for it in (getattr(u, "iterations", None) or [])
    ]
    d: dict[str, Any] = {
        "model_requested": requested_model,
        "model_served": resp.model,
        "fallback_used": any(it["type"] == "fallback_message" for it in iterations),
        "iterations": iterations,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cache_read_input_tokens": u.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
        "cache_creation_5m": getattr(cc, "ephemeral_5m_input_tokens", 0) if cc else 0,
        "cache_creation_1h": getattr(cc, "ephemeral_1h_input_tokens", 0) if cc else 0,
        "thinking_tokens": getattr(otd, "thinking_tokens", 0) if otd else 0,
        "web_search_requests": getattr(stu, "web_search_requests", 0) if stu else 0,
        "request_id": getattr(resp, "_request_id", None),
        "stop_reason": resp.stop_reason,
    }
    sd = resp.stop_details
    if sd is not None:
        d["stop_details"] = {
            "category": sd.category,
            "explanation": sd.explanation,
            "recommended_model": getattr(sd, "recommended_model", None),
        }
    d["cost_usd"] = estimate_cost(d, has_output=any(b.type in ("text", "thinking") for b in resp.content))
    return d


def estimate_cost(d: dict[str, Any], *, has_output: bool = True) -> float:
    # Top-level usage belongs to the attempt that produced the message. It is billed at the
    # served model's rates. A refusal before any output is not billed.
    if d["stop_reason"] == "refusal" and not has_output:
        return 0.0
    pin, pout = PRICING.get(d["model_served"], PRICING[DEFAULT_MODEL])
    five_m = d["cache_creation_5m"] or (d["cache_creation_input_tokens"] - d["cache_creation_1h"])
    usd = (
        d["input_tokens"] * pin
        + five_m * pin * 1.25
        + d["cache_creation_1h"] * pin * 2.0
        + d["cache_read_input_tokens"] * pin * 0.1
        + d["output_tokens"] * pout
    ) / 1e6
    # FIX (verifier): a hop that declined AFTER producing output is billed (docs: "declined-before-output
    # attempts are reported but not billed"), but it is not in top-level usage. Add iteration entries from a
    # different model than the served one that have output_tokens > 0, at that model's rates (5m cache write assumed).
    for it in d.get("iterations", []):
        if it["type"] == "message" and it["model"] and it["model"] != d["model_served"] and it["output_tokens"] > 0:
            ipin, ipout = PRICING.get(it["model"], PRICING[DEFAULT_MODEL])
            usd += (it["input_tokens"] * ipin + it["cache_creation_input_tokens"] * ipin * 1.25
                    + it["cache_read_input_tokens"] * ipin * 0.1 + it["output_tokens"] * ipout) / 1e6
    return round(usd + d["web_search_requests"] * WEB_SEARCH_USD_PER_CALL, 6)


_SUM_KEYS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
    "cache_creation_5m", "cache_creation_1h", "thinking_tokens", "web_search_requests", "cost_usd",
)


def _merge_usage(total: dict[str, Any] | None, d: dict[str, Any]) -> dict[str, Any]:
    if total is None:
        return {**d, "attempts": 1, "request_ids": [d.get("request_id")]}
    out = {**d, "attempts": total["attempts"] + 1, "request_ids": total["request_ids"] + [d.get("request_id")]}
    for k in _SUM_KEYS:
        out[k] = round(total.get(k, 0) + d.get(k, 0), 6) if k == "cost_usd" else total.get(k, 0) + d.get(k, 0)
    out["fallback_used"] = total["fallback_used"] or d["fallback_used"]
    return out


# --------------------------------------------------------------------------- call state machine


class _StructuredCall:
    """Retry and interpretation logic shared by the sync and async wrappers."""

    def __init__(self, model: str, system: str, user_content: UserContent, schema_model: type[T], effort: Effort,
                 max_tokens: int, use_fallbacks: bool, outer_retries: int, validation_retries: int,
                 escalate_max_tokens: bool) -> None:
        if max_tokens > MAX_NONSTREAMING_TOKENS:
            raise ValueError(f"max_tokens={max_tokens} > {MAX_NONSTREAMING_TOKENS}: non-streaming limit; use streaming")
        self.model, self.system, self.user_content, self.schema_model = model, system, user_content, schema_model
        self.effort, self.max_tokens, self.use_fallbacks = effort, max_tokens, use_fallbacks
        self.outer_left, self.validation_left, self.escalate = outer_retries, validation_retries, escalate_max_tokens
        self.outer_attempt = 0
        self.usage: dict[str, Any] | None = None

    def request(self) -> dict[str, Any]:
        return dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_cached_system(self.system),
            messages=[{"role": "user", "content": self.user_content}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema_for(self.schema_model)}},
            **_fallback_kwargs(self.use_fallbacks),
        )

    def on_error(self, e: Exception) -> float:
        """Return the number of seconds to sleep before retrying. Re-raises errors that should not be retried."""
        if isinstance(e, anthropic.BadRequestError):
            if self.use_fallbacks and _is_beta_rejection(e):
                log.warning("fallback beta rejected (%s); retrying without fallbacks", e.message)
                self.use_fallbacks = False
                return 0.0
            raise e
        if isinstance(e, _FATAL):
            raise e
        if isinstance(e, _RETRYABLE) and self.outer_left > 0:
            self.outer_left -= 1
            delay = _retry_delay(e, self.outer_attempt)
            self.outer_attempt += 1
            log.warning("%s after SDK retries; outer retry in %.1fs", type(e).__name__, delay)
            return delay
        raise e

    def on_response(self, resp: Any) -> tuple[T | None, dict[str, Any], str] | None:
        """Return the final (parsed, usage, stop_reason), or None to send the request again."""
        u = usage_dict(resp, self.model)
        self.usage = _merge_usage(self.usage, u)
        stop = resp.stop_reason or "unknown"

        if stop == "refusal":  # whole fallback chain refused (or fallbacks disabled)
            return None, self.usage, "refusal"
        if stop == "max_tokens":
            if self.escalate and self.max_tokens < MAX_NONSTREAMING_TOKENS:
                self.escalate = False
                self.max_tokens = min(self.max_tokens * 2, MAX_NONSTREAMING_TOKENS)
                log.warning("max_tokens hit; retrying with max_tokens=%d", self.max_tokens)
                return None  # retry
            return None, self.usage, "max_tokens"
        if stop == "model_context_window_exceeded":
            return None, self.usage, stop

        texts = [b.text for b in resp.content if b.type == "text"]
        if not texts:
            return None, self.usage, "no_text"
        try:
            parsed = self.schema_model.model_validate_json(texts[-1])
        except pydantic.ValidationError as ve:
            self.usage["validation_errors"] = ve.errors(include_url=False)[:5]
            if self.validation_left > 0:
                self.validation_left -= 1
                log.warning("output failed client-side validation (%s); re-sampling", ve.errors()[0]["type"])
                return None
            return None, self.usage, "invalid_output"
        self.usage.pop("validation_errors", None)
        return parsed, self.usage, stop


def call_structured(
    model: str,
    system: str,
    user_content: UserContent,
    schema_model: type[T],
    effort: Effort = "medium",
    max_tokens: int = 16_000,
    *,
    client: Anthropic | None = None,
    use_fallbacks: bool = True,
    outer_retries: int = 2,
    validation_retries: int = 1,
    escalate_max_tokens: bool = True,
) -> tuple[T | None, dict[str, Any], str]:
    """Sync call. Returns (parsed_obj | None, usage_dict, stop_reason).

    stop_reason is the API value ("end_turn", "refusal", "max_tokens", ...) or one of the wrapper's
    own values: "invalid_output" (still failed validation after retries) or "no_text".
    """
    client = client or make_client()
    st = _StructuredCall(model, system, user_content, schema_model, effort, max_tokens, use_fallbacks,
                         outer_retries, validation_retries, escalate_max_tokens)
    while True:
        try:
            resp = client.beta.messages.create(**st.request())
        except anthropic.APIError as e:
            time.sleep(st.on_error(e))
            continue
        out = st.on_response(resp)
        if out is not None:
            return out


async def call_structured_async(
    model: str,
    system: str,
    user_content: UserContent,
    schema_model: type[T],
    effort: Effort = "medium",
    max_tokens: int = 16_000,
    *,
    client: AsyncAnthropic,
    use_fallbacks: bool = True,
    outer_retries: int = 2,
    validation_retries: int = 1,
    escalate_max_tokens: bool = True,
) -> tuple[T | None, dict[str, Any], str]:
    st = _StructuredCall(model, system, user_content, schema_model, effort, max_tokens, use_fallbacks,
                         outer_retries, validation_retries, escalate_max_tokens)
    while True:
        try:
            resp = await client.beta.messages.create(**st.request())
        except anthropic.APIError as e:
            await asyncio.sleep(st.on_error(e))
            continue
        out = st.on_response(resp)
        if out is not None:
            return out


async def gather_bounded(items: Iterable[Any], fn: Callable[[Any], Awaitable[R]], concurrency: int = 4) -> list[R | BaseException]:
    """Run fn(item) for every item, at most `concurrency` at a time. Results keep input order; exceptions are returned, not raised."""
    sem = asyncio.Semaphore(concurrency)

    async def one(x: Any) -> R:
        async with sem:
            return await fn(x)

    return await asyncio.gather(*(one(x) for x in items), return_exceptions=True)


# --------------------------------------------------------------------------- optional: web research (step 1 of 2)


def _echoable(content: Sequence[Any]) -> list[Any]:
    """Blocks that are safe to send back after a server-side fallback: before the last `fallback`
    block keep only text blocks and paired server-tool blocks (per the fallback echo rules)."""
    idx = max((i for i, b in enumerate(content) if b.type == "fallback"), default=-1)
    if idx < 0:
        return list(content)
    result_ids = {getattr(b, "tool_use_id", None) for b in content if b.type.endswith("_tool_result")}
    keep = []
    for i, b in enumerate(content):
        if i > idx or b.type == "text" or b.type.endswith("_tool_result") or (b.type == "server_tool_use" and b.id in result_ids):
            keep.append(b)
    return keep


def web_research(
    query: str,
    system: str,
    *,
    client: Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    allowed_domains: Sequence[str] | None = None,
    max_uses: int = 5,
    effort: Effort = "medium",
    max_tokens: int = 16_000,
    max_continuations: int = 3,
    use_fallbacks: bool = True,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], str]:
    """Step 1: web search with free-text output and citations. Returns (text, sources, usage, stop_reason).
    Step 2: pass the text to call_structured(). Whether output_config.format can be combined with
    server tools in the same call is not documented, so the two steps stay separate."""
    client = client or make_client()
    tool: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = list(allowed_domains)
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    usage: dict[str, Any] | None = None
    fb = use_fallbacks
    resp = None
    blocks: list[Any] = []  # content from every (continued) response in this turn
    calls = 0
    while calls <= max_continuations:  # FIX (verifier): a beta-rejection retry no longer uses up a continuation
        try:
            resp = client.beta.messages.create(
                model=model, max_tokens=max_tokens, system=_cached_system(system), messages=messages,
                tools=[tool], thinking={"type": "adaptive"}, output_config={"effort": effort}, **_fallback_kwargs(fb),
            )
        except anthropic.BadRequestError as e:
            if fb and _is_beta_rejection(e):
                fb = False
                continue
            raise
        calls += 1
        usage = _merge_usage(usage, usage_dict(resp, model))
        blocks += list(resp.content)
        if resp.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": _echoable(resp.content)})  # continue the paused turn as-is
    if resp is None or usage is None:
        raise RuntimeError("web_research: no response (fallback beta rejected on every attempt?)")
    # FIX (verifier): one entry per URL (search results and citations used to duplicate each other);
    # a citation's cited_text is merged into the matching search-result entry.
    by_url: dict[str, dict[str, Any]] = {}
    for b in blocks:
        if b.type == "web_search_tool_result" and isinstance(b.content, list):
            for r in b.content:
                by_url.setdefault(r.url, {"url": r.url, "title": r.title, "page_age": r.page_age, "cited_text": []})
        elif b.type == "web_search_tool_result":
            log.warning("web search error: %s", getattr(b.content, "error_code", b.content))
        elif b.type == "text" and b.citations:
            for c in b.citations:
                url = getattr(c, "url", None)
                if not url:
                    continue
                entry = by_url.setdefault(url, {"url": url, "title": getattr(c, "title", None), "page_age": None, "cited_text": []})
                if getattr(c, "cited_text", None):
                    entry["cited_text"].append(c.cited_text)
    sources = list(by_url.values())
    text = "".join(b.text for b in blocks if b.type == "text")
    return text, sources, usage, resp.stop_reason or "unknown"
