"""LLM client initialization and Groq rate-limit retry logic."""

import time
import re
import logging
import asyncio
from functools import lru_cache

import groq
from langchain_groq import ChatGroq

from src.config import get_settings


logger = logging.getLogger(__name__)


class GroqRateLimitExhaustedError(RuntimeError):
    """Raised when Groq API rate limit is reached and bounded retries are exhausted."""
    pass


def is_rate_limit_error(e: Exception) -> bool:
    """Detect Groq HTTP 429 / rate-limit errors."""
    if isinstance(e, groq.RateLimitError):
        return True
    if getattr(e, "status_code", None) == 429:
        return True
    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    cause = getattr(e, "__cause__", None) or getattr(e, "__context__", None)
    if cause is not None and cause is not e and is_rate_limit_error(cause):
        return True
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "rate_limit" in msg or "too many requests" in msg


def _parse_duration_seconds(val: str, is_ms: bool = False) -> float | None:
    """Safely parse duration string like '2.5', '2s', '500ms', '1m' into float seconds."""
    if not val:
        return None
    val_str = str(val).strip().lower()
    if is_ms:
        try:
            return float(val_str) / 1000.0
        except ValueError:
            pass
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(ms|s|m)?$", val_str)
    if m:
        num = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            return num / 1000.0
        elif unit == "m":
            return num * 60.0
        return num
    try:
        return float(val_str)
    except ValueError:
        return None


def _get_retry_after(e: Exception, default: float = 2.0) -> float:
    """Extract retry-after duration in seconds from Groq response headers or error message."""
    response = getattr(e, "response", None)
    if response is not None and hasattr(response, "headers"):
        headers = response.headers
        ra = headers.get("retry-after")
        if ra:
            sec = _parse_duration_seconds(ra)
            if sec is not None:
                return max(0.5, min(sec, 10.0))
        ra_ms = headers.get("retry-after-ms")
        if ra_ms:
            sec = _parse_duration_seconds(ra_ms, is_ms=True)
            if sec is not None:
                return max(0.5, min(sec, 10.0))
        ra_reset = headers.get("x-ratelimit-reset-requests")
        if ra_reset:
            sec = _parse_duration_seconds(ra_reset)
            if sec is not None:
                return max(0.5, min(sec, 10.0))

    msg = str(e)
    m = re.search(r"(?:try again in|retry after|wait)\s+(\d+(?:\.\d+)?)\s*(ms|s|m)?", msg, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "s").lower()
        if unit == "ms":
            sec = val / 1000.0
        elif unit == "m":
            sec = val * 60.0
        else:
            sec = val
        return max(0.5, min(sec, 10.0))
    return default


class RateLimitedChatGroq(ChatGroq):
    """ChatGroq with bounded retry handling for HTTP 429 RateLimitErrors."""

    max_rate_limit_retries: int = 2

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        attempts = 0
        while True:
            try:
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                attempts += 1
                if attempts > self.max_rate_limit_retries:
                    logger.warning(
                        "Groq rate limit (HTTP 429) retries exhausted after %d attempts: %s",
                        attempts,
                        e,
                    )
                    raise GroqRateLimitExhaustedError(
                        f"Groq API rate limit reached after {attempts} attempts. {e}"
                    ) from e

                wait_sec = _get_retry_after(e, default=2.0 * attempts)
                logger.info(
                    "Groq rate limit (HTTP 429) hit. Waiting %.2fs before retry %d/%d...",
                    wait_sec,
                    attempts,
                    self.max_rate_limit_retries,
                )
                time.sleep(wait_sec)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        attempts = 0
        while True:
            try:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                attempts += 1
                if attempts > self.max_rate_limit_retries:
                    logger.warning(
                        "Groq rate limit (HTTP 429) retries exhausted after %d attempts: %s",
                        attempts,
                        e,
                    )
                    raise GroqRateLimitExhaustedError(
                        f"Groq API rate limit reached after {attempts} attempts. {e}"
                    ) from e

                wait_sec = _get_retry_after(e, default=2.0 * attempts)
                logger.info(
                    "Groq rate limit (HTTP 429) hit. Waiting %.2fs before retry %d/%d...",
                    wait_sec,
                    attempts,
                    self.max_rate_limit_retries,
                )
                await asyncio.sleep(wait_sec)


@lru_cache
def get_llm():
    """Return the main LLM instance (cached, thread-safe)."""
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return RateLimitedChatGroq(
        api_key=s.groq_api_key,
        model=s.groq_model,
        temperature=0,
    )


@lru_cache
def get_fast_llm():
    """Return a smaller/faster LLM for classification tasks (cached, thread-safe)."""
    s = get_settings()
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    return RateLimitedChatGroq(
        api_key=s.groq_api_key,
        model="openai/gpt-oss-20b",
        temperature=0,
    )
