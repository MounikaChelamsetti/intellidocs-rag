"""Gemini client construction, error translation and retry/backoff."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from google import genai
from google.genai import errors as genai_errors

from .config import Settings
from .errors import (
    GeminiAuthError,
    GeminiRateLimitError,
    GeminiServiceError,
    IntelliDocsError,
)

T = TypeVar("T")


def make_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.require_api_key())


def translate_error(exc: Exception) -> IntelliDocsError:
    """Map SDK/network exceptions to safe, user-presentable errors (never include the key)."""
    if isinstance(exc, IntelliDocsError):
        return exc
    if isinstance(exc, genai_errors.APIError):
        code = getattr(exc, "code", None)
        message = str(getattr(exc, "message", "") or "").lower()
        if code in (401, 403) or (code == 400 and "api key" in message):
            return GeminiAuthError("The Gemini API rejected the API key. Check GEMINI_API_KEY.")
        if code == 429:
            return GeminiRateLimitError(
                "Gemini rate limit or quota reached. Wait a minute and retry, or check your quota."
            )
        if code is not None and code >= 500:
            return GeminiServiceError("The Gemini service is temporarily unavailable. Please retry.")
        return GeminiServiceError(f"The Gemini API returned an error (HTTP {code}).")
    return GeminiServiceError("Could not reach the Gemini API. Check your network connection and retry.")


def call_with_retries(
    fn: Callable[[], T],
    max_retries: int = 4,
    base_delay_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry rate-limit and 5xx/network errors with exponential backoff + jitter.
    Authentication and other client errors are not retried."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - translated below
            err = translate_error(exc)
            retryable = isinstance(err, (GeminiRateLimitError, GeminiServiceError))
            if not retryable or attempt >= max_retries:
                raise err from None
            sleep(min(base_delay_s * (2**attempt), 60.0) + random.uniform(0, 0.5))
            attempt += 1
