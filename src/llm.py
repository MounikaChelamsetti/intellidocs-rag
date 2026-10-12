"""Answer-generation back-end (Gemini)."""
from __future__ import annotations

from typing import Callable, Optional, Protocol

from google.genai import types

from .config import Settings
from .errors import GeminiServiceError
from .gemini_utils import call_with_retries, make_client


class LLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class GeminiLLM:
    def __init__(self, settings: Settings, client=None, sleep: Optional[Callable[[float], None]] = None):
        self.settings = settings
        self._client = client or make_client(settings)
        self._sleep = sleep

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        cfg = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
        )
        kwargs = {"sleep": self._sleep} if self._sleep else {}

        def _call() -> str:
            resp = self._client.models.generate_content(
                model=self.settings.generation_model, contents=user_prompt, config=cfg
            )
            text = (getattr(resp, "text", None) or "").strip()
            if not text:
                raise GeminiServiceError(
                    "The model returned no answer (it may have been blocked by safety filters). Try rephrasing."
                )
            return text

        return call_with_retries(_call, self.settings.max_retries, self.settings.retry_base_delay_s, **kwargs)
