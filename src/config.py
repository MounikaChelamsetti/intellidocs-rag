"""Centralised configuration. All tunable values and model names live here.

The API key is read from the GEMINI_API_KEY environment variable (optionally loaded from
a local .env file) or from Streamlit secrets. It is never hardcoded or logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "data" / "sample_documents"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVALUATION_DIR / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

DEFAULT_GENERATION_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: Optional[str] = field(default=None, repr=False)  # never printed
    generation_model: str = DEFAULT_GENERATION_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = 768
    embed_batch_size: int = 100
    chunk_size_words: int = 400
    chunk_overlap_words: int = 60
    top_k: int = 5
    # Minimum cosine similarity (1 - cosine distance). A tunable heuristic, NOT a universal
    # constant: it depends on the embedding model. See README "Relevance filtering".
    min_relevance: float = 0.30
    max_upload_mb: int = 20
    max_files: int = 10
    max_pages_per_pdf: int = 300
    history_turns: int = 4
    temperature: float = 0.0
    max_output_tokens: int = 1024
    max_retries: int = 4
    retry_base_delay_s: float = 2.0
    persist_dir: Optional[str] = None  # None => in-memory store, scoped to the session
    demo_mode: bool = False  # offline lexical embedder + extractive answers (no API key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def require_api_key(self) -> str:
        if not self.gemini_api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Add it to a .env file, your environment, or "
                "Streamlit secrets (see README)."
            )
        return self.gemini_api_key

    def validate(self) -> "Settings":
        if self.chunk_size_words <= 0:
            raise ConfigError("chunk_size_words must be positive.")
        if not 0 <= self.chunk_overlap_words < self.chunk_size_words:
            raise ConfigError("chunk_overlap_words must be >= 0 and < chunk_size_words.")
        if self.top_k < 1:
            raise ConfigError("top_k must be at least 1.")
        if not -1.0 <= self.min_relevance <= 1.0:
            raise ConfigError("min_relevance must be within [-1, 1] (cosine similarity).")
        if self.embed_batch_size < 1 or self.embed_batch_size > 100:
            raise ConfigError("embed_batch_size must be between 1 and 100.")
        return self


def _get(env: Mapping[str, str], name: str, cast, default):
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"Invalid value for {name}: expected {cast.__name__}.") from exc


def _truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def load_settings(
    env: Optional[Mapping[str, str]] = None,
    secrets: Optional[Mapping[str, str]] = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Build Settings from env vars (+ optional .env file and Streamlit secrets)."""
    if env is None:
        if load_dotenv_file:
            load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ
    key = (env.get("GEMINI_API_KEY") or "").strip()
    if not key and secrets is not None:
        try:
            key = str(secrets.get("GEMINI_API_KEY", "") or "").strip()
        except Exception:  # missing secrets file etc.
            key = ""
    defaults = Settings()
    return Settings(
        gemini_api_key=key or None,
        generation_model=_get(env, "GENERATION_MODEL", str, defaults.generation_model),
        embedding_model=_get(env, "EMBEDDING_MODEL", str, defaults.embedding_model),
        embedding_dim=_get(env, "EMBEDDING_DIM", int, defaults.embedding_dim),
        chunk_size_words=_get(env, "CHUNK_SIZE_WORDS", int, defaults.chunk_size_words),
        chunk_overlap_words=_get(env, "CHUNK_OVERLAP_WORDS", int, defaults.chunk_overlap_words),
        top_k=_get(env, "TOP_K", int, defaults.top_k),
        min_relevance=_get(env, "MIN_RELEVANCE", float, defaults.min_relevance),
        max_upload_mb=_get(env, "MAX_UPLOAD_MB", int, defaults.max_upload_mb),
        max_files=_get(env, "MAX_FILES", int, defaults.max_files),
        max_pages_per_pdf=_get(env, "MAX_PAGES_PER_PDF", int, defaults.max_pages_per_pdf),
        persist_dir=_get(env, "CHROMA_PERSIST_DIR", str, None),
        demo_mode=_truthy(env.get("INTELLIDOCS_DEMO_MODE", "")),
    ).validate()
