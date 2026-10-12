"""Embedding back-ends: Gemini (production), a disk cache wrapper, and a protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence

from google.genai import types

from .config import Settings
from .gemini_utils import call_with_retries, make_client


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


def _l2_normalize(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class GeminiEmbedder:
    """Gemini embeddings via the google-genai SDK. Uses RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY
    task types and batches up to `embed_batch_size` texts per request."""

    def __init__(self, settings: Settings, client=None, sleep: Optional[Callable[[float], None]] = None):
        self.settings = settings
        self._client = client or make_client(settings)
        self._sleep = sleep
        self.name = f"{settings.embedding_model}@{settings.embedding_dim}"

    def _embed(self, texts: Sequence[str], task_type: str) -> List[List[float]]:
        config = types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self.settings.embedding_dim
        )
        kwargs = {"sleep": self._sleep} if self._sleep else {}

        def request(batch):
            def _call():
                resp = self._client.models.embed_content(
                    model=self.settings.embedding_model, contents=list(batch), config=config
                )
                return [_l2_normalize(e.values) for e in resp.embeddings]

            return call_with_retries(
                _call, self.settings.max_retries, self.settings.retry_base_delay_s, **kwargs
            )

        out: List[List[float]] = []
        size = self.settings.embed_batch_size
        for i in range(0, len(texts), size):
            out.extend(request(texts[i : i + size]))
        return out

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT") if texts else []

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class CachedEmbedder:
    """Wraps any embedder with a JSON disk cache keyed by (embedder name, task, text hash),
    so repeated evaluation runs do not re-spend API quota on unchanged text."""

    def __init__(self, inner: Embedder, cache_path: Path):
        self.inner = inner
        self.name = inner.name
        self.path = Path(cache_path)
        self._cache: dict = {}
        self.hits = 0
        self.misses = 0
        if self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text())
            except (OSError, ValueError):
                self._cache = {}

    def _key(self, task: str, text: str) -> str:
        return hashlib.sha256(f"{self.name}|{task}|{text}".encode()).hexdigest()

    def _lookup(self, task: str, texts: Sequence[str], fn) -> List[List[float]]:
        keys = [self._key(task, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(texts) - len(missing)
        self.misses += len(missing)
        if missing:
            new = fn([texts[i] for i in missing])
            for i, vec in zip(missing, new):
                self._cache[keys[i]] = vec
            self.flush()
        return [self._cache[k] for k in keys]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._lookup("doc", texts, self.inner.embed_documents) if texts else []

    def embed_query(self, text: str) -> List[float]:
        return self._lookup("query", [text], lambda ts: [self.inner.embed_query(ts[0])])[0]

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache))
