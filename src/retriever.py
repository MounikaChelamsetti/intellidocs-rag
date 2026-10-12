"""Semantic retrieval with relevance filtering and optional multi-query merging."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .embeddings import Embedder
from .models import RetrievedChunk
from .vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunks: List[RetrievedChunk]  # after relevance filtering, best first, at most top_k
    candidates: List[RetrievedChunk] = field(default_factory=list)  # before filtering

    @property
    def top_score(self):
        return self.candidates[0].score if self.candidates else None


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
        extra_queries: Sequence[str] = (),
    ) -> RetrievalResult:
        """Retrieve for `query` (and any `extra_queries`, e.g. a history-aware variant),
        merge by best score per chunk, then drop chunks whose cosine similarity < min_score."""
        best: Dict[str, RetrievedChunk] = {}
        for q in [query, *extra_queries]:
            if not q.strip():
                continue
            for chunk in self.store.query(self.embedder.embed_query(q), top_k):
                if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
                    best[chunk.chunk_id] = chunk
        candidates = sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]
        return RetrievalResult(chunks=[c for c in candidates if c.score >= min_score], candidates=candidates)
