"""Shared plain data containers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int  # 1-based, the real PDF page
    chunk_index: int  # 0-based position within the document
    text: str
    word_count: int


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float  # cosine similarity = 1 - cosine distance (higher is better)
    distance: float
