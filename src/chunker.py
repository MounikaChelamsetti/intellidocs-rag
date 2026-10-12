"""Word-based chunking with overlap. Chunks never cross page boundaries, so every chunk
maps to exactly one real PDF page and citations are exact."""
from __future__ import annotations

from typing import List

from .errors import ConfigError
from .models import Chunk
from .pdf_processor import ProcessedDocument


def split_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    if chunk_size <= 0:
        raise ConfigError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ConfigError("overlap must be >= 0 and smaller than chunk_size.")
    if not words:
        return []
    step = chunk_size - overlap
    windows, start = [], 0
    while True:
        end = min(start + chunk_size, len(words))
        windows.append(words[start:end])
        if end >= len(words):
            break
        start += step
    return windows


def chunk_document(doc: ProcessedDocument, chunk_size: int = 400, overlap: int = 60) -> List[Chunk]:
    chunks: List[Chunk] = []
    index = 0
    for page in doc.pages:
        words = page.text.split()
        for window in split_words(words, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-p{page.page_number}-c{index}",
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=" ".join(window),
                    word_count=len(window),
                )
            )
            index += 1
    return chunks
