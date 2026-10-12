"""Offline components used for demo mode, tests and key-less evaluation.

These are NOT substitutes for Gemini. HashingEmbedder is a lexical (bag-of-words) embedder;
ExtractiveDemoLLM just quotes the top passage. They exist so the whole pipeline can run
without network access or an API key.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import List, Sequence

_STOP = set(
    "a an the and or but if of to in on at by for with from as is are was were be been being it its this that these "
    "those what which who whom how when where why do does did can could should would may might will shall have has "
    "had i you he she we they them his her our your their my me not no than then so such about into over per "
    "there here any all each more most some other also".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOP:
            continue
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class HashingEmbedder:
    """Deterministic hashed bag-of-words embedder (sublinear TF, L2-normalised)."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hashing-bow@{dim}"

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        counts: dict = {}
        for tok in tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
        for tok, c in counts.items():
            vec[zlib.crc32(tok.encode()) % self.dim] += 1.0 + math.log(c)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0  # avoid zero vectors (undefined cosine)
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class ExtractiveDemoLLM:
    """Returns the first two sentences of the first provided source, cited as [id].
    Demonstrates the UI and citation plumbing; performs no reasoning."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        match = re.search(r'<source id="(\d+)"[^>]*>\s*(.*?)\s*</source>', user_prompt, re.S)
        if not match:
            from .prompts import NO_ANSWER_MESSAGE

            return NO_ANSWER_MESSAGE
        sid, body = match.group(1), match.group(2)
        sentences = re.split(r"(?<=[.!?])\s+", body)
        excerpt = " ".join(sentences[:2]).strip()
        return f"[Demo mode - extractive, no LLM] {excerpt} [{sid}]"
