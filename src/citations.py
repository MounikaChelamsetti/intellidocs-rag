"""Citation parsing and validation. Any cited id that was not actually retrieved is removed."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .prompts import NO_ANSWER_MESSAGE

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass
class CitationReport:
    clean_text: str
    cited_ids: List[int]  # valid ids, in order of first appearance
    invalid_ids: List[int] = field(default_factory=list)
    coverage: Optional[float] = None  # share of claim sentences carrying >=1 valid citation
    sentences_total: int = 0
    sentences_cited: int = 0


def _ids(group: str) -> List[int]:
    return [int(x) for x in re.split(r"\s*[,;]\s*", group.strip())]


def extract_cited_ids(text: str) -> List[int]:
    return [i for m in CITATION_RE.finditer(text) for i in _ids(m.group(1))]


def is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(NO_ANSWER_MESSAGE.lower()[:60])


def _split_sentences(text: str) -> List[str]:
    # Move a citation sitting after sentence punctuation in front of it: "fact. [1]" -> "fact [1]."
    text = re.sub(r"([.!?])\s*((?:\[\d+\])+)", r"\2\1", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def validate_citations(text: str, valid_ids: Iterable[int]) -> CitationReport:
    """Remove citations that do not map to a retrieved passage; normalise the rest to [1][2]."""
    valid = set(valid_ids)
    invalid: List[int] = []
    cited: List[int] = []

    def repl(match: re.Match) -> str:
        good = []
        for i in _ids(match.group(1)):
            if i in valid:
                good.append(i)
                if i not in cited:
                    cited.append(i)
            elif i not in invalid:
                invalid.append(i)
        return "".join(f"[{i}]" for i in dict.fromkeys(good))

    clean = CITATION_RE.sub(repl, text)
    clean = re.sub(r"[ \t]+([.,;:!?])", r"\1", clean)  # tidy gaps left by removed citations
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()

    coverage = None
    total = cited_n = 0
    if not is_refusal(clean):
        sentences = _split_sentences(clean)
        total = len(sentences)
        cited_n = sum(1 for s in sentences if CITATION_RE.search(s))
        coverage = (cited_n / total) if total else None
    return CitationReport(clean, cited, invalid, coverage, total, cited_n)
