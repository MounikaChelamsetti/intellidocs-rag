"""Evaluation metrics and runners.

Definitions (all automated approximations; none replaces human review):

* Recall@k  - for each answerable case, the fraction of gold (document, page) locations that
  appear among the top-k retrieved chunks; averaged over answerable cases. Hit@k is the share of
  answerable cases with at least one gold location retrieved.
* Answer correctness - rule-based: the answer must contain at least one accepted string from
  EVERY required-fact group (case-insensitive substring). Cannot judge wording quality or
  detect extra wrong claims. Needs a live LLM.
* Citation correctness - an answerable case passes when the answer cites >=1 source, every cited
  source maps to a retrieved chunk, and at least one cited chunk lies on a gold (document, page).
* Citation precision - cited sources on gold pages / all cited sources (answerable cases).
* Citation coverage - mean share of claim sentences carrying a valid citation.
* Unsupported-answer rate - among unanswerable + irrelevant cases, the share where the system
  returned a substantive (non-refusal) answer instead of declining. Lower is better.
  Needs a live LLM (the retrieval filter alone is reported separately as "filter refusal rate").
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .rag_pipeline import Answer, RAGPipeline

Location = Tuple[str, int]


@dataclass
class EvalCase:
    id: str
    category: str
    question: str
    answerable: bool
    expected_facts: List[List[str]] = field(default_factory=list)
    expected_sources: List[Location] = field(default_factory=list)


def load_cases(path: Path) -> List[EvalCase]:
    raw = json.loads(Path(path).read_text())["cases"]
    return [
        EvalCase(
            c["id"], c["category"], c["question"], c["answerable"], c.get("expected_facts", []),
            [(s["document"], int(s["page"])) for s in c.get("expected_sources", [])],
        )
        for c in raw
    ]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def answer_is_correct(answer_text: str, expected_facts: Sequence[Sequence[str]]) -> bool:
    if not expected_facts:
        return False
    t = _norm(answer_text)
    return all(any(_norm(alt) in t for alt in group) for group in expected_facts)


def retrieval_recall(retrieved: Iterable[Location], gold: Sequence[Location]) -> Optional[float]:
    if not gold:
        return None
    got: Set[Location] = set(retrieved)
    return sum(1 for g in set(gold) if g in got) / len(set(gold))


def cited_locations(answer: Answer) -> List[Location]:
    return [(s.chunk.filename, s.chunk.page_number) for s in answer.sources if s.cited]


def citation_metrics(answer: Answer, gold: Sequence[Location]) -> Dict[str, Optional[float]]:
    cited = cited_locations(answer)
    on_gold = [c for c in cited if c in set(gold)]
    return {
        "citation_correct": bool(cited and on_gold),
        "citation_precision": (len(on_gold) / len(cited)) if cited else 0.0,
    }


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answerable: bool
    top_score: Optional[float]
    retrieved: List[Location]
    gold: List[Location] = field(default_factory=list)
    recall: Optional[float] = None
    # Only populated when an LLM was used:
    answer_text: Optional[str] = None
    refused: Optional[bool] = None
    correct: Optional[bool] = None
    citation_correct: Optional[bool] = None
    citation_precision: Optional[float] = None
    coverage: Optional[float] = None
    filter_refused: Optional[bool] = None


def evaluate_retrieval_case(pipeline: RAGPipeline, case: EvalCase, top_k: int, min_score: float) -> CaseResult:
    res = pipeline.retriever.retrieve(case.question, top_k, min_score)
    locs = [(c.filename, c.page_number) for c in res.candidates]  # recall uses the unfiltered top-k
    return CaseResult(
        case.id, case.category, case.question, case.answerable, res.top_score, locs, list(case.expected_sources),
        retrieval_recall(locs, case.expected_sources) if case.answerable else None,
        filter_refused=(len(res.chunks) == 0),
    )


def evaluate_answer_case(pipeline: RAGPipeline, case: EvalCase, top_k: int, min_score: float) -> CaseResult:
    base = evaluate_retrieval_case(pipeline, case, top_k, min_score)
    ans = pipeline.ask(case.question, top_k=top_k, min_score=min_score)
    base.answer_text, base.refused, base.coverage = ans.text, ans.refused, ans.coverage
    if case.answerable:
        base.correct = (not ans.refused) and answer_is_correct(ans.text, case.expected_facts)
        cm = citation_metrics(ans, case.expected_sources)
        base.citation_correct, base.citation_precision = cm["citation_correct"], cm["citation_precision"]
    return base


def _m(values: List[float]) -> Optional[float]:
    return round(mean(values), 4) if values else None


def summarize(results: Sequence[CaseResult], with_answers: bool) -> Dict:
    ans = [r for r in results if r.answerable]
    neg = [r for r in results if not r.answerable]
    out: Dict = {
        "cases_evaluated": len(results),
        "answerable_cases": len(ans),
        "unanswerable_or_irrelevant_cases": len(neg),
        "recall_at_k": _m([r.recall for r in ans if r.recall is not None]),
        "hit_rate_at_k": _m([1.0 if (r.recall or 0) > 0 else 0.0 for r in ans]),
        "recall_at_1": _m([retrieval_recall(r.retrieved[:1], r.gold) for r in ans]),
        "recall_at_3": _m([retrieval_recall(r.retrieved[:3], r.gold) for r in ans]),
        "retrieval_top1_score_answerable": _m([r.top_score for r in ans if r.top_score is not None]),
        "retrieval_top1_score_unanswerable": _m([r.top_score for r in neg if r.top_score is not None]),
        "filter_refusal_rate_unanswerable": _m([1.0 if r.filter_refused else 0.0 for r in neg]),
        "filter_refusal_rate_answerable (lower is better)": _m([1.0 if r.filter_refused else 0.0 for r in ans]),
    }
    if with_answers:
        out.update(
            {
                "answer_correctness": _m([1.0 if r.correct else 0.0 for r in ans]),
                "citation_correctness": _m([1.0 if r.citation_correct else 0.0 for r in ans]),
                "citation_precision": _m([r.citation_precision for r in ans if r.citation_precision is not None]),
                "citation_coverage": _m([r.coverage for r in results if r.coverage is not None]),
                "unsupported_answer_rate": _m([0.0 if r.refused else 1.0 for r in neg]),
            }
        )
    return out


def results_to_json(results: Sequence[CaseResult]) -> List[dict]:
    return [asdict(r) for r in results]
