import json
from pathlib import Path

from src.config import SAMPLE_DOCS_DIR
from src.evaluation import answer_is_correct, load_cases, retrieval_recall, summarize, evaluate_retrieval_case
from src.offline import HashingEmbedder, ExtractiveDemoLLM
from src.config import Settings
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

QUESTIONS = Path(__file__).resolve().parent.parent / "evaluation" / "questions.json"


def test_answer_correctness_rule_requires_every_fact_group():
    facts = [["10:00", "10 am"], ["15:00", "3 pm"]]
    assert answer_is_correct("Core hours are 10:00 to 15:00 [1].", facts)
    assert answer_is_correct("From 10 AM to 3 PM", facts)
    assert not answer_is_correct("Core hours start at 10:00.", facts)
    assert not answer_is_correct("anything", [])


def test_recall_definition():
    gold = [("a.pdf", 1), ("b.pdf", 2)]
    assert retrieval_recall([("a.pdf", 1), ("c.pdf", 9)], gold) == 0.5
    assert retrieval_recall([], gold) == 0.0 and retrieval_recall([("a.pdf", 1)], []) is None


def test_question_set_meets_spec_and_gold_pages_exist():
    from pypdf import PdfReader

    cases = load_cases(QUESTIONS)
    assert len(cases) >= 20
    assert {"direct", "paraphrase", "multi_passage", "unanswerable", "irrelevant"} <= {c.category for c in cases}
    pages = {p.name: len(PdfReader(p).pages) for p in SAMPLE_DOCS_DIR.glob("*.pdf")}
    for c in cases:
        for doc, page in c.expected_sources:
            assert doc in pages and 1 <= page <= pages[doc], c.id
        assert c.answerable == bool(c.expected_facts)


def test_gold_facts_literally_appear_on_gold_pages():
    """Guards the ground truth: every expected fact must be findable on a cited gold page."""
    from pypdf import PdfReader

    texts = {(p.name, i + 1): (pg.extract_text() or "").lower()
             for p in SAMPLE_DOCS_DIR.glob("*.pdf") for i, pg in enumerate(PdfReader(p).pages)}
    for c in load_cases(QUESTIONS):
        if c.id.startswith("para-") or not c.answerable:
            continue  # paraphrase cases may phrase facts differently ('20%' vs '20 percent' is handled by alternates)
        blob = " ".join(texts[loc] for loc in c.expected_sources)
        for group in c.expected_facts:
            assert any(alt.lower() in blob for alt in group), (c.id, group)


def test_offline_retrieval_evaluation_runs_end_to_end():
    pipe = RAGPipeline(Settings(), HashingEmbedder(), ExtractiveDemoLLM(), VectorStore())
    for pdf in SAMPLE_DOCS_DIR.glob("*.pdf"):
        assert pipe.ingest_pdf(pdf.name, pdf.read_bytes()).status == "added"
    cases = load_cases(QUESTIONS)
    results = [evaluate_retrieval_case(pipe, c, 5, 0.1) for c in cases]
    s = summarize(results, with_answers=False)
    assert s["cases_evaluated"] == len(cases) and "answer_correctness" not in s
    assert 0.0 <= s["recall_at_k"] <= 1.0
