"""End-to-end orchestration: ingest PDFs and answer questions with validated citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .chunker import chunk_document
from .citations import extract_cited_ids, is_refusal, validate_citations
from .config import Settings
from .embeddings import Embedder
from .errors import IntelliDocsError
from .llm import LLM
from .models import RetrievedChunk
from .pdf_processor import compute_doc_id, extract_pdf
from .prompts import NO_ANSWER_MESSAGE, NO_DOCUMENTS_MESSAGE, SYSTEM_PROMPT, build_user_prompt
from .retriever import Retriever
from .vector_store import VectorStore


@dataclass
class IngestResult:
    filename: str
    status: str  # added | duplicate | ocr_required | error
    message: str
    doc_id: Optional[str] = None
    pages: int = 0
    chunks: int = 0
    empty_pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Source:
    id: int  # the [n] shown in the answer
    chunk: RetrievedChunk
    cited: bool


@dataclass
class Answer:
    question: str
    text: str
    sources: List[Source]
    refused: bool
    cited_ids: List[int] = field(default_factory=list)
    removed_invalid_ids: List[int] = field(default_factory=list)
    coverage: Optional[float] = None
    regenerated: bool = False
    llm_called: bool = False
    top_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, llm: LLM, store: Optional[VectorStore] = None):
        self.settings = settings
        self.embedder = embedder
        self.llm = llm
        self.store = store or VectorStore(persist_dir=settings.persist_dir)
        self.retriever = Retriever(self.store, embedder)

    # ---------------------------------------------------------------- ingestion
    def ingest_pdf(
        self, filename: str, data: bytes, chunk_size: Optional[int] = None, overlap: Optional[int] = None
    ) -> IngestResult:
        s = self.settings
        try:
            doc_id = compute_doc_id(data) if data else None
            if doc_id and self.store.has_document(doc_id):  # skip re-embedding unchanged files
                return IngestResult(filename, "duplicate", "Already processed - skipped.", doc_id)
            doc = extract_pdf(filename, data, s.max_upload_bytes, s.max_pages_per_pdf)
            if doc.needs_ocr:
                return IngestResult(
                    filename, "ocr_required", doc.warnings[-1], doc.doc_id, doc.total_pages, 0,
                    doc.empty_pages, doc.warnings,
                )
            chunks = chunk_document(doc, chunk_size or s.chunk_size_words, overlap if overlap is not None else s.chunk_overlap_words)
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
            self.store.add_chunks(chunks, embeddings)
            return IngestResult(
                filename, "added", f"Indexed {len(doc.non_empty_pages)} page(s) into {len(chunks)} chunk(s).",
                doc.doc_id, doc.total_pages, len(chunks), doc.empty_pages, doc.warnings,
            )
        except IntelliDocsError as exc:
            return IngestResult(filename, "error", str(exc))

    def documents(self) -> List[dict]:
        return self.store.list_documents()

    def reset(self) -> None:
        self.store.reset()

    # ---------------------------------------------------------------- question answering
    def ask(
        self,
        question: str,
        history: Sequence[Dict[str, str]] = (),
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> Answer:
        s = self.settings
        question = (question or "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        if self.store.count() == 0:
            return Answer(question, NO_DOCUMENTS_MESSAGE, [], refused=True)

        # History-aware retrieval without an extra LLM call: also search with the previous user
        # question prepended, so follow-ups like "and how long is it?" still find evidence.
        extra: List[str] = []
        prev_users = [m["content"] for m in history if m.get("role") == "user"]
        if prev_users:
            extra.append(f"{prev_users[-1]} {question}")

        result = self.retriever.retrieve(
            question, top_k or s.top_k, s.min_relevance if min_score is None else min_score, extra
        )
        if not result.chunks:  # nothing relevant: refuse without calling the LLM
            return Answer(question, NO_ANSWER_MESSAGE, [], refused=True, top_score=result.top_score)

        chunks = result.chunks
        valid_ids = list(range(1, len(chunks) + 1))
        hist_msgs = s.history_turns * 2
        raw = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks, history, hist_msgs))
        report = validate_citations(raw, valid_ids)
        regenerated = False

        needs_retry = not is_refusal(report.clean_text) and (report.invalid_ids or not report.cited_ids)
        if needs_retry:
            note = (
                f"Your previous answer had citation problems (invalid ids: {report.invalid_ids or 'none'}; "
                f"valid citations found: {len(report.cited_ids)}). Valid source ids are {valid_ids}. "
                "Rewrite the answer citing only those ids for every factual statement."
            )
            raw = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks, history, hist_msgs, note))
            regenerated = True
            second = validate_citations(raw, valid_ids)
            report = second
            report.invalid_ids = sorted(set(report.invalid_ids))

        warnings: List[str] = []
        refused = is_refusal(report.clean_text)
        if refused:
            text, cited = NO_ANSWER_MESSAGE, []
        else:
            text, cited = report.clean_text, report.cited_ids
            if report.invalid_ids:
                warnings.append(f"Removed invalid citation id(s) {report.invalid_ids} from the model's answer.")
            if not cited:
                warnings.append("The answer contains no verifiable citations; treat it with caution.")
        sources = [Source(i, c, cited=(i in cited)) for i, c in enumerate(chunks, start=1)]
        return Answer(
            question, text, sources, refused, cited, report.invalid_ids, None if refused else report.coverage,
            regenerated, True, result.top_score, warnings,
        )
