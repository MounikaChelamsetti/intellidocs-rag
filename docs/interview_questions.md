# Interview Questions & Answers (based on this implementation)

1. **What problem does RAG solve here?** LLMs don't know your private PDFs and may hallucinate. RAG retrieves relevant passages at query time and makes the model answer from them, so answers are current, source-grounded and auditable.
2. **Walk through the pipeline.** Validate → extract per page → chunk → embed → store in ChromaDB → embed query → top-k cosine search → relevance filter → prompt with numbered sources → Gemini → citation validation → UI.
3. **Why chunk at all?** Embedding a whole document blurs many topics into one vector, and the context window/cost limits what can be sent. Chunks keep each vector topical and the prompt small.
4. **Why 400 words with 60 overlap?** A common starting point: big enough for self-contained context, small enough for precise retrieval; overlap protects facts spanning a boundary. It is configurable and should be tuned by evaluation, not trusted blindly.
5. **Why do chunks never cross pages?** So each chunk has exactly one page number, making page citations exact. The cost is lost context for ideas spanning pages.
6. **Why cosine similarity, and what is the score?** Embedding direction carries meaning; cosine ignores magnitude. Chroma returns cosine *distance*, so score = 1 − distance. Using the raw distance as a "similarity" would invert the filter.
7. **How is the relevance threshold chosen?** It is a tunable heuristic (default 0.30). Scales differ per embedding model, so the harness produces a threshold sweep (recall vs. blocked unanswerable) to pick a value per model/corpus.
8. **What are task types in Gemini embeddings?** `RETRIEVAL_DOCUMENT` for stored chunks and `RETRIEVAL_QUERY` for questions; the model optimises the asymmetric query→document match.
9. **Why L2-normalise vectors?** Reduced-dimension Gemini embeddings aren't guaranteed unit length; normalising keeps cosine comparisons consistent.
10. **How do you prevent hallucination?** Strict system prompt (sources only, fixed refusal sentence), temperature 0, relevance filtering, skipping the LLM when nothing is relevant, and citation validation. None is a guarantee.
11. **How are citations validated?** Regex parses `[n]` groups; ids not in `1..N` (N = passages sent) are removed; if invalid/missing citations are found the model is asked once to rewrite; page numbers come from metadata.
12. **What can citation validation *not* catch?** A real id attached to a claim the passage doesn't support. Detecting that needs human review or an entailment/judge model.
13. **How do you handle prompt injection from PDFs?** Source text is wrapped in `<source>` tags with `<`/`>` escaped (a document can't close the tag), the system prompt declares sources as data, and we don't execute document content. This reduces but doesn't eliminate risk; no tools or secrets are exposed to the model.
14. **How do follow-up questions work?** The current question is retrieved on its own *and* concatenated with the previous user question; results are merged by best score. History goes into the prompt only for reference resolution, never as evidence. No extra LLM call is needed.
15. **How do you detect scanned PDFs?** If every page's extracted text is empty the file is flagged `needs_ocr` and not indexed; the UI says so rather than pretending extraction worked.
16. **How is duplicate detection done?** SHA-256 of the file bytes → doc id; `store.has_document` is checked before extraction/embedding, so re-uploads (even renamed) cost nothing.
17. **Why a per-session in-memory Chroma collection?** Prevents data leaking between visitors and avoids promising persistence that Streamlit Community Cloud doesn't guarantee. `weakref.finalize` frees the collection when the session ends.
18. **How do you handle API rate limits?** Exponential backoff with jitter on 429/5xx/network errors (not on auth errors), then a user-friendly error. Embeddings are batched (≤100) to reduce request count.
19. **How did you make the code testable without an API key?** Embedder and LLM are injected via small protocols; tests use a deterministic hashing embedder, scripted LLMs and fake SDK clients. Streamlit `AppTest` covers the UI.
20. **Define Recall@k.** Fraction of gold (document, page) locations appearing in the top-k retrieved chunks, averaged over answerable questions. Hit@k is "at least one gold found".
21. **Why is Recall@5 = 1.0 on the sample set not impressive?** The corpus has 11 chunks, so top-5 already covers ~45% of it. Recall@1/@3 are more informative, and a larger corpus is needed for real conclusions.
22. **How do you measure unsupported answers?** On unanswerable/irrelevant questions, the share of responses that are not refusals. It needs a live LLM; the offline run only reports how often the retrieval filter alone blocked such questions.
23. **What are the limits of the automated answer-correctness metric?** It checks for required strings; it can't judge phrasing, completeness beyond those strings, or additional wrong claims. Human review is still required.
24. **Why skip the LLM when retrieval returns nothing?** It saves cost/latency and removes a hallucination path; the refusal is deterministic.
25. **What would you add to improve retrieval?** Hybrid BM25 + vector search, a cross-encoder re-ranker, cross-page chunking with page ranges, query rewriting with the LLM, and metadata filters per document.
26. **How would you scale this beyond a demo?** Persistent vector DB (managed Chroma/pgvector/Qdrant), background ingestion workers, auth + per-user collections, rate limiting, caching, observability and an offline eval suite in CI.
27. **Why `temperature=0`?** Reduces variance for grounded QA and makes evaluation more repeatable (still not perfectly deterministic).
28. **How do you keep API keys safe?** Env var / `.env` / Streamlit secrets only; `.env` is git-ignored; the key field is excluded from `repr`; error messages never include it.
