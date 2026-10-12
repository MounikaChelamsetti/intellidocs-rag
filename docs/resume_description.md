# Resume Description

## One-line bullet
Built **IntelliDocs**, a multi-PDF RAG chatbot (Python, Streamlit, Gemini, ChromaDB, pypdf) that answers questions with validated page-level citations, refusal on missing evidence, and a reproducible evaluation harness covered by 75 offline pytest tests.

## Project summary (3–4 sentences)
IntelliDocs lets users upload several PDFs and ask natural-language questions. It extracts text per page, chunks it with configurable overlap, embeds it with Gemini, and retrieves evidence from per-session ChromaDB collections. Answers are generated only from numbered passages, every citation id is validated against what was retrieved (invalid ones are removed or regenerated), and the UI shows the filename, true PDF page, score and exact passage. Includes input validation, OCR-needed detection, duplicate detection, rate-limit handling, and deployment configuration for Streamlit Community Cloud.

## Technologies
Python 3.11+, Streamlit, Google Gemini API (google-genai), ChromaDB, pypdf, pandas, Plotly, pytest, python-dotenv, GitHub Actions.

## Measured metrics (fill in only what you measured yourself)
Measured by the build (offline lexical embedder, retrieval only, 25 fictional-corpus cases, 11 pages): Recall@1 = 0.76, Recall@3 = 0.94, Recall@5 = 1.00 (n = 17 answerable questions). Test suite: 75 tests passing.

Live Gemini results — **run `python evaluation/run_evaluation.py --mode live` and fill in**:
- Answer correctness (rule-based): `[__ %]` on `[__]` answerable cases
- Citation correctness: `[__ %]`  · Citation coverage: `[__ %]`
- Unsupported-answer rate on out-of-document questions: `[__ %]` on `[__]` cases
- Gemini Recall@5 / chosen `min_relevance`: `[__]` / `[__]`

> When quoting numbers, state the corpus size (small, synthetic) and that correctness is automated, not human-graded.
