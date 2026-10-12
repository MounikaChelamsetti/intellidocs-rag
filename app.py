"""IntelliDocs - AI Document Assistant (Streamlit UI)."""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import replace

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import RESULTS_DIR, load_settings
from src.errors import ConfigError, IntelliDocsError
from src.offline import ExtractiveDemoLLM, HashingEmbedder
from src.rag_pipeline import Answer, RAGPipeline

logger = logging.getLogger("intellidocs")

st.set_page_config(page_title="IntelliDocs - AI Document Assistant", page_icon="📄", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.hero {background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%); color:#fff; padding:1.4rem 1.6rem;
       border-radius:14px; margin-bottom:1.2rem;}
.hero h1 {color:#fff; margin:0; font-size:1.9rem;}
.hero p {margin:.3rem 0 0 0; opacity:.92;}
.pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.75rem; font-weight:600;}
.pill-live {background:#dcfce7; color:#166534;} .pill-demo {background:#fef3c7; color:#92400e;}
.pill-off {background:#fee2e2; color:#991b1b;}
.src-meta {color:#6b7280; font-size:.85rem;}
[data-testid="stSidebar"] {background:#f4f5fb;}
@media (max-width: 640px) {.hero h1 {font-size:1.4rem;} .block-container {padding-left:1rem; padding-right:1rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- setup
def _secrets():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_settings():
    return load_settings(secrets=_secrets())


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_id", uuid.uuid4().hex)
    ss.setdefault("pipeline", None)
    ss.setdefault("messages", [])
    ss.setdefault("ingest_log", [])
    ss.setdefault("uploader_key", 0)


def get_pipeline(settings):
    """One pipeline (= one isolated in-memory collection) per browser session."""
    ss = st.session_state
    if ss.pipeline is None:
        if settings.demo_mode:
            ss.pipeline = RAGPipeline(settings, HashingEmbedder(), ExtractiveDemoLLM())
        else:
            from src.embeddings import GeminiEmbedder
            from src.gemini_utils import make_client
            from src.llm import GeminiLLM

            client = make_client(settings)  # raises ConfigError if no key
            ss.pipeline = RAGPipeline(settings, GeminiEmbedder(settings, client=client), GeminiLLM(settings, client=client))
    return ss.pipeline


def mode_pill(settings) -> str:
    if settings.demo_mode:
        return '<span class="pill pill-demo">DEMO MODE · offline, no LLM</span>'
    if settings.gemini_api_key:
        return '<span class="pill pill-live">LIVE · Gemini</span>'
    return '<span class="pill pill-off">NO API KEY</span>'


def render_citations(text: str) -> str:
    text = text.replace("$", "\\$")
    return re.sub(r"\[(\d+)\]", r":blue-badge[\1]", text)


def answer_to_message(ans: Answer) -> dict:
    return {
        "role": "assistant",
        "content": ans.text,
        "refused": ans.refused,
        "warnings": ans.warnings,
        "meta": {"top_score": ans.top_score, "coverage": ans.coverage, "regenerated": ans.regenerated,
                 "llm_called": ans.llm_called, "n_sources": len(ans.sources), "n_cited": len(ans.cited_ids)},
        "sources": [
            {"id": s.id, "filename": s.chunk.filename, "page": s.chunk.page_number, "score": s.chunk.score,
             "text": s.chunk.text, "cited": s.cited, "chunk_id": s.chunk.chunk_id}
            for s in ans.sources
        ],
    }


def render_message(msg: dict) -> None:
    if msg["role"] == "user":
        st.markdown(msg["content"].replace("$", "\\$"))
        return
    if msg.get("refused"):
        st.info(msg["content"])
    else:
        st.markdown(render_citations(msg["content"]))
    for w in msg.get("warnings", []):
        st.warning(w)
    sources = msg.get("sources", [])
    if sources:
        label = "Sources" if not msg.get("refused") else "Retrieved passages (not used in an answer)"
        st.caption(f"{label} · expand to read the exact passage")
        for s in sorted(sources, key=lambda x: (not x["cited"], x["id"])):
            tag = "✅ cited" if s["cited"] else "retrieved"
            with st.expander(f"[{s['id']}] {s['filename']} · page {s['page']} · score {s['score']:.2f} · {tag}"):
                st.text(s["text"])
                st.markdown(f"<span class='src-meta'>chunk id: {s['chunk_id']}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
def sidebar(settings):
    ss = st.session_state
    with st.sidebar:
        st.markdown("### 📄 IntelliDocs")
        st.markdown(mode_pill(settings), unsafe_allow_html=True)
        if settings.demo_mode:
            st.caption("Demo mode uses a lexical embedder and quotes the best passage. It does not call any LLM.")
        elif not settings.gemini_api_key:
            st.error("GEMINI_API_KEY is not configured. Add it to `.env` or Streamlit secrets, or set "
                     "`INTELLIDOCS_DEMO_MODE=1` for an offline demo.")

        st.markdown("#### 1 · Upload")
        files = st.file_uploader(
            "PDF documents", type=["pdf"], accept_multiple_files=True, key=f"uploader_{ss.uploader_key}",
            help=f"Up to {settings.max_files} files, {settings.max_upload_mb} MB each.",
        )
        with st.expander("Chunking options"):
            size = st.number_input("Chunk size (words)", 20, 2000, settings.chunk_size_words, 10)
            overlap = st.number_input("Overlap (words)", 0, 500, settings.chunk_overlap_words, 5)
            st.caption("Applies to documents processed after changing it.")
        if st.button("⚙️ Process Documents", type="primary", use_container_width=True, disabled=not files):
            process_files(settings, files, int(size), int(overlap))

        st.markdown("#### 2 · Knowledge base")
        pipe = ss.pipeline
        docs = pipe.documents() if pipe else []
        if docs:
            st.dataframe(
                pd.DataFrame([{"File": d["filename"], "Pages": d["pages_indexed"], "Chunks": d["chunks"]} for d in docs]),
                hide_index=True, use_container_width=True,
            )
            st.caption(f"{len(docs)} document(s) · {sum(d['chunks'] for d in docs)} chunks")
        else:
            st.caption("No documents processed yet.")
        for r in ss.ingest_log:
            {"added": st.success, "duplicate": st.info, "ocr_required": st.warning, "error": st.error}[r["status"]](
                f"**{r['filename']}** - {r['message']}")

        st.markdown("#### 3 · Retrieval settings")
        default_min = 0.10 if settings.demo_mode else settings.min_relevance
        ss["top_k"] = st.slider("Passages to retrieve (top_k)", 1, 10, settings.top_k)
        ss["min_score"] = st.slider(
            "Minimum relevance (cosine similarity)", 0.0, 0.9, float(default_min), 0.01,
            help="Passages scoring below this are discarded. The right value depends on the embedding model; "
                 "tune it (see README → Relevance filtering).",
        )
        c1, c2 = st.columns(2)
        if c1.button("🧹 Clear chat", use_container_width=True):
            ss.messages = []
            st.rerun()
        if c2.button("🗑️ Reset KB", use_container_width=True, help="Delete all indexed documents and the chat."):
            if ss.pipeline:
                ss.pipeline.reset()
            ss.messages, ss.ingest_log = [], []
            ss.uploader_key += 1
            st.rerun()


def process_files(settings, files, size: int, overlap: int) -> None:
    ss = st.session_state
    if overlap >= size:
        st.sidebar.error("Overlap must be smaller than chunk size.")
        return
    if len(files) > settings.max_files:
        st.sidebar.error(f"Please upload at most {settings.max_files} files at once.")
        return
    try:
        pipe = get_pipeline(settings)
    except ConfigError as exc:
        st.sidebar.error(str(exc))
        return
    ss.ingest_log = []
    bar = st.sidebar.progress(0.0, text="Processing…")
    for i, f in enumerate(files):
        bar.progress(i / len(files), text=f"Processing {f.name}…")
        try:
            r = pipe.ingest_pdf(f.name, f.getvalue(), size, overlap)
            ss.ingest_log.append({"filename": r.filename, "status": r.status, "message": r.message})
        except IntelliDocsError as exc:  # e.g. Gemini quota exhausted during embedding
            ss.ingest_log.append({"filename": f.name, "status": "error", "message": str(exc)})
        except Exception:  # pragma: no cover - last-resort guard, no stack trace to the user
            logger.exception("Unexpected ingestion failure")
            ss.ingest_log.append({"filename": f.name, "status": "error", "message": "Unexpected error while processing this file."})
    bar.empty()
    st.rerun()


# --------------------------------------------------------------------------- chat
def chat_tab(settings) -> None:
    ss = st.session_state
    if not ss.messages:
        st.info("👈 Upload one or more PDFs, click **Process Documents**, then ask a question. "
                "Every answer cites the document and page it came from.")
    for m in ss.messages:
        with st.chat_message(m["role"]):
            render_message(m)

    question = st.chat_input("Ask a question about your documents…")
    if not question:
        return
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages if not m.get("refused")]
    user_msg = {"role": "user", "content": question}
    ss.messages.append(user_msg)
    with st.chat_message("user"):
        render_message(user_msg)
    with st.chat_message("assistant"):
        try:
            pipe = get_pipeline(settings)
            with st.spinner("Searching documents and drafting a grounded answer…"):
                ans = pipe.ask(question, history=history, top_k=ss.get("top_k"), min_score=ss.get("min_score"))
            msg = answer_to_message(ans)
        except IntelliDocsError as exc:
            msg = {"role": "assistant", "content": f"⚠️ {exc}", "refused": True, "sources": [], "warnings": [], "meta": {}}
        except Exception:  # pragma: no cover
            logger.exception("Unexpected error while answering")
            msg = {"role": "assistant", "content": "⚠️ Something went wrong while answering. Please try again.",
                   "refused": True, "sources": [], "warnings": [], "meta": {}}
        render_message(msg)
    ss.messages.append(msg)


# --------------------------------------------------------------------------- evaluation dashboard
def evaluation_tab() -> None:
    ss = st.session_state
    st.subheader("This session")
    answered = [m for m in ss.messages if m["role"] == "assistant" and m.get("meta")]
    if answered:
        refused = sum(1 for m in answered if m.get("refused"))
        scores = [m["meta"]["top_score"] for m in answered if m["meta"].get("top_score") is not None]
        covs = [m["meta"]["coverage"] for m in answered if m["meta"].get("coverage") is not None]
        c = st.columns(4)
        c[0].metric("Questions", len(answered))
        c[1].metric("Declined (insufficient evidence)", refused)
        c[2].metric("Avg top retrieval score", f"{sum(scores) / len(scores):.2f}" if scores else "-")
        c[3].metric("Avg citation coverage", f"{sum(covs) / len(covs):.0%}" if covs else "-")
        if scores:
            fig = px.bar(x=list(range(1, len(scores) + 1)), y=scores, labels={"x": "Question #", "y": "Top score"},
                         title="Top retrieval score per question")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("These are unlabelled live statistics, not accuracy: no ground truth exists for your own questions.")
    else:
        st.caption("Ask a question to see session statistics.")

    st.subheader("Benchmark results (sample documents)")
    files = sorted(RESULTS_DIR.glob("*_latest.json"))
    if not files:
        st.info("No evaluation results found. Run `python evaluation/run_evaluation.py` (see README).")
    for path in files:
        data = json.loads(path.read_text())
        with st.expander(f"{data['mode'].upper()} run · {data['timestamp_utc']} · embedder: {data['embedder']}",
                         expanded=True):
            if data.get("note"):
                st.warning(data["note"])
            rows = [{"Metric": k, "Value": v} for k, v in data["summary"].items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            sweep = data.get("threshold_sweep")
            if sweep:
                df = pd.DataFrame(sweep).melt("min_score", var_name="measure", value_name="rate")
                st.plotly_chart(px.line(df, x="min_score", y="rate", color="measure", markers=True,
                                        title="Effect of the relevance threshold (this embedder only)"),
                                use_container_width=True)
            exp = data.get("chunk_size_experiment (retrieval only)")
            if exp:
                st.markdown("**Chunk-size experiment (retrieval only)**")
                st.dataframe(pd.DataFrame(exp), hide_index=True, use_container_width=True)
            st.caption("Limitations: " + " ".join(data.get("limitations", [])))


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    st.markdown(
        '<div class="hero"><h1>IntelliDocs — AI Document Assistant</h1>'
        "<p>Ask questions across your PDFs and get grounded answers with page-level citations "
        "you can verify.</p></div>", unsafe_allow_html=True)
    sidebar(settings)
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])
    with tab_chat:
        chat_tab(settings)
    with tab_eval:
        evaluation_tab()


main()
