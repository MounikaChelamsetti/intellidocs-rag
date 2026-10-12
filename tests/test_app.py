"""Headless UI smoke tests using Streamlit's AppTest, in offline demo mode (no API key, no network)."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def demo_env(monkeypatch):
    monkeypatch.setenv("INTELLIDOCS_DEMO_MODE", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import streamlit as st
    st.cache_resource.clear()


def test_app_renders_without_error_in_demo_mode(demo_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("IntelliDocs" in m.value for m in at.markdown)


def test_missing_api_key_shows_clear_message_not_traceback(monkeypatch):
    monkeypatch.delenv("INTELLIDOCS_DEMO_MODE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.config.load_dotenv", lambda *a, **k: None)
    import streamlit as st
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in e.value for e in at.sidebar.error)


def test_question_without_documents_is_handled(demo_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("What is the leave policy?").run()
    assert not at.exception
    assert any("No documents have been processed" in i.value for i in at.info)


def test_answer_shows_citation_and_expandable_source_with_page(demo_env):
    from src.config import SAMPLE_DOCS_DIR, Settings
    from src.offline import ExtractiveDemoLLM, HashingEmbedder
    from src.rag_pipeline import RAGPipeline

    pipe = RAGPipeline(Settings(), HashingEmbedder(), ExtractiveDemoLLM())
    for pdf in SAMPLE_DOCS_DIR.glob("*.pdf"):
        pipe.ingest_pdf(pdf.name, pdf.read_bytes())
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["pipeline"] = pipe
    at.run()
    at.chat_input[0].set_value("What is the maximum flight time of the Nimbus X2?").run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("Nimbus_X2_Drone_Manual.pdf · page 1" in l and "cited" in l for l in labels)
    assert any(":blue-badge[" in m.value for m in at.markdown)


def test_reset_button_clears_knowledge_base(demo_env):
    from src.config import SAMPLE_DOCS_DIR, Settings
    from src.offline import ExtractiveDemoLLM, HashingEmbedder
    from src.rag_pipeline import RAGPipeline

    pipe = RAGPipeline(Settings(), HashingEmbedder(), ExtractiveDemoLLM())
    pipe.ingest_pdf("a.pdf", next(SAMPLE_DOCS_DIR.glob("*.pdf")).read_bytes())
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["pipeline"] = pipe
    at.run()
    next(b for b in at.sidebar.button if "Reset KB" in b.label).click().run()
    assert pipe.store.count() == 0
