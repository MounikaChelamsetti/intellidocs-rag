import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings  # noqa: E402
from src.offline import HashingEmbedder  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402
from tests.helpers import ScriptedLLM, make_pdf  # noqa: E402

HANDBOOK = [
    "Annual leave: employees receive 27 days of annual leave per year and 3 wellbeing days.",
    "Remote work: staff may work remotely up to 3 days per week with a 400 euro home office stipend.",
]
MANUAL = ["Battery: the NB-5200 battery has a capacity of 5200 mAh and charges in 85 minutes."]


@pytest.fixture
def settings():
    return Settings(min_relevance=0.05).validate()


@pytest.fixture
def make_pipeline(settings):
    def _make(llm=None):
        llm = llm or ScriptedLLM("Employees receive 27 days of annual leave [1].")
        return RAGPipeline(settings, HashingEmbedder(), llm, VectorStore()), llm

    return _make


@pytest.fixture
def handbook_pdf():
    return make_pdf(HANDBOOK)


@pytest.fixture
def manual_pdf():
    return make_pdf(MANUAL)
