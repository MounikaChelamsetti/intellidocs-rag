import pytest

from src.models import Chunk
from src.vector_store import VectorStore


def chunk(i, doc="d1"):
    return Chunk(f"{doc}-c{i}", doc, "f.pdf", i + 1, i, f"text {i}", 2)


def test_cosine_score_is_one_for_identical_vectors_and_orders_results():
    s = VectorStore()
    s.add_chunks([chunk(0), chunk(1)], [[1.0, 0.0], [0.0, 1.0]])
    res = s.query([1.0, 0.0], 2)
    assert res[0].chunk_id == "d1-c0" and res[0].score == pytest.approx(1.0, abs=1e-5)
    assert res[1].score == pytest.approx(0.0, abs=1e-5)
    assert res[0].page_number == 1 and res[0].filename == "f.pdf"


def test_empty_store_query_returns_nothing():
    assert VectorStore().query([1.0, 0.0], 5) == []


def test_has_document_delete_and_reset():
    s = VectorStore()
    s.add_chunks([chunk(0), chunk(1, "d2")], [[1.0, 0.0], [0.0, 1.0]])
    assert s.has_document("d1") and s.has_document("d2") and not s.has_document("nope")
    s.delete_document("d1")
    assert not s.has_document("d1") and s.count() == 1
    s.reset()
    assert s.count() == 0


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        VectorStore().add_chunks([chunk(0)], [])
