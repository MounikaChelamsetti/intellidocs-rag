import pytest

from src.chunker import chunk_document, split_words
from src.errors import ConfigError
from src.pdf_processor import PageText, ProcessedDocument


def make_doc(pages):
    return ProcessedDocument("docid", "f.pdf", len(pages), [PageText(i + 1, t) for i, t in enumerate(pages)])


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_short_page_yields_single_chunk():
    chunks = chunk_document(make_doc([words(50)]), 400, 60)
    assert len(chunks) == 1 and chunks[0].word_count == 50


def test_window_sizes_and_overlap():
    ws = [f"w{i}" for i in range(250)]
    parts = split_words(ws, 100, 20)
    assert [len(p) for p in parts] == [100, 100, 90]
    assert parts[0][-20:] == parts[1][:20]  # configured overlap really shared
    assert parts[-1][-1] == "w249"  # nothing lost at the tail


def test_every_word_is_covered():
    ws = [f"w{i}" for i in range(333)]
    covered = {w for p in split_words(ws, 50, 10) for w in p}
    assert covered == set(ws)


def test_metadata_preserved_and_chunks_do_not_cross_pages():
    doc = make_doc([words(130, "a"), "", words(30, "c")])
    chunks = chunk_document(doc, 100, 10)
    assert [c.page_number for c in chunks] == [1, 1, 3]
    assert all(c.doc_id == "docid" and c.filename == "f.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    assert len({c.chunk_id for c in chunks}) == 3
    assert all(c.text.split()[0].startswith("a") for c in chunks[:2]) and chunks[2].text.startswith("c0")


def test_empty_document_has_no_chunks():
    assert chunk_document(make_doc(["", ""])) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ConfigError):
        split_words(["a"] * 20, size, overlap)
