import pytest

from src.errors import PDFProcessingError, PDFValidationError
from src.pdf_processor import compute_doc_id, extract_pdf, normalize_text
from tests.helpers import make_pdf


def test_extracts_text_with_correct_page_numbers():
    doc = extract_pdf("a.pdf", make_pdf(["alpha page one", "bravo page two", "charlie page three"]))
    assert doc.total_pages == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert "bravo" in doc.pages[1].text and "alpha" not in doc.pages[1].text


def test_blank_page_in_middle_is_reported_not_dropped():
    doc = extract_pdf("a.pdf", make_pdf(["first page text", "", "third page text"]))
    assert doc.empty_pages == [2]
    assert [p.page_number for p in doc.non_empty_pages] == [1, 3]
    assert not doc.needs_ocr
    assert any("no extractable text" in w for w in doc.warnings)


def test_pdf_without_any_text_is_flagged_as_needing_ocr():
    doc = extract_pdf("scan.pdf", make_pdf(["", ""]))
    assert doc.needs_ocr and doc.non_empty_pages == []
    assert any("OCR" in w for w in doc.warnings)


def test_malformed_pdf_raises_processing_error():
    with pytest.raises(PDFProcessingError):
        extract_pdf("bad.pdf", b"%PDF-1.4\nthis is not really a pdf at all")


@pytest.mark.parametrize(
    "name,data",
    [("notes.txt", b"%PDF-1.4"), ("empty.pdf", b""), ("fake.pdf", b"hello world, not a pdf")],
)
def test_validation_rejects_bad_uploads(name, data):
    with pytest.raises(PDFValidationError):
        extract_pdf(name, data)


def test_size_limit_enforced():
    with pytest.raises(PDFValidationError, match="limit"):
        extract_pdf("big.pdf", make_pdf(["x " * 50]), max_bytes=100)


def test_page_limit_enforced():
    with pytest.raises(PDFValidationError, match="pages"):
        extract_pdf("long.pdf", make_pdf(["a b c"] * 4), max_pages=3)


def test_doc_id_is_content_based():
    a, b = make_pdf(["same"]), make_pdf(["same"])
    assert compute_doc_id(a) == compute_doc_id(b) != compute_doc_id(make_pdf(["different"]))


def test_normalize_text_collapses_whitespace_and_nulls():
    assert normalize_text("a \t b\x00c \n\n  d ") == "a b c\nd"
