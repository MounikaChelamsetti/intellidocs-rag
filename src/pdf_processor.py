"""PDF validation and text extraction with page-level metadata."""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from typing import List

from pypdf import PdfReader

from .errors import PDFProcessingError, PDFValidationError


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str


@dataclass
class ProcessedDocument:
    doc_id: str
    filename: str
    total_pages: int
    pages: List[PageText]
    warnings: List[str] = field(default_factory=list)

    @property
    def non_empty_pages(self) -> List[PageText]:
        return [p for p in self.pages if p.text]

    @property
    def empty_pages(self) -> List[int]:
        return [p.page_number for p in self.pages if not p.text]

    @property
    def needs_ocr(self) -> bool:
        """True when the PDF has pages but no extractable text (likely scanned images)."""
        return self.total_pages > 0 and not self.non_empty_pages


def compute_doc_id(data: bytes) -> str:
    """Content-based ID: identical bytes => identical ID, regardless of filename."""
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def validate_upload(filename: str, data: bytes, max_bytes: int) -> None:
    safe_name = (filename or "").strip()
    if not safe_name.lower().endswith(".pdf"):
        raise PDFValidationError(f"'{safe_name or 'file'}' is not a PDF (only .pdf files are accepted).")
    if not data:
        raise PDFValidationError(f"'{safe_name}' is empty.")
    if len(data) > max_bytes:
        raise PDFValidationError(
            f"'{safe_name}' is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_bytes / 1024 / 1024:.0f} MB."
        )
    if b"%PDF-" not in data[:1024]:
        raise PDFValidationError(f"'{safe_name}' does not look like a valid PDF file.")


def extract_pdf(filename: str, data: bytes, max_bytes: int = 20 * 1024 * 1024, max_pages: int = 300) -> ProcessedDocument:
    """Extract text per page. Never executes anything from the document."""
    validate_upload(filename, data, max_bytes)
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            if not reader.decrypt(""):
                raise PDFProcessingError(f"'{filename}' is password-protected and cannot be read.")
        total = len(reader.pages)
    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(f"'{filename}' is corrupted or could not be parsed.") from exc

    if total == 0:
        raise PDFProcessingError(f"'{filename}' contains no pages.")
    if total > max_pages:
        raise PDFValidationError(f"'{filename}' has {total} pages; the limit is {max_pages}.")

    pages: List[PageText] = []
    warnings: List[str] = []
    for index in range(total):
        number = index + 1
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception:
            raw = ""
            warnings.append(f"Page {number}: text extraction failed.")
        pages.append(PageText(number, normalize_text(raw)))

    doc = ProcessedDocument(compute_doc_id(data), filename, total, pages, warnings)
    if doc.needs_ocr:
        doc.warnings.append(
            "No extractable text found. This looks like a scanned/image-only PDF and needs OCR, "
            "which IntelliDocs does not perform."
        )
    elif doc.empty_pages:
        doc.warnings.append(f"{len(doc.empty_pages)} page(s) had no extractable text: {doc.empty_pages[:10]}")
    return doc
