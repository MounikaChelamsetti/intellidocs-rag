"""Test helpers: a dependency-free minimal PDF writer and scripted fakes."""
import textwrap
from typing import List


def make_pdf(pages: List[str]) -> bytes:
    """Build a valid PDF with one text page per string ('' => blank page without text)."""
    n = len(pages)
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        lines = textwrap.wrap(text, 85) if text else []
        body = "BT /F1 11 Tf 40 750 Td 14 TL "
        for line in lines:
            esc = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            body += f"({esc}) Tj T* "
        body += "ET"
        objs.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n{obj}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


class ScriptedLLM:
    """Returns queued responses in order; records every prompt it receives."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
