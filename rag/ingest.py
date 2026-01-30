from pypdf import PdfReader
from docx import Document as DocxDocument

def extract_pdf(path: str) -> list[tuple[int, str]]:
    """Return list of (page_no, text)."""
    reader = PdfReader(path)
    out: list[tuple[int, str]] = []
    for idx, page in enumerate(reader.pages):
        txt = (page.extract_text() or "").strip()
        if txt:
            out.append((idx + 1, txt))
    return out

def extract_docx(path: str) -> list[tuple[int, str]]:
    """Return list of (block_no, text). DOCX doesn't expose stable page numbers; we use paragraph blocks."""
    doc = DocxDocument(path)
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    block = 1
    for i, p in enumerate(doc.paragraphs, start=1):
        t = (p.text or "").strip()
        if t:
            buf.append(t)
        # every 8 paragraphs -> one block
        if i % 8 == 0 and buf:
            out.append((block, "\n".join(buf)))
            buf = []
            block += 1
    if buf:
        out.append((block, "\n".join(buf)))
    return out
