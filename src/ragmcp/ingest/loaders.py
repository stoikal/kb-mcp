"""Loaders. Each returns (title, markdown/plain text)."""

import io
import re
from pathlib import Path

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)

SUPPORTED = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm", ".docx"}


def load_file(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    return load_bytes(p.name, p.read_bytes())


def load_bytes(filename: str, data: bytes) -> tuple[str, str]:
    """Parse in-memory content, choosing the parser by filename extension. Only the
    basename is used, so a client-supplied name can never reach the filesystem."""
    name = Path(filename.replace("\\", "/")).name
    ext = Path(name).suffix.lower()
    stem = Path(name).stem or "document"
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    if ext == ".pdf":
        return stem, load_pdf(data)
    if ext == ".docx":
        return load_docx(data, stem)
    text = data.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        return load_html(text, stem)
    if ext in (".md", ".markdown"):
        m = _H1.search(text)
        return (m.group(1) if m else stem), text
    return stem, text


def load_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    pages = [(page.extract_text() or "").strip() for page in PdfReader(io.BytesIO(data)).pages]
    return "\n\n".join(t for t in pages if t)


def load_html(html: str, fallback_title: str) -> tuple[str, str]:
    """Main content of a web page as markdown (boilerplate such as nav and footers dropped)."""
    import trafilatura

    text = trafilatura.extract(html, output_format="markdown", include_tables=True, favor_recall=True)
    if not text or not text.strip():
        raise ValueError("no extractable text found in HTML")
    meta = trafilatura.extract_metadata(html)
    return (meta.title if meta and meta.title else fallback_title), text.strip()


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("\n", " ").replace("|", "\\|").strip() for c in cells) + " |"


def load_docx(data: bytes, fallback_title: str) -> tuple[str, str]:
    """DOCX to markdown: Heading N -> '#' * N, tables -> pipe tables, in document order."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    parts: list[str] = []
    title = None
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name if para.style is not None else "") or ""
            m = re.fullmatch(r"Heading (\d)", style)
            if style == "Title":
                title = title or text
                parts.append(f"# {text}")
            elif m:
                level = min(int(m.group(1)), 6)
                if level == 1:
                    title = title or text
                parts.append(f"{'#' * level} {text}")
            elif style.startswith("List"):
                parts.append(f"- {text}")
            else:
                parts.append(text)
        elif tag == "tbl":
            rows = [[c.text for c in r.cells] for r in Table(child, doc).rows]
            if rows:
                parts.append("\n".join([_md_row(rows[0]), _md_row(["---"] * len(rows[0])),
                                        *(_md_row(r) for r in rows[1:])]))
    if not parts:
        raise ValueError("no extractable text found in DOCX")
    return title or fallback_title, "\n\n".join(parts)
