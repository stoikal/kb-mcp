"""File loaders. Each returns (title, markdown/plain text)."""

import re
from pathlib import Path

_H1 = re.compile(r"^#\s+(.+?)\s*#*\s*$", re.MULTILINE)

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


def load_file(path: str | Path) -> tuple[str, str]:
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type {ext!r}; supported: {sorted(SUPPORTED)}")
    if ext == ".pdf":
        return p.stem, _load_pdf(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    if ext in (".md", ".markdown"):
        m = _H1.search(text)
        return (m.group(1) if m else p.stem), text
    return p.stem, text


def _load_pdf(p: Path) -> str:
    from pypdf import PdfReader

    pages = [(page.extract_text() or "").strip() for page in PdfReader(str(p)).pages]
    return "\n\n".join(t for t in pages if t)
