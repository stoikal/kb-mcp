"""Heading-aware chunker.

Splits text into sections by markdown headings, then packs paragraphs of each
section into chunks of ~`max_tokens` with ~`overlap` tokens carried over between
consecutive chunks of the same section. Each chunk keeps its heading path.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def approx_tokens(text: str) -> int:
    """Cheap tokenizer-free estimate (no model load needed for chunking)."""
    return len(_TOKEN.findall(text))


@dataclass
class RawChunk:
    text: str
    heading_path: str


def _sections(text: str) -> list[tuple[str, str]]:
    """[(heading_path, body)] in document order; preamble has empty path."""
    out: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        else:
            buf.append(line)
    flush()
    return out


def _split_long(par: str, max_tokens: int, count: Callable[[str], int]) -> list[str]:
    """Break an oversized paragraph on sentences, then on words."""
    if count(par) <= max_tokens:
        return [par]
    pieces: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", par):
        if count(sent) <= max_tokens:
            pieces.append(sent)
        else:
            words = sent.split()
            cur: list[str] = []
            for w in words:
                cur.append(w)
                if count(" ".join(cur)) >= max_tokens:
                    pieces.append(" ".join(cur))
                    cur = []
            if cur:
                pieces.append(" ".join(cur))
    return pieces


def _tail(text: str, overlap: int, count: Callable[[str], int]) -> str:
    """Last ~overlap tokens of text, cut on word boundaries."""
    if overlap <= 0:
        return ""
    words = text.split()
    out: list[str] = []
    for w in reversed(words):
        out.insert(0, w)
        if count(" ".join(out)) >= overlap:
            break
    return " ".join(out)


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap: int = 60,
    count: Callable[[str], int] = approx_tokens,
) -> list[RawChunk]:
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")
    chunks: list[RawChunk] = []
    for path, body in _sections(text):
        paras: list[str] = []
        for par in re.split(r"\n\s*\n", body):
            par = par.strip()
            if par:
                paras.extend(_split_long(par, max_tokens, count))
        cur: list[str] = []
        cur_tokens = 0
        for par in paras:
            n = count(par)
            if cur and cur_tokens + n > max_tokens:
                joined = "\n\n".join(cur)
                chunks.append(RawChunk(joined, path))
                tail = _tail(joined, overlap, count)
                cur = [tail] if tail else []
                cur_tokens = count(tail) if tail else 0
            cur.append(par)
            cur_tokens += n
        if cur:
            chunks.append(RawChunk("\n\n".join(cur), path))
    return chunks
