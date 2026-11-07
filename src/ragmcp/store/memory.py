"""In-memory reference backend. Used by tests and as a readable spec of the
contract; not persistent."""

import math
import re
from collections.abc import Sequence
from dataclasses import replace

from ..config import Settings
from ..models import Chunk, Document, Filters, SearchHit
from .base import KnowledgeStore

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


class MemoryStore(KnowledgeStore):
    def __init__(self, settings: Settings | None = None) -> None:
        self._docs: dict[str, Document] = {}
        self._chunks: dict[str, Chunk] = {}
        self._dim: int | None = None

    def init(self, embedding_dim: int) -> None:
        if self._dim not in (None, embedding_dim):
            raise ValueError(f"store built with dim {self._dim}, got {embedding_dim}")
        self._dim = embedding_dim

    def replace_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        self.delete_document(doc.id)
        self._docs[doc.id] = doc
        for c in chunks:
            if c.embedding is not None and len(c.embedding) != self._dim:
                raise ValueError("embedding dimension mismatch")
            self._chunks[c.id] = c

    def delete_document(self, doc_id: str) -> bool:
        existed = self._docs.pop(doc_id, None) is not None
        for cid in [c.id for c in self._chunks.values() if c.doc_id == doc_id]:
            del self._chunks[cid]
        return existed

    def get_document(self, doc_id: str) -> Document | None:
        return self._docs.get(doc_id)

    def find_by_source(self, source: str, collection: str) -> Document | None:
        return next(
            (d for d in self._docs.values() if d.source == source and d.collection == collection),
            None,
        )

    def list_documents(self, collection: str | None = None) -> list[Document]:
        return [d for d in self._docs.values() if collection in (None, d.collection)]

    def list_collections(self) -> list[str]:
        return sorted({d.collection for d in self._docs.values()})

    def _ordered(self, doc_id: str) -> list[Chunk]:
        return sorted((c for c in self._chunks.values() if c.doc_id == doc_id), key=lambda c: c.ordinal)

    def get_chunk(self, chunk_id: str, neighbors: int = 0) -> list[Chunk]:
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return []
        lo, hi = chunk.ordinal - neighbors, chunk.ordinal + neighbors
        return [self._strip(c) for c in self._ordered(chunk.doc_id) if lo <= c.ordinal <= hi]

    def iter_chunks(self, doc_id: str) -> list[Chunk]:
        return [self._strip(c) for c in self._ordered(doc_id)]

    @staticmethod
    def _strip(c: Chunk) -> Chunk:
        return replace(c, embedding=None)

    def _allowed(self, c: Chunk, f: Filters | None) -> bool:
        if f is None:
            return True
        d = self._docs[c.doc_id]
        if f.collection and d.collection != f.collection:
            return False
        if f.doc_ids and d.id not in f.doc_ids:
            return False
        if f.tags and not set(f.tags) & set(d.tags):
            return False
        return True

    def _hit(self, c: Chunk, score: float) -> SearchHit:
        d = self._docs[c.doc_id]
        return SearchHit(chunk=self._strip(c), score=score, source=d.source, title=d.title)

    def vector_search(
        self, embedding: Sequence[float], top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]:
        def cos(a: Sequence[float], b: Sequence[float]) -> float:
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0

        scored = [
            (cos(embedding, c.embedding), c)
            for c in self._chunks.values()
            if c.embedding is not None and self._allowed(c, filters)
        ]
        scored.sort(key=lambda t: -t[0])
        return [self._hit(c, s) for s, c in scored[:top_k]]

    def keyword_search(
        self, query: str, top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]:
        q = set(_tokens(query))
        scored = []
        for c in self._chunks.values():
            if not self._allowed(c, filters):
                continue
            overlap = len(q & set(_tokens(c.text)))
            if overlap:
                scored.append((float(overlap), c))
        scored.sort(key=lambda t: -t[0])
        return [self._hit(c, s) for s, c in scored[:top_k]]

    def stats(self) -> dict:
        return {"backend": "memory", "documents": len(self._docs), "chunks": len(self._chunks)}
