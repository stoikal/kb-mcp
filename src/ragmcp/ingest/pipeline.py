"""Ingest pipeline: parse -> hash-dedup -> chunk -> embed -> store."""

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..models import Chunk, Document
from ..store.base import KnowledgeStore
from .chunker import approx_tokens, chunk_text
from .embedder import Embedder
from .loaders import load_file

_NS = uuid.UUID("6f1c1d0e-5a3b-4c8e-9a52-0d6b1e3f7a11")


@dataclass
class IngestResult:
    doc_id: str
    status: str  # "created" | "updated" | "unchanged"
    chunks: int


class IngestPipeline:
    def __init__(self, store: KnowledgeStore, embedder: Embedder, chunk_tokens: int = 500, chunk_overlap: int = 60):
        self.store = store
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: str | Path, collection: str = "default", tags: list[str] | None = None) -> IngestResult:
        p = Path(path).expanduser().resolve()
        title, text = load_file(p)
        return self.ingest_text(text, source=str(p), title=title, collection=collection, tags=tags)

    def ingest_text(
        self,
        text: str,
        source: str,
        title: str | None = None,
        collection: str = "default",
        tags: list[str] | None = None,
    ) -> IngestResult:
        tags = tags or []
        # Hash covers text and tags so changing either re-ingests; unchanged -> no embedding cost.
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self.store.find_by_source(source, collection)
        if existing and existing.content_hash == content_hash and sorted(existing.tags) == sorted(tags):
            return IngestResult(existing.id, "unchanged", len(self.store.iter_chunks(existing.id)))

        doc_id = existing.id if existing else str(uuid.uuid5(_NS, f"{collection}\0{source}"))
        raw = chunk_text(text, self.chunk_tokens, self.chunk_overlap, approx_tokens)
        # Embed heading path + text so headings help retrieval; stored text stays raw.
        vecs = self.embedder.embed([f"{c.heading_path}\n{c.text}" if c.heading_path else c.text for c in raw])
        chunks = [
            Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, ordinal=i, text=c.text, heading_path=c.heading_path, embedding=v)
            for i, (c, v) in enumerate(zip(raw, vecs))
        ]
        doc = Document(
            id=doc_id,
            source=source,
            title=title or Path(source).stem,
            content_hash=content_hash,
            collection=collection,
            tags=tags,
            ingested_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.replace_document(doc, chunks)
        return IngestResult(doc_id, "updated" if existing else "created", len(chunks))
