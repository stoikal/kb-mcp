"""Backend-neutral data types. Nothing here may import a storage library."""

from dataclasses import dataclass, field
from typing import Any, Literal

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass
class Document:
    id: str
    source: str  # file path or URL
    title: str
    content_hash: str  # sha256 of parsed text; drives idempotent re-ingest
    collection: str = "default"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    ingested_at: str = ""  # ISO-8601 UTC, set by the pipeline


@dataclass
class Chunk:
    id: str
    doc_id: str
    ordinal: int  # position in document; neighbours are ordinal +/- n
    text: str  # raw chunk text (returned to clients)
    heading_path: str  # e.g. "Bab II > Pasal 5"
    embedding: list[float] | None = None  # set before upsert, never returned
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float  # higher is better; only comparable within one result list
    source: str
    title: str


@dataclass
class Filters:
    collection: str | None = None
    tags: list[str] | None = None  # match any
    doc_ids: list[str] | None = None
