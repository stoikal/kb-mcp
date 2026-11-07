"""The single storage contract. Every backend implements `KnowledgeStore`.

Design rules that keep migration painless:
- One interface covers documents AND chunks, so a backend can keep them in one
  transactional database (Postgres) or two engines (LanceDB + SQLite).
- Backends only expose primitive searches (vector, keyword). Hybrid fusion lives
  in `retrieve/fusion.py`, so behaviour is identical on every backend.
- Backends never receive raw SQL/filters; they get the neutral `Filters`.
- Anything that cannot be done identically everywhere does not belong here.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from ..models import Chunk, Document, Filters, SearchHit


class KnowledgeStore(ABC):
    @abstractmethod
    def init(self, embedding_dim: int) -> None:
        """Create tables/indexes if missing. Must raise if an existing store was
        built with a different dimension (embedding model changed -> reindex)."""

    # --- writes -----------------------------------------------------------
    @abstractmethod
    def replace_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        """Atomically (as far as the backend allows) store `doc` and its chunks,
        removing any previous version with the same doc id."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Delete doc and all its chunks. Return False if it did not exist."""

    # --- reads ------------------------------------------------------------
    @abstractmethod
    def get_document(self, doc_id: str) -> Document | None: ...

    @abstractmethod
    def find_by_source(self, source: str, collection: str) -> Document | None:
        """Used by ingest to compare content_hash and skip unchanged files."""

    @abstractmethod
    def list_documents(self, collection: str | None = None) -> list[Document]: ...

    @abstractmethod
    def list_collections(self) -> list[str]: ...

    @abstractmethod
    def get_chunk(self, chunk_id: str, neighbors: int = 0) -> list[Chunk]:
        """The chunk plus `neighbors` chunks on each side, ordered by ordinal."""

    @abstractmethod
    def iter_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks of a doc in order, without embeddings (used by reindex)."""

    # --- retrieval primitives --------------------------------------------
    @abstractmethod
    def vector_search(
        self, embedding: Sequence[float], top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]: ...

    @abstractmethod
    def keyword_search(
        self, query: str, top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]: ...

    # --- admin ------------------------------------------------------------
    @abstractmethod
    def stats(self) -> dict:
        """At least: {"backend": str, "documents": int, "chunks": int}."""
