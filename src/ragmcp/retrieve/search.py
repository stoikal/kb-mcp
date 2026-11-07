"""Query-side service: embeds the query and calls a store primitive.

Hybrid (RRF over both primitives) arrives in phase 5; until then `mode` is
"vector" or "keyword".
"""

from ..ingest.embedder import Embedder
from ..models import Chunk, Filters, SearchHit
from ..store.base import KnowledgeStore

MAX_TOP_K = 50
MAX_NEIGHBORS = 10


class Retriever:
    def __init__(self, store: KnowledgeStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def search(
        self, query: str, mode: str = "vector", top_k: int = 5, filters: Filters | None = None
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if mode == "vector":
            return self.store.vector_search(self.embedder.embed([query])[0], top_k, filters)
        if mode == "keyword":
            return self.store.keyword_search(query, top_k, filters)
        raise ValueError(f"unsupported mode {mode!r}; use 'vector' or 'keyword'")

    def get_chunk(self, chunk_id: str, neighbors: int = 1) -> list[Chunk]:
        if not 0 <= neighbors <= MAX_NEIGHBORS:
            raise ValueError(f"neighbors must be between 0 and {MAX_NEIGHBORS}")
        chunks = self.store.get_chunk(chunk_id, neighbors)
        if not chunks:
            raise ValueError(f"unknown chunk id {chunk_id!r}")
        return chunks
