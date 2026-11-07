"""Query-side service: embeds the query, calls store primitives, fuses and reranks.

`mode` is "hybrid" (default: vector + keyword fused with RRF), "vector" or "keyword".
An optional cross-encoder reranks a wider candidate pool before truncating to top_k.
"""

from dataclasses import replace

from ..ingest.embedder import Embedder
from ..models import Chunk, Filters, SearchHit
from ..store.base import KnowledgeStore
from .fusion import rrf
from .rerank import Reranker

MAX_TOP_K = 50
MAX_NEIGHBORS = 10


class Retriever:
    def __init__(
        self,
        store: KnowledgeStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        rerank_default: bool = False,
        rerank_candidates: int = 30,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_default = rerank_default and reranker is not None
        self.rerank_candidates = rerank_candidates

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        filters: Filters | None = None,
        rerank: bool | None = None,
    ) -> list[SearchHit]:
        """rerank: None uses the server default (off unless configured); True needs a reranker."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError(f"unsupported mode {mode!r}; use 'hybrid', 'vector' or 'keyword'")
        if rerank and self.reranker is None:
            raise ValueError("reranking is not enabled on this server (set RAGMCP_RERANK=1)")
        do_rerank = self.rerank_default if rerank is None else rerank

        # Reranking needs a wider pool than top_k; plain hybrid needs enough overlap for RRF.
        pool = max(top_k, self.rerank_candidates) if do_rerank else top_k
        if mode == "hybrid":
            pool = max(pool, top_k * 3)
        if mode == "vector":
            hits = self.store.vector_search(self.embedder.embed([query])[0], pool, filters)
        elif mode == "keyword":
            hits = self.store.keyword_search(query, pool, filters)
        else:
            hits = rrf(
                [
                    self.store.vector_search(self.embedder.embed([query])[0], pool, filters),
                    self.store.keyword_search(query, pool, filters),
                ],
                top_k=pool,
            )
        if do_rerank and hits:
            texts = [f"{h.chunk.heading_path}\n{h.chunk.text}" if h.chunk.heading_path else h.chunk.text
                     for h in hits]
            scores = self.reranker.score(query, texts)
            order = sorted(range(len(hits)), key=lambda i: -scores[i])
            hits = [replace(hits[i], score=scores[i]) for i in order]
        return hits[:top_k]

    def get_chunk(self, chunk_id: str, neighbors: int = 1) -> list[Chunk]:
        if not 0 <= neighbors <= MAX_NEIGHBORS:
            raise ValueError(f"neighbors must be between 0 and {MAX_NEIGHBORS}")
        chunks = self.store.get_chunk(chunk_id, neighbors)
        if not chunks:
            raise ValueError(f"unknown chunk id {chunk_id!r}")
        return chunks
