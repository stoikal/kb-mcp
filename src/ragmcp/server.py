"""MCP server exposing ingestion and retrieval. No LLM calls happen here."""

import argparse
from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import MCPServer

from . import __version__
from .config import settings
from .ingest.embedder import SentenceTransformerEmbedder
from .models import Chunk, Filters, SearchHit
from .retrieve.search import Retriever
from .store import create_store

EMBED_DIM = 1024  # bge-m3

mcp = MCPServer(
    "ragmcp",
    instructions=(
        "Knowledge-base server. Ingest documents, then search them. "
        "Results are cited chunks; synthesis is up to the client."
    ),
)


@lru_cache(maxsize=1)
def _retriever() -> Retriever:
    """Built on first use so `stats` and tool listing need no database or model."""
    store = create_store(settings)
    store.init(EMBED_DIM)
    return Retriever(store, SentenceTransformerEmbedder(settings.embed_model, EMBED_DIM))


def _chunk_dict(c: Chunk) -> dict:
    return {"chunk_id": c.id, "doc_id": c.doc_id, "ordinal": c.ordinal,
            "heading_path": c.heading_path, "text": c.text}


def _hit_dict(h: SearchHit) -> dict:
    return {**_chunk_dict(h.chunk), "score": h.score, "source": h.source, "title": h.title}


@mcp.tool()
def search(
    query: str,
    mode: Literal["vector", "keyword"] = "vector",
    top_k: int = 5,
    collection: str | None = None,
    tags: list[str] | None = None,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """Search the knowledge base. Returns chunks with source, title and heading path
    for citation. mode: 'vector' (semantic) or 'keyword' (exact terms, any match).
    Filters: collection, tags (match any), doc_ids."""
    filters = Filters(collection=collection, tags=tags, doc_ids=doc_ids)
    return [_hit_dict(h) for h in _retriever().search(query, mode, top_k, filters)]


@mcp.tool()
def get_chunk(chunk_id: str, neighbors: int = 1) -> list[dict]:
    """Fetch a chunk plus `neighbors` chunks on each side (max 10), in document order."""
    return [_chunk_dict(c) for c in _retriever().get_chunk(chunk_id, neighbors)]


@mcp.tool()
def stats() -> dict:
    """Report server version, configuration and (later) index size."""
    return {
        "version": __version__,
        "data_dir": str(settings.data_dir),
        "embed_model": settings.embed_model,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "documents": 0,
        "chunks": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragmcp")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
