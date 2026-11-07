"""MCP server exposing ingestion and retrieval. No LLM calls happen here."""

import argparse
from functools import lru_cache
from typing import Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .auth import BearerAuthMiddleware, RateLimiter, TokenRegistry, role_policy
from .config import settings
from .ingest.pipeline import IngestPipeline
from .ingest.embedder import SentenceTransformerEmbedder
from .models import Chunk, Document, Filters, SearchHit
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


# Admin-only tools are hidden from readers and refused for them by auth.role_policy.
mcp.middleware.append(role_policy)


@lru_cache(maxsize=1)
def _services() -> tuple[Retriever, IngestPipeline]:
    """Built on first use so `stats` and tool listing need no database or model."""
    store = create_store(settings)
    store.init(EMBED_DIM)
    embedder = SentenceTransformerEmbedder(settings.embed_model, EMBED_DIM)
    pipeline = IngestPipeline(store, embedder, settings.chunk_tokens, settings.chunk_overlap)
    return Retriever(store, embedder), pipeline


def _retriever() -> Retriever:
    return _services()[0]


def _pipeline() -> IngestPipeline:
    return _services()[1]


def _doc_dict(d: Document) -> dict:
    return {"doc_id": d.id, "source": d.source, "title": d.title, "collection": d.collection,
            "tags": d.tags, "ingested_at": d.ingested_at}


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
def get_document(doc_id: str) -> dict:
    """Full stored text of a document, with its metadata."""
    store = _retriever().store
    doc = store.get_document(doc_id)
    if doc is None:
        raise ValueError(f"unknown document id {doc_id!r}")
    return {**_doc_dict(doc), "text": "\n\n".join(c.text for c in store.iter_chunks(doc_id))}


@mcp.tool()
def list_documents(collection: str | None = None) -> list[dict]:
    """List indexed documents, optionally within one collection."""
    return [_doc_dict(d) for d in _retriever().store.list_documents(collection)]


@mcp.tool()
def list_collections() -> list[str]:
    """List collection names."""
    return _retriever().store.list_collections()


# ---- Admin tools (see auth.ADMIN_TOOLS) ----

@mcp.tool()
def ingest_text(text: str, source: str, title: str | None = None,
                collection: str = "default", tags: list[str] | None = None) -> dict:
    """Parse, chunk, embed and store text. Re-ingesting unchanged text is a no-op.
    `source` is a label used for citation and idempotency (e.g. a name or URL)."""
    if len(text) > settings.max_doc_chars:
        raise ToolError(f"document too large: {len(text)} chars (max {settings.max_doc_chars})")
    r = _pipeline().ingest_text(text, source, title, collection, tags)
    return {"doc_id": r.doc_id, "status": r.status, "chunks": r.chunks}


@mcp.tool()
def ingest_file(path: str, collection: str = "default", tags: list[str] | None = None) -> dict:
    """Development only (stdio): ingest a file from the server's disk."""
    r = _pipeline().ingest_file(path, collection, tags)
    return {"doc_id": r.doc_id, "status": r.status, "chunks": r.chunks}


@mcp.tool()
def delete_document(doc_id: str) -> dict:
    """Remove a document and all its chunks."""
    return {"deleted": _retriever().store.delete_document(doc_id)}


@mcp.tool()
def reindex() -> dict:
    """Re-embed all stored text (needed after an embedding model change)."""
    return {"documents": _pipeline().reindex()}


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


def build_http_app(registry: TokenRegistry, host: str = "127.0.0.1"):
    """Streamable HTTP app with bearer auth and per-token rate limiting in front."""
    app = mcp.streamable_http_app(host=host, max_request_body_size=settings.max_request_bytes)
    limiter = RateLimiter(settings.rate_per_minute, settings.rate_burst)
    app.add_middleware(BearerAuthMiddleware, registry=registry, limiter=limiter)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragmcp")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http bind address")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return
    registry = TokenRegistry.from_settings(settings)
    if not registry:  # refuse to expose an unauthenticated server on the network
        parser.error("streamable-http needs RAGMCP_ADMIN_TOKENS and/or RAGMCP_READER_TOKENS")
    import uvicorn

    uvicorn.run(build_http_app(registry, args.host), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
