"""MCP server exposing ingestion and retrieval. No LLM calls happen here."""

import argparse

from mcp.server.mcpserver import MCPServer

from . import __version__
from .config import settings

mcp = MCPServer(
    "ragmcp",
    instructions=(
        "Knowledge-base server. Ingest documents, then search them. "
        "Results are cited chunks; synthesis is up to the client."
    ),
)


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
