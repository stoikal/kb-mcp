# ragmcp

A standalone MCP server that ingests documents (md, txt, html, pdf, docx, URLs, uploads) and returns cited
passages via hybrid (vector + keyword) search with optional cross-encoder reranking. It makes no LLM calls:
the client writes the answers. See `PLAN.md` for scope and design, `docs/DEPLOYMENT.md` to run it as a service.

```bash
uv sync
uv run ragmcp                                   # stdio (local, trusted admin)
RAGMCP_ADMIN_TOKENS=... RAGMCP_READER_TOKENS=... \
  uv run ragmcp --transport streamable-http --host 192.168.1.20 --port 8000
uv run pytest                                   # needs Postgres test DB ragmcp_test (see tests/conftest.py)
uv run ragmcp-eval -v                           # retrieval quality: recall@5 and MRR
```

Settings are `RAGMCP_*` environment variables; `deploy/ragmcp.env.example` lists them all.
