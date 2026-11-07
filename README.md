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

## Web UI

A small admin UI (FastAPI + Jinja2, plain HTML/CSS, no build step) for uploading and browsing knowledge.
It is a client of the running MCP server: it talks to it over Streamable HTTP with the MCP SDK, so the
server's roles, limits and SSRF guard all still apply.

```bash
export RAGMCP_ADMIN_TOKEN=...        # an admin token of the MCP server (required; never stored in code)
uv run ragmcp-web                    # http://127.0.0.1:8001
```

| Page | What it does |
| --- | --- |
| Upload (`/`) | Pick a file (md, txt, pdf, docx, html), ingest a URL, or paste text; shows the ingest result |
| Browse (`/browse`) | Collections, documents per collection, a document's chunks, delete a document |
| Search (`/search`) | Hybrid, vector or keyword search; ranked hits with score and snippet, each linking to the chunk in context |

Settings (environment): `RAGMCP_ADMIN_TOKEN`, `RAGMCP_MCP_URL` (default `http://127.0.0.1:8000/mcp`),
`RAGMCP_WEB_HOST` / `RAGMCP_WEB_PORT` (default `127.0.0.1` / `8001`), `RAGMCP_MAX_UPLOAD_BYTES`,
`RAGMCP_WEB_ALLOWED_HOSTS` (extra `Host` names, comma-separated).

**Security notes**
- The UI has **no login** and holds an admin token: anyone who can reach its port is an admin. Keep it on
  `127.0.0.1` (the default), or only on a machine and network you trust. Binding elsewhere logs a warning.
- Requests with an unexpected `Host` header (DNS rebinding) or a cross-origin `Origin` on a POST (CSRF) are refused.
- Stored text is escaped when rendered, and a strict Content-Security-Policy is sent.
- The UI shares one MCP session, but every page action is still a request counted against the admin token's rate
  limit (`RAGMCP_RATE_PER_MIN` / `RAGMCP_RATE_BURST` on the server); raise them if you browse heavily.
- The UI needs the server's `list_chunks` tool, so run the web UI and server from the same version.
