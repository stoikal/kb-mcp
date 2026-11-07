# RAG MCP Server: Scope and Plan

Date: 2026-11-07

## Goal and scope

This is a standalone MCP server that ingests documents and retrieves cited passages from them. It never writes answers: the client's own LLM does that, so prompting and model choice stay with the client.

- **In scope:** ingest files, URLs and plain text; hybrid (vector + keyword) search with sources; fetch neighbouring chunks; list, delete and reindex documents.
- **Out of scope:** answer generation, a chat UI, per-user private knowledge bases, and OCR for scanned PDFs (added only if needed).
- **Hosting:** you run one server as a service. There is one shared knowledge base: only you (admin) can ingest or delete, and other users get read-only tokens for search. Your database password stays in the server's environment and is never shared.
- **Client controls:** collection, search mode, number of results, filters and whether to rerank. The server makes no LLM calls.
- **Corpus:** mixed PDFs, markdown, HTML, web pages and text, in Indonesian and English.

## Architecture

Ingest and search are separate flows that meet at one storage interface, so the backend can change without touching either.

```
                    ┌───────────────────────────────────┐
                    │ MCP client: owns prompting + LLM  │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │ ragmcp server                     │
                    │ ingest, search, manage; no LLM    │
                    └───────┬───────────────────┬───────┘
                            ▼                   ▼
   ┌───────────────────────────────┐ ┌───────────────────────────────┐
   │ Ingest                        │ │ Search                        │
   │ 1. Parse md/pdf/html/url      │ │ 1. Embed the query (bge-m3)   │
   │ 2. Chunk by headings ~500 tok │ │ 2. Vector + keyword search    │
   │ 3. Skip if hash unchanged     │ │ 3. Fuse with RRF              │
   │ 4. Embed chunks (bge-m3)      │ │ 4. Optional rerank            │
   └───────────────┬───────────────┘ └───────────────▲───────────────┘
              write│chunks                      read │chunks
                   ▼                                 │
   ┌─────────────────────────────────────────────────┴───────────────┐
   │ KnowledgeStore interface: one contract, swappable backends      │
   │ ┌───────────────┐ ┌─────────────────────┐ ┌─────────────────┐   │
   │ │ Memory        │ │ Postgres + pgvector │ │ LanceDB         │   │
   │ │ tests, built  │ │ first real, phase 2 │ │ optional, later │   │
   │ └───────────────┘ └─────────────────────┘ └─────────────────┘   │
   └─────────────────────────────────────────────────────────────────┘
```

Ingest writes chunks through the store and search reads them back. Postgres + pgvector is built next. Every backend must pass the same contract tests.

## MCP interface

The server exposes tools for writing, searching and managing, plus two browsable resources. Only `stats` exists today.

| Tool | What it does |
| --- | --- |
| `ingest_file`, `ingest_url`, `ingest_text` | Parse, chunk, embed and store; unchanged content is skipped by hash |
| `search` | Hybrid, vector or keyword search with filters; optional rerank; returns chunks with source and heading path |
| `get_chunk` | One chunk plus N neighbours on each side, for more context |
| `get_document` | Full stored document |
| `list_documents`, `list_collections` | Browse what is indexed |
| `delete_document` | Remove a document and all its chunks |
| `reindex` | Re-embed stored text, needed after an embedding model change |
| `stats` | Version, configuration and index size (built) |

Resources: `kb://collections` and `kb://doc/{id}`. An optional `answer_with_citations` prompt template is offered, and clients may ignore it.

Design rules: every result carries a source and a locator for citation, every write is idempotent, and long ingests report progress.

**Service rules (hosted mode):**
- Two token roles. **Reader** tokens see only `search`, `get_chunk`, `get_document`, `list_documents`, `list_collections` and `stats`. **Admin** tokens also see the ingest, delete and reindex tools.
- `ingest_file(path)` reads the server's disk, so it is development-only. Hosted ingest takes uploaded content (`ingest_text` or an upload), never a server path.
- `ingest_url` blocks private and loopback addresses (SSRF guard) and limits size and redirects.
- Rate limits per token, and a maximum request and document size.

## Deployment

- The service is reachable on the local network only, by IP address. No tunnel and no internet exposure.
- The server binds to your LAN IP (or `0.0.0.0`) over streamable HTTP, for example `http://192.168.1.20:8000/mcp`. Users add that URL plus their reader token to their MCP client.
- Postgres listens on `localhost` only and is never reachable from the network.
- The machine has a fixed IP (a DHCP reservation in the router, or a static address), so clients' URLs keep working.
- A firewall rule allows the server port only from the local subnet (for example `ufw allow from 192.168.1.0/24 to any port 8000`).
- Traffic is plain HTTP on the LAN, so tokens are readable on the network. That is acceptable on a trusted home network. If a client requires HTTPS, or the network is shared, put Caddy with a local certificate in front, or use Tailscale.
- Tokens are issued, revoked and rotated by you; the database password never leaves the server environment.
- Embedding runs on your machine, so concurrency is bounded by its CPU or GPU. Queries are rate limited, and the embedder is loaded once and kept warm.
- Postgres is backed up regularly (`pg_dump` on a schedule), and the server runs as a systemd service (`uv run ragmcp --transport streamable-http`) that starts on boot and restarts on failure. The app itself is portable Python; only these deployment steps are Linux-specific (systemd, `ufw`). The unit reads its settings (`RAGMCP_STORE_URL`, tokens) from a root-only environment file. Docker is not part of the plan; it can be added later if the server moves to another host.
- Uptime depends on the machine and its network staying on.

**If it runs on Windows instead**
- **Service:** run `uv run ragmcp --transport streamable-http` as a Windows service with NSSM, or as a Task Scheduler task set to start at boot and restart on failure. Settings go in system environment variables or a protected `.env` file.
- **Firewall:** a Windows Defender Firewall inbound rule for the server port, limited to `LocalSubnet` (`New-NetFirewallRule -DisplayName ragmcp -Direction Inbound -LocalPort 8000 -Protocol TCP -RemoteAddress LocalSubnet -Action Allow`).
- **Postgres and `pgvector`:** the standard Windows Postgres installer does not include `pgvector`. It has to be installed from a prebuilt release or compiled with Visual Studio build tools, so check this before choosing Windows. It is the main risk.
- **Backups:** `pg_dump` from a scheduled task instead of cron.
- **Code:** use `pathlib` throughout and avoid shell-specific calls, so the same code runs on both systems. Torch installs on Windows with CPU or CUDA builds.

## Technology choices

Python, local models and an embedded store keep this a single process with no data leaving the machine.

| Part | Choice | Why |
| --- | --- | --- |
| Server | Python, official `mcp` SDK 2.x (`MCPServer`), stdio and streamable HTTP | Best ecosystem for parsing and ML; uv already installed |
| Embeddings | `bge-m3` (~2 GB) via `sentence-transformers` and torch | Multilingual, 8k token input, runs locally |
| Reranker (optional) | `bge-reranker-v2-m3` (~2 GB) via `CrossEncoder` | Better ordering of the top ~30 candidates; same libraries, one more download |
| Storage, first backend | Postgres 17 (local) with `pgvector` | Already running; documents and chunks in one transactional database; HNSW vector index plus `tsvector` full-text index |
| Parsing | `markitdown` or `docling`, `trafilatura` for web pages | PDF, DOCX, HTML to markdown |
| Chunking | Heading-aware, ~500 tokens, ~60 overlap | Keeps the heading path with every chunk |

The project needs one model (`bge-m3`) and a second only if reranking is enabled. There is no generation model.

**Backends stay swappable.** All storage sits behind one `KnowledgeStore` interface. Hybrid fusion (RRF) lives outside it, so ranking logic is the same on every backend. A shared contract test suite must pass for any backend. Postgres is built first (`store/postgres.py`, selected with `RAGMCP_STORE=postgres` and `RAGMCP_STORE_URL`). A LanceDB backend can be added later the same way if a zero-setup option is wanted. Keyword scores will differ slightly between backends.

**Postgres notes.** Use a dedicated database (`ragmcp`) and run `CREATE EXTENSION vector;` once. Full-text search uses the `simple` config because Postgres has no good Indonesian stemmer. The vector column dimension (1024 for `bge-m3`) is fixed at `init`; changing the embedding model means `reindex`.

## Phases and status

Phase 1 and the storage abstraction are done; ingestion is next. The seven phases run in order.

| # | Phase | Delivers | Status |
| --- | --- | --- | --- |
| 1 | Skeleton | uv project, stdio server, `stats` tool, storage interface, in-memory backend, contract tests, RRF fusion | Done |
| 2 | Ingest MVP | md, txt and pdf loaders, chunker, `bge-m3` embedder, `PostgresStore` (pgvector + full-text), hash dedup | Done |
| 3 | Retrieval MVP | `search` (vector and keyword), filters, `get_chunk` with neighbours | Done |
| 4 | Service layer | Streamable HTTP, bearer tokens with admin and reader roles, ingest and delete tools hidden from readers, rate limits | Not started |
| 5 | Hybrid and rerank | Wire RRF into `search`, optional cross-encoder reranker | Not started |
| 6 | More sources | URLs with SSRF guard, HTML, DOCX, uploads, `reindex` | Not started |
| 7 | Deployment | systemd unit and environment file, LAN binding with firewall rule and fixed IP, Postgres backups, logging, retrieval eval set | Not started |

## Verification

Nine contract tests and a stdio smoke test pass today; retrieval quality is measured later with a labeled question set.

- **Contract tests** (done): ranking, filters, replace, delete, neighbour ordering and dimension mismatch. Any new backend must pass the same file unchanged.
- **Chunker and pipeline tests:** heading paths and boundaries are correct; re-ingesting the same file changes nothing; deleting removes every chunk.
- **Integration test:** ingest fixture documents, run `search`, and check the expected chunk is in the top results.
- **Eval set:** labeled questions mapped to expected chunks, scored by recall@5 and MRR, re-run whenever chunk size, model or reranker changes.
- **Auth tests:** no token or a bad token is rejected; a reader token cannot see or call ingest, delete or reindex; an admin token can; `ingest_url` refuses `localhost` and private addresses.
- **Manual check:** register the server in Claude Code, call `ingest_file` then `search`, and confirm results carry sources. MCP Inspector (`npx @modelcontextprotocol/inspector`) checks tool schemas and transports.

## Risks and open questions

The main risks are install size, parsing quality and single-writer storage; none blocks phase 2.

- **Heavy install:** torch and `sentence-transformers` run to several GB, and each model is a ~2 GB download on first use. Load models lazily.
- **Regulation-style documents:** articles (pasal), sub-clauses, cross-references and tables may need a structure-aware chunker beyond heading splits.
- **Scanned PDFs** need OCR, which adds another model. Deferred until a real document needs it.
- **Anyone on the network can reach it:** a guest on the Wi-Fi could otherwise search or ingest. Tokens must be long and random, auth must be on before the server binds to the network, and tokens are never logged.
- **No TLS on the LAN:** tokens travel in plain HTTP. Fine at home; add Caddy or Tailscale if the network is shared or a client demands HTTPS.
- **Home hosting:** uptime depends on your machine and network. Slow CPU embedding under several simultaneous users is the likely bottleneck.
- **Hybrid search is hand-built on Postgres:** a vector index and a `tsvector` index combined with the shared RRF fusion; covered by the contract tests.
- **Keyword ranking** differs slightly between backends, so exact scores are not portable.
- **Indonesian text:** neither backend stems Indonesian well, so retrieval quality there depends mostly on the embedding model.

Open questions:

- [ ] Do you run Ollama or have a GPU? It decides whether the torch install is worth it.
- [x] Postgres 17 runs locally with `pgvector` files installed: build it first (decided).
- [ ] Create the `ragmcp` database and role, and confirm `CREATE EXTENSION vector;` works.
- [ ] Should reranking be on by default once the eval set exists?
- [ ] Which MCP clients will connect? If any requires HTTPS, add Caddy or Tailscale.
- [ ] How many people will use it at once? This sets rate limits and whether CPU embedding is enough.
