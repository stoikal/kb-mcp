"""Auth tests over real Streamable HTTP: a uvicorn server on a free port, real MCP client.

Uses the in-memory store and a fake embedder, so no database or model is needed.
"""

import json
import socket
import threading
import time

import anyio
import httpx2
import pytest
import uvicorn

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ragmcp import server
from ragmcp.auth import RateLimiter, TokenRegistry
from ragmcp.config import Settings
from ragmcp.ingest.pipeline import IngestPipeline
from ragmcp.retrieve.search import Retriever
from ragmcp.store.memory import MemoryStore

ADMIN = "admin-token-0123456789abcdefghij"
READER = "reader-token-0123456789abcdefghi"
ADMIN_TOOLS = {"ingest_text", "ingest_file", "delete_document", "reindex"}
READER_TOOLS = {"search", "get_chunk", "get_document", "list_documents", "list_collections", "stats"}


class FakeEmbedder:
    dim = 3

    def embed(self, texts):
        return [[float(len(t) % 7 + 1), 1.0, 0.5] for t in texts]


@pytest.fixture
def base_url(monkeypatch):
    store = MemoryStore()
    store.init(3)
    emb = FakeEmbedder()
    services = (Retriever(store, emb), IngestPipeline(store, emb))
    monkeypatch.setattr(server, "_services", lambda: services)

    registry = TokenRegistry([ADMIN], [READER])
    app = server.build_http_app(registry)
    # Fresh session manager per test; bucket big enough not to interfere.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    t.join(timeout=5)


def _session(url, token):
    async def run(fn):
        client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"} if token else {})
        async with streamable_http_client(url, http_client=client) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                return await fn(s)

    return lambda fn: anyio.run(run, fn)


def _status(url, headers):
    return httpx2.post(url, headers={"Accept": "application/json, text/event-stream", **headers},
                       json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code


def test_no_token_rejected(base_url):
    assert _status(base_url, {}) == 401


@pytest.mark.parametrize("header", ["Bearer nope", "Bearer", "Basic " + ADMIN, ADMIN, "Bearer " + ADMIN[:-1]])
def test_bad_token_rejected(base_url, header):
    assert _status(base_url, {"Authorization": header}) == 401


def test_reader_sees_only_read_tools(base_url):
    async def fn(s):
        return {t.name for t in (await s.list_tools()).tools}

    names = _session(base_url, READER)(fn)
    assert names == READER_TOOLS


def test_admin_sees_admin_tools_but_not_stdio_only(base_url):
    async def fn(s):
        return {t.name for t in (await s.list_tools()).tools}

    names = _session(base_url, ADMIN)(fn)
    assert names == READER_TOOLS | {"ingest_text", "delete_document", "reindex"}  # ingest_file is stdio-only


@pytest.mark.parametrize("tool,args", [
    ("ingest_text", {"text": "hello", "source": "x"}),
    ("delete_document", {"doc_id": "abc"}),
    ("reindex", {}),
    ("ingest_file", {"path": "/etc/hostname"}),
])
def test_reader_cannot_call_admin_tools(base_url, tool, args):
    async def fn(s):
        return await s.call_tool(tool, args)

    res = _session(base_url, READER)(fn)
    assert res.is_error
    assert "Unknown tool" in res.content[0].text


def test_admin_can_ingest_reindex_delete_and_reader_can_search(base_url):
    async def admin(s):
        r = await s.call_tool("ingest_text", {"text": "# Hi\n\nThe quick brown fox.", "source": "fox"})
        assert not r.is_error, r
        doc_id = json.loads(r.content[0].text)["doc_id"]
        assert not (await s.call_tool("reindex", {})).is_error
        return doc_id

    doc_id = _session(base_url, ADMIN)(admin)

    async def reader(s):
        r = await s.call_tool("search", {"query": "fox", "mode": "keyword"})
        assert not r.is_error
        assert "fox" in r.content[0].text
        return await s.call_tool("list_documents", {})

    assert not _session(base_url, READER)(reader).is_error

    async def delete(s):
        return await s.call_tool("delete_document", {"doc_id": doc_id})

    res = _session(base_url, ADMIN)(delete)
    assert not res.is_error

    async def after(s):
        return await s.call_tool("list_documents", {})

    res = _session(base_url, READER)(after)
    assert not res.is_error and not any("fox" in c.text for c in res.content)


def test_oversized_document_rejected(base_url, monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(max_doc_chars=10))

    async def fn(s):
        return await s.call_tool("ingest_text", {"text": "x" * 11, "source": "big"})

    res = _session(base_url, ADMIN)(fn)
    assert res.is_error and "too large" in res.content[0].text


def test_rate_limit_returns_429_per_token():
    clock = [0.0]
    lim = RateLimiter(per_minute=60, burst=2, clock=lambda: clock[0])
    assert lim.check("a") == 0 and lim.check("a") == 0
    assert lim.check("a") > 0          # burst exhausted
    assert lim.check("b") == 0         # other tokens unaffected
    clock[0] += 1.0                    # 1 req/sec refill
    assert lim.check("a") == 0


def test_http_rate_limit_enforced(monkeypatch):
    monkeypatch.setattr(server, "settings", Settings(rate_per_minute=1, rate_burst=1))
    from starlette.testclient import TestClient

    app = server.build_http_app(TokenRegistry([ADMIN], []))
    h = {"Authorization": f"Bearer {ADMIN}", "Accept": "application/json, text/event-stream"}
    with TestClient(app) as c:
      codes = [c.post("/mcp", headers=h, json={"jsonrpc": "2.0", "id": 1, "method": "ping"}).status_code
               for _ in range(3)]
      assert codes[0] != 429 and codes[-1] == 429
      r = c.post("/mcp", headers=h, json={})
      assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_registry_rejects_short_and_duplicate_tokens():
    with pytest.raises(ValueError):
        TokenRegistry(["short"], [])
    with pytest.raises(ValueError):
        TokenRegistry([ADMIN], [ADMIN])
    assert not TokenRegistry([], [])
