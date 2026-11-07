"""Web UI: routes with the MCP layer mocked, plus one end-to-end check against a real server."""

import base64

import anyio
import pytest
from fastapi.testclient import TestClient

from ragmcp.web import create_app
from ragmcp.web.mcp_client import McpError, McpGateway, McpUnavailable

from test_auth import ADMIN, base_url  # noqa: F401  (fixture: real HTTP server on a free port)

DOC = {"doc_id": "d1", "source": "upload:a.md", "title": "Judul <b>x</b>", "collection": "default",
       "tags": ["law"], "ingested_at": "2026-11-07T10:00:00+00:00"}
CH0 = {"chunk_id": "d1:0", "doc_id": "d1", "ordinal": 0, "heading_path": "A", "text": "Satu <script>alert(1)</script>"}
CH1 = {"chunk_id": "d1:1", "doc_id": "d1", "ordinal": 1, "heading_path": "A > B", "text": "Dua"}


class FakeGateway:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    async def call(self, tool, arguments=None, many=False):
        self.calls.append((tool, arguments or {}))
        r = self.responses.get(tool)
        if isinstance(r, Exception):
            raise r
        return r


def make(responses=None, **kw):
    gw = FakeGateway(responses)
    app = create_app(gw, allowed_hosts={"testserver"}, **kw)
    return TestClient(app), gw


def test_pages_render_without_mcp():
    c, gw = make()
    for path in ("/", "/upload"):
        r = c.get(path)
        assert r.status_code == 200 and "Paste text" in r.text
    assert gw.calls == []
    assert r.headers["x-frame-options"] == "DENY" and "default-src 'none'" in r.headers["content-security-policy"]


def test_upload_file_sends_base64_and_shows_result():
    c, gw = make({"ingest_upload": {"doc_id": "d1", "status": "created", "chunks": 3}})
    r = c.post("/upload/file", files={"file": ("../x/notes.md", b"# Hi\n\ntext", "text/markdown")},
               data={"collection": " docs ", "tags": "a, b ,,"})
    assert r.status_code == 200 and "created" in r.text and "/doc/d1" in r.text
    tool, args = gw.calls[0]
    assert tool == "ingest_upload" and args["collection"] == "docs" and args["tags"] == ["a", "b"]
    assert base64.b64decode(args["content_base64"]) == b"# Hi\n\ntext"


def test_upload_file_rejects_bad_type_and_oversize_before_calling_mcp():
    c, gw = make(max_upload_bytes=10)
    assert c.post("/upload/file", files={"file": ("a.exe", b"x")}).status_code == 400
    r = c.post("/upload/file", files={"file": ("a.txt", b"0123456789ab")})
    assert r.status_code == 400 and "too large" in r.text
    assert gw.calls == []


def test_upload_url_and_text_and_error_display():
    c, gw = make({"ingest_url": McpError("URL refused: private address"),
                  "ingest_text": {"doc_id": "d2", "status": "unchanged", "chunks": 1}})
    r = c.post("/upload/url", data={"url": "http://127.0.0.1/"})
    assert r.status_code == 400 and "URL refused" in r.text and 'name="url"' in r.text  # form still there
    r = c.post("/upload/text", data={"text": "hello", "source": "s", "title": ""})
    assert r.status_code == 200 and "unchanged" in r.text
    assert gw.calls[-1] == ("ingest_text", {"text": "hello", "source": "s", "title": None,
                                            "collection": "default", "tags": None})


def test_mcp_down_gives_502_page():
    c, _ = make({"list_collections": McpUnavailable("cannot reach the MCP server at x")})
    r = c.get("/browse")
    assert r.status_code == 502 and "cannot reach" in r.text


def test_browse_lists_collections_and_documents():
    c, gw = make({"list_collections": ["default", "web"], "list_documents": [DOC]})
    r = c.get("/browse")
    assert "default" in r.text and "1 document(s)" in r.text and "0 document(s)" in r.text
    r = c.get("/browse", params={"collection": "default"})
    assert "/doc/d1" in r.text
    assert "<b>x</b>" not in r.text and "&lt;b&gt;x&lt;/b&gt;" in r.text  # titles are escaped


def test_document_view_escapes_content_and_has_delete():
    c, gw = make({"list_chunks": {"document": DOC, "chunks": [CH0, CH1]}})
    r = c.get("/doc/d1")
    assert "<script>alert(1)</script>" not in r.text and "&lt;script&gt;" in r.text
    assert "Dua" in r.text and "A &gt; B" in r.text and "/doc/d1/delete" in r.text
    assert gw.calls == [("list_chunks", {"doc_id": "d1"})]


def test_delete_calls_tool_and_redirects():
    c, gw = make({"list_chunks": {"document": DOC, "chunks": []}, "delete_document": {"deleted": True}})
    r = c.post("/doc/d1/delete", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/browse?collection=default&deleted=1"
    assert ("delete_document", {"doc_id": "d1"}) in gw.calls


def test_doc_id_is_url_encoded_in_links():
    c, _ = make({"ingest_text": {"doc_id": "a/b c", "status": "created", "chunks": 1}})
    r = c.post("/upload/text", data={"text": "t", "source": "s"})
    assert "/doc/a%2Fb%20c" in r.text


def test_chunk_view_marks_selected():
    c, gw = make({"get_chunk": [CH0, CH1]})
    r = c.get("/chunk/d1:1")
    assert "selected" in r.text and "/doc/d1" in r.text
    assert gw.calls == [("get_chunk", {"chunk_id": "d1:1", "neighbors": 2})]


def test_search_ranks_hits_with_score_snippet_and_link():
    long = "kata " * 200
    hits = [{**CH1, "text": long, "score": 0.0325, "source": "upload:a.md", "title": "Judul"},
            {**CH0, "score": 0.01, "source": "upload:a.md", "title": "Judul"}]
    c, gw = make({"search": hits})
    r = c.get("/search", params={"q": "pajak", "mode": "keyword", "top_k": 99, "collection": "default"})
    assert r.text.index("0.0325") < r.text.index("0.0100") and "/chunk/d1%3A1" in r.text and " …" in r.text
    assert long not in r.text
    assert gw.calls == [("search", {"query": "pajak", "mode": "keyword", "top_k": 50, "collection": "default"})]
    assert c.get("/search").status_code == 200 and len(gw.calls) == 1  # empty query: no MCP call


def test_host_and_origin_guards():
    c, gw = make({"delete_document": {}, "list_chunks": {"document": DOC, "chunks": []}})
    assert c.get("/", headers={"host": "evil.example"}).status_code == 400  # DNS rebinding
    r = c.post("/doc/d1/delete", headers={"origin": "http://evil.example"}, follow_redirects=False)
    assert r.status_code == 403 and gw.calls == []  # CSRF
    r = c.post("/doc/d1/delete", headers={"origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 303


@pytest.mark.parametrize("host", ["192.168.8.185:8001", "10.0.0.5:8001", "127.0.0.1:8001", "public.example:8001"])
def test_null_origin_is_refused_for_every_host(host, caplog):
    # `Origin: null` is also what a sandboxed iframe or data: page sends when forging a POST, and the
    # Host header is the *target's* address (private on a LAN), so it must never count as same-origin.
    gw = FakeGateway({"delete_document": {}, "list_chunks": {"document": DOC, "chunks": []}})
    c = TestClient(create_app(gw, allowed_hosts={_h(host)}))
    with caplog.at_level("WARNING", logger="ragmcp.web"):
        r = c.post("/doc/d1/delete", headers={"origin": "null", "host": host}, follow_redirects=False)
    assert r.status_code == 403 and gw.calls == []
    assert "origin='null'" in caplog.text and host in caplog.text


def _h(host):
    return host.rsplit(":", 1)[0]


def test_real_origin_on_lan_host_is_accepted_and_referrer_policy_keeps_it_that_way():
    # The upload bug: Chrome sent Origin: null because we set Referrer-Policy: no-referrer.
    gw = FakeGateway({"ingest_upload": {"doc_id": "d1", "status": "created", "chunks": 1}})
    host = "192.168.8.185:8001"
    c = TestClient(create_app(gw, allowed_hosts={"192.168.8.185"}))
    r = c.get("/", headers={"host": host})
    assert r.headers["referrer-policy"] == "same-origin"
    r = c.post("/upload/file", files={"file": ("a.md", b"# x")},
               headers={"origin": f"http://{host}", "host": host})
    assert r.status_code == 200 and gw.calls[0][0] == "ingest_upload"


def test_unconfigured_gateway_is_a_clean_error():
    c = TestClient(create_app(None, allowed_hosts={"testserver"}))
    assert c.get("/browse").status_code == 502


def test_main_requires_token(monkeypatch):
    from ragmcp.web import main

    monkeypatch.delenv("RAGMCP_ADMIN_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="RAGMCP_ADMIN_TOKEN"):
        main()


# ---- end to end: real gateway against a real HTTP MCP server (memory store, fake embedder) ----

def test_gateway_against_real_server(base_url):
    gw = McpGateway(base_url, ADMIN)

    async def run():
        res = await gw.call("ingest_text", {"text": "# Pajak\n\n## Cukai\n\nCukai rokok.", "source": "s", "collection": "c1"})
        assert res["status"] == "created"
        assert await gw.call("list_collections", many=True) == ["c1"]
        docs = await gw.call("list_documents", {"collection": "c1"}, many=True)
        assert len(docs) == 1 and docs[0]["source"] == "s"
        data = await gw.call("list_chunks", {"doc_id": res["doc_id"]})
        assert data["document"]["doc_id"] == res["doc_id"] and data["chunks"][0]["text"]
        hits = await gw.call("search", {"query": "cukai", "mode": "keyword"}, many=True)
        assert hits and hits[0]["chunk_id"] == data["chunks"][0]["chunk_id"]
        near = await gw.call("get_chunk", {"chunk_id": hits[0]["chunk_id"], "neighbors": 1}, many=True)
        assert near[0]["chunk_id"] == hits[0]["chunk_id"]
        with pytest.raises(McpError, match="refused"):
            await gw.call("ingest_url", {"url": "http://127.0.0.1/"})
        assert await gw.call("delete_document", {"doc_id": res["doc_id"]}) == {"deleted": True}
        assert await gw.call("list_documents", many=True) == []

    anyio.run(run)


def test_gateway_bad_token_and_dead_server(base_url):
    async def run():
        with pytest.raises(McpUnavailable):
            await McpGateway(base_url, "wrong-token-0123456789abcdefghij").call("list_collections", many=True)
        with pytest.raises(McpUnavailable):
            await McpGateway("http://127.0.0.1:9/mcp", ADMIN).call("list_collections")

    anyio.run(run)


def test_gateway_reuses_one_session_within_rate_limit(base_url):
    # 12 calls cost 12 + 4 handshake requests, inside the burst of 20; a session per call would need ~60.
    gw = McpGateway(base_url, ADMIN)

    async def run():
        try:
            for _ in range(12):
                assert await gw.call("list_collections", many=True) == []
        finally:
            await gw.aclose()
        assert gw._session is None and gw._owner is None

    anyio.run(run, backend="asyncio")


def test_gateway_reconnects_after_session_dies(base_url):
    gw = McpGateway(base_url, ADMIN)

    class Dead:
        async def call_tool(self, *a, **k):
            raise RuntimeError("connection closed")

    async def run():
        try:
            await gw.call("list_collections", many=True)
            gw._session = Dead()  # as if the server had restarted under us
            assert await gw.call("list_collections", many=True) == []
            assert not isinstance(gw._session, Dead)
        finally:
            await gw.aclose()

    anyio.run(run, backend="asyncio")


def test_lifespan_closes_gateway():
    closed = []

    class G(FakeGateway):
        async def aclose(self):
            closed.append(True)

    with TestClient(create_app(G(), allowed_hosts={"testserver"})) as c:
        assert c.get("/").status_code == 200
    assert closed == [True]
