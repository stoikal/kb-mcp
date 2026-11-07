"""Phase 6: loaders (HTML, DOCX), uploads, ingest_url with its SSRF guard, reindex.
No network, database or model: fetching uses a mock transport and a fake resolver."""

import base64
import io

import httpx2
import pytest
from docx import Document as Docx
from mcp.server.mcpserver.exceptions import ToolError

from ragmcp import server
from ragmcp.config import Settings
from ragmcp.ingest.fetch import BlockedURL, check_url, safe_fetch
from ragmcp.ingest.loaders import load_bytes
from ragmcp.ingest.pipeline import IngestPipeline
from ragmcp.retrieve.search import Retriever
from ragmcp.store.memory import MemoryStore

PUBLIC = "93.184.216.34"
HTML = ("<html><head><title>Halaman Uji</title></head><body><nav>menu menu</nav>"
        "<h1>Judul Utama</h1><p>Paragraf pertama berisi cukup banyak kata supaya ekstraksi berhasil dengan baik.</p>"
        "<h2>Bagian Dua</h2><p>Paragraf kedua juga cukup panjang dan berisi informasi tentang pajak penghasilan.</p>"
        "</body></html>")


class Emb:
    dim = 3
    version = 1

    def embed(self, texts):
        return [[float(self.version), float(len(t) % 5), 1.0] for t in texts]


def docx_bytes() -> bytes:
    d = Docx()
    d.add_heading("Peraturan Uji", 1)
    d.add_paragraph("Pasal pertama berisi ketentuan umum tentang pajak.")
    d.add_heading("Bab Dua", 2)
    d.add_paragraph("Butir daftar", style="List Bullet")
    t = d.add_table(rows=2, cols=2)
    for c, v in zip(t.rows[0].cells + t.rows[1].cells, ["Kolom A", "Kolom B", "1", "2"]):
        c.text = v
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


# ---- loaders ----

def test_html_loader_extracts_main_content():
    title, text = load_bytes("page.html", HTML.encode())
    assert "Paragraf pertama" in text and "menu menu" not in text
    assert text.startswith("# ") and title


def test_docx_loader_headings_lists_tables():
    title, text = load_bytes("x.docx", docx_bytes())
    assert title == "Peraturan Uji"
    assert "# Peraturan Uji" in text and "## Bab Dua" in text and "- Butir daftar" in text
    assert "| Kolom A | Kolom B |" in text and "| 1 | 2 |" in text


def test_loader_uses_basename_only_and_rejects_unknown_types():
    title, _ = load_bytes("../../etc/passwd.txt", b"hello")
    assert title == "passwd"
    with pytest.raises(ValueError):
        load_bytes("evil.exe", b"x")
    with pytest.raises(ValueError):
        load_bytes("empty.html", b"<html></html>")


# ---- SSRF guard ----

def resolver(mapping):
    return lambda host, port: mapping[host]


@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://127.1/", "http://2130706433/", "http://0.0.0.0/",
    "http://[::1]/", "http://10.0.0.5/", "http://192.168.1.1/", "http://172.16.0.1/",
    "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/", "http://[::ffff:127.0.0.1]/",
    "http://[fe80::1]/", "http://[fc00::1]/", "http://224.0.0.1/",
])
def test_private_literals_blocked(url):
    with pytest.raises(BlockedURL):
        check_url(url, resolver({"localhost": ["127.0.0.1"], "127.1": ["127.0.0.1"], "2130706433": ["127.0.0.1"]}))


@pytest.mark.parametrize("url", ["ftp://example.com/", "file:///etc/passwd", "gopher://x/", "http://user:pw@example.com/", "http:///x"])
def test_scheme_and_credentials_blocked(url):
    with pytest.raises(BlockedURL):
        check_url(url, resolver({"example.com": [PUBLIC]}))


def test_hostname_resolving_to_private_blocked_even_if_one_address_is_public():
    r = resolver({"evil.test": [PUBLIC, "10.0.0.1"], "ok.test": [PUBLIC]})
    with pytest.raises(BlockedURL):
        check_url("http://evil.test/", r)
    assert check_url("https://ok.test/x", r) == ("https", "ok.test", 443, PUBLIC)


def make_client(handler):
    return httpx2.Client(transport=httpx2.MockTransport(handler), follow_redirects=False)


def test_fetch_pins_resolved_ip_and_sets_host():
    seen = {}

    def handler(req):
        seen["url"], seen["host"] = str(req.url), req.headers["host"]
        return httpx2.Response(200, headers={"content-type": "text/html; charset=utf-8"}, content=HTML.encode())

    body, name, final = safe_fetch("http://ok.test/docs/page", resolver=resolver({"ok.test": [PUBLIC]}),
                                   client=make_client(handler))
    assert PUBLIC in seen["url"] and seen["host"] == "ok.test"
    assert name == "page.html" and b"Judul Utama" in body


def test_redirect_to_private_address_blocked():
    def handler(req):
        return httpx2.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    with pytest.raises(BlockedURL):
        safe_fetch("http://ok.test/", resolver=resolver({"ok.test": [PUBLIC]}), client=make_client(handler))


def test_redirect_to_hostname_that_resolves_private_blocked():
    def handler(req):
        return httpx2.Response(302, headers={"location": "http://internal.test/"})

    with pytest.raises(BlockedURL):
        safe_fetch("http://ok.test/", resolver=resolver({"ok.test": [PUBLIC], "internal.test": ["192.168.0.9"]}),
                   client=make_client(handler))


def test_redirect_limit_size_limit_and_content_type():
    r = resolver({"ok.test": [PUBLIC]})
    loop = make_client(lambda req: httpx2.Response(302, headers={"location": "/again"}))
    with pytest.raises(BlockedURL, match="redirects"):
        safe_fetch("http://ok.test/", resolver=r, client=loop)
    big = make_client(lambda req: httpx2.Response(200, headers={"content-type": "text/plain"}, content=b"x" * 100))
    with pytest.raises(BlockedURL, match="too large"):
        safe_fetch("http://ok.test/a.txt", resolver=r, client=big, max_bytes=10)
    binary = make_client(lambda req: httpx2.Response(200, headers={"content-type": "application/zip"}, content=b"PK"))
    with pytest.raises(BlockedURL, match="content type"):
        safe_fetch("http://ok.test/a.zip", resolver=r, client=binary)


# ---- server tools (called directly; auth/visibility is covered in test_auth) ----

@pytest.fixture
def svc(monkeypatch):
    store = MemoryStore()
    store.init(3)
    emb = Emb()
    services = (Retriever(store, emb), IngestPipeline(store, emb))
    monkeypatch.setattr(server, "_services", lambda: services)
    return store, emb, services[1]


def test_ingest_upload_docx_and_size_limits(svc, monkeypatch):
    store, _, _ = svc
    res = server.ingest_upload("../x/reg.docx", base64.b64encode(docx_bytes()).decode(), tags=["law"])
    assert res["status"] == "created" and res["chunks"] >= 1
    doc = store.list_documents()[0]
    assert doc.source == "upload:reg.docx" and doc.tags == ["law"]
    assert server.ingest_upload("../x/reg.docx", base64.b64encode(docx_bytes()).decode(), tags=["law"])["status"] == "unchanged"
    with pytest.raises(ToolError, match="base64"):
        server.ingest_upload("a.txt", "!!!not base64!!!")
    monkeypatch.setattr(server, "settings", Settings(max_upload_bytes=5))
    with pytest.raises(ToolError, match="too large"):
        server.ingest_upload("a.txt", base64.b64encode(b"0123456789").decode())
    monkeypatch.setattr(server, "settings", Settings(max_doc_chars=5))
    with pytest.raises(ToolError, match="too large"):
        server.ingest_upload("a.txt", base64.b64encode(b"0123456789").decode())


def test_ingest_url_tool_refuses_private_and_ingests_public(svc, monkeypatch):
    store, _, _ = svc
    for bad in ["http://localhost:8000/mcp", "http://127.0.0.1/", "http://192.168.1.10/admin",
                "http://169.254.169.254/", "file:///etc/passwd"]:
        with pytest.raises(ToolError, match="refused"):
            server.ingest_url(bad)
    assert store.list_documents() == []

    def fake_fetch(url, **kw):
        return HTML.encode(), "page.html", url

    monkeypatch.setattr(server, "safe_fetch", fake_fetch)
    res = server.ingest_url("https://ok.test/page", collection="web")
    assert res["status"] == "created"
    assert store.list_documents("web")[0].source == "https://ok.test/page"


# ---- reindex ----

def test_reindex_reembeds_every_chunk(svc):
    store, emb, pipe = svc
    pipe.ingest_text("# A\n\nsatu dua tiga", source="a")
    pipe.ingest_text("# B\n\nempat lima enam", source="b")
    before = [c.embedding for c in store._chunks.values()]
    emb.version = 2
    assert server.reindex() == {"documents": 2}
    after = [c.embedding for c in store._chunks.values()]
    assert len(after) == len(before) and all(a[0] == 2.0 for a in after) and all(b[0] == 1.0 for b in before)
