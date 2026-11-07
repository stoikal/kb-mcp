import hashlib
import re

import pytest

from ragmcp.config import Settings
from ragmcp.ingest.loaders import load_file
from ragmcp.ingest.pipeline import IngestPipeline
from ragmcp.models import Filters
from ragmcp.store import create_store

DIM = 3


class FakeEmbedder:
    """Deterministic hashed bag-of-words into 3 dims; no model download."""

    dim = DIM

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            out.append(v)
        return out


@pytest.fixture(params=["memory", "postgres"])
def env(request, tmp_path):
    store = create_store(Settings(data_dir=tmp_path, store_backend=request.param))
    store.init(DIM)
    emb = FakeEmbedder()
    return store, emb, IngestPipeline(store, emb, chunk_tokens=40, chunk_overlap=5), tmp_path


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


DOC = "# Pajak\n\n## Penghasilan\n\n" + "pajak penghasilan orang pribadi. " * 20 + "\n\n## Cukai\n\ncukai rokok dan minuman."


def test_ingest_then_reingest_is_noop(env):
    store, emb, pipe, tmp = env
    p = write(tmp, "a.md", DOC)
    r1 = pipe.ingest_file(p)
    assert r1.status == "created" and r1.chunks >= 2
    calls = emb.calls
    r2 = pipe.ingest_file(p)
    assert r2.status == "unchanged" and r2.doc_id == r1.doc_id and r2.chunks == r1.chunks
    assert emb.calls == calls  # no re-embedding
    assert store.stats()["documents"] == 1 and store.stats()["chunks"] == r1.chunks


def test_changed_content_replaces_chunks(env):
    store, _, pipe, tmp = env
    p = write(tmp, "a.md", DOC)
    r1 = pipe.ingest_file(p)
    write(tmp, "a.md", "# Baru\n\nisi baru saja.")
    r2 = pipe.ingest_file(p)
    assert r2.status == "updated" and r2.doc_id == r1.doc_id and r2.chunks == 1
    assert store.stats()["chunks"] == 1
    assert store.keyword_search("rokok", 5) == []


def test_delete_removes_every_chunk(env):
    store, _, pipe, tmp = env
    r = pipe.ingest_file(write(tmp, "a.md", DOC))
    assert store.delete_document(r.doc_id) is True
    assert store.stats() == {"backend": store.stats()["backend"], "documents": 0, "chunks": 0}
    assert store.iter_chunks(r.doc_id) == []


def test_search_finds_expected_chunk_with_heading(env):
    store, emb, pipe, tmp = env
    pipe.ingest_file(write(tmp, "a.md", DOC))
    pipe.ingest_file(write(tmp, "b.txt", "resep nasi goreng dengan telur."), collection="food")
    hits = store.keyword_search("rokok", 3)
    assert hits[0].chunk.heading_path == "Pajak > Cukai"
    assert hits[0].title == "Pajak"
    top = store.vector_search(emb.embed(["Pajak > Cukai\ncukai rokok dan minuman."])[0], 1)
    assert top[0].chunk.heading_path == "Pajak > Cukai"
    assert {h.chunk.doc_id for h in store.keyword_search("nasi", 3, Filters(collection="food"))}


def test_tag_change_reingests(env):
    _, _, pipe, tmp = env
    p = write(tmp, "a.md", DOC)
    pipe.ingest_file(p, tags=["x"])
    assert pipe.ingest_file(p, tags=["y"]).status == "updated"


def test_loaders(tmp_path):
    assert load_file(write(tmp_path, "n.txt", "halo"))[1] == "halo"
    assert load_file(write(tmp_path, "m.md", "# Judul\n\nisi"))[0] == "Judul"
    with pytest.raises(ValueError):
        load_file(write(tmp_path, "x.xyz", "x"))


def test_pdf_loader(tmp_path):
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    p = tmp_path / "blank.pdf"
    with p.open("wb") as f:
        w.write(f)
    assert load_file(p) == ("blank", "")  # blank page: no text, no crash
