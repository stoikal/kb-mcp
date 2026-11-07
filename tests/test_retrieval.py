"""Retrieval integration: ingest fixture docs, search, expect the right chunk on top.

Uses the real bge-m3 embedder (~2GB download on first run). Postgres needs dim 1024,
while the rest of the suite uses dim 3, so that backend drops its tables before and
after.
"""

import psycopg
import pytest

from ragmcp.config import Settings
from ragmcp.ingest.embedder import SentenceTransformerEmbedder
from ragmcp.ingest.pipeline import IngestPipeline
from ragmcp.models import Filters
from ragmcp.retrieve.search import Retriever
from ragmcp.store import create_store

from conftest import _guard

DIM = 1024

TAX = """# Pajak

## Penghasilan

Pajak penghasilan orang pribadi dihitung dari penghasilan neto setahun dikurangi penghasilan tidak kena pajak.

## Cukai

Cukai dikenakan atas rokok dan minuman beralkohol untuk mengendalikan konsumsi.
"""
COOK = """# Resep

## Nasi goreng

Panaskan minyak, tumis bawang putih, masukkan nasi dan kecap manis, lalu aduk hingga rata.
"""
ENG = """# Climate

## Sea level

Rising ocean temperatures cause thermal expansion, which raises global sea levels.
"""


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder()


@pytest.fixture(params=["memory", "postgres"])
def rig(request, tmp_path, embedder):
    import os

    def drop():
        with psycopg.connect(os.environ["RAGMCP_STORE_URL"]) as conn:
            _guard(conn)
            conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")

    if request.param == "postgres":
        drop()
    store = create_store(Settings(data_dir=tmp_path, store_backend=request.param,
                                  store_url=os.environ["RAGMCP_STORE_URL"]))
    store.init(DIM)
    pipe = IngestPipeline(store, embedder, chunk_tokens=60, chunk_overlap=5)
    for name, text, kw in [("tax.md", TAX, dict(tags=["law"])), ("cook.md", COOK, dict(collection="food")),
                           ("eng.md", ENG, dict(tags=["science"]))]:
        (tmp_path / name).write_text(text, encoding="utf-8")
        pipe.ingest_file(tmp_path / name, **kw)
    yield Retriever(store, embedder), store
    if request.param == "postgres":
        drop()


def test_vector_search_top_result(rig):
    r, _ = rig
    # Cross-lingual and paraphrased: no shared keywords with the chunk.
    assert r.search("how is excise charged on cigarettes", "vector", 3)[0].chunk.heading_path == "Pajak > Cukai"
    assert r.search("cara memasak nasi goreng", "vector", 3)[0].chunk.heading_path == "Resep > Nasi goreng"
    top = r.search("why do oceans rise as the planet warms", "vector", 3)[0]
    assert top.chunk.heading_path == "Climate > Sea level" and top.source.endswith("eng.md")


def test_keyword_search_top_result(rig):
    r, _ = rig
    assert r.search("rokok", "keyword", 3)[0].chunk.heading_path == "Pajak > Cukai"
    assert r.search("zzzz", "keyword", 3) == []


def test_filters(rig):
    r, _ = rig
    q = "nasi goreng"
    assert r.search(q, "vector", 1, Filters(collection="food"))[0].title == "Resep"
    assert all(h.title != "Resep" for h in r.search(q, "vector", 5, Filters(collection="default")))
    assert {h.title for h in r.search("pajak", "vector", 5, Filters(tags=["law"]))} == {"Pajak"}
    doc_id = r.search("rokok", "keyword", 1)[0].chunk.doc_id
    assert {h.chunk.doc_id for h in r.search("nasi", "vector", 5, Filters(doc_ids=[doc_id]))} == {doc_id}


def test_get_chunk_with_neighbours(rig):
    r, _ = rig
    hit = r.search("rokok", "keyword", 1)[0]
    got = r.get_chunk(hit.chunk.id, 1)
    assert [c.ordinal for c in got] == sorted(c.ordinal for c in got)
    assert hit.chunk.id in [c.id for c in got] and len(got) >= 2
    assert [c.id for c in r.get_chunk(hit.chunk.id, 0)] == [hit.chunk.id]
    with pytest.raises(ValueError):
        r.get_chunk("nope", 1)


def test_argument_validation(rig):
    r, _ = rig
    for bad in [dict(query=" "), dict(query="x", top_k=0), dict(query="x", mode="hybrid")]:
        with pytest.raises(ValueError):
            r.search(**bad)
    with pytest.raises(ValueError):
        r.get_chunk("x", 99)
