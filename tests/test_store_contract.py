"""Every KnowledgeStore backend must pass these tests unchanged.

To validate a new backend (lance, postgres), add it to BACKENDS below.
"""

import pytest

from ragmcp.config import Settings
from ragmcp.models import Chunk, Document, Filters
from ragmcp.retrieve.fusion import rrf
from ragmcp.store import KnowledgeStore, create_store

BACKENDS = ["memory", "postgres"]  # later: "lance"
DIM = 3


@pytest.fixture(params=BACKENDS)
def store(request, tmp_path) -> KnowledgeStore:
    s = create_store(Settings(data_dir=tmp_path, store_backend=request.param))
    s.init(DIM)
    return s


def make_doc(doc_id="d1", source="a.md", collection="default", tags=None) -> Document:
    return Document(
        id=doc_id, source=source, title=doc_id, content_hash="h", collection=collection, tags=tags or []
    )


def make_chunks(doc_id, texts, vecs) -> list[Chunk]:
    return [
        Chunk(id=f"{doc_id}:{i}", doc_id=doc_id, ordinal=i, text=t, heading_path="H", embedding=v)
        for i, (t, v) in enumerate(zip(texts, vecs))
    ]


def seed(store):
    store.replace_document(
        make_doc("d1", tags=["tax"]),
        make_chunks("d1", ["pajak penghasilan", "bea masuk impor", "cukai rokok"],
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    store.replace_document(
        make_doc("d2", "b.md", "other"),
        make_chunks("d2", ["pajak daerah"], [[1, 0.1, 0]]),
    )


def test_vector_search_ranks_nearest_first(store):
    seed(store)
    hits = store.vector_search([0, 1, 0], top_k=2)
    assert hits[0].chunk.id == "d1:1"
    assert hits[0].chunk.embedding is None  # embeddings never leak to clients
    assert hits[0].source == "a.md"


def test_keyword_search(store):
    seed(store)
    ids = {h.chunk.id for h in store.keyword_search("pajak", top_k=5)}
    assert ids == {"d1:0", "d2:0"}


def test_filters(store):
    seed(store)
    assert {h.chunk.doc_id for h in store.keyword_search("pajak", 5, Filters(collection="other"))} == {"d2"}
    assert {h.chunk.doc_id for h in store.keyword_search("pajak", 5, Filters(tags=["tax"]))} == {"d1"}


def test_replace_removes_old_chunks(store):
    seed(store)
    store.replace_document(make_doc("d1"), make_chunks("d1", ["baru"], [[1, 1, 1]]))
    assert [c.id for c in store.iter_chunks("d1")] == ["d1:0"]
    assert store.stats()["chunks"] == 2  # 1 from d1 + 1 from d2


def test_delete(store):
    seed(store)
    assert store.delete_document("d1") is True
    assert store.delete_document("d1") is False
    assert store.get_document("d1") is None
    assert store.keyword_search("rokok", 5) == []


def test_neighbors_are_ordered_and_clamped(store):
    seed(store)
    assert [c.ordinal for c in store.get_chunk("d1:1", neighbors=1)] == [0, 1, 2]
    assert [c.ordinal for c in store.get_chunk("d1:0", neighbors=1)] == [0, 1]
    assert store.get_chunk("missing") == []


def test_find_by_source_and_collections(store):
    seed(store)
    assert store.find_by_source("a.md", "default").id == "d1"
    assert store.find_by_source("a.md", "other") is None
    assert store.list_collections() == ["default", "other"]


def test_dimension_mismatch_rejected(store):
    with pytest.raises(ValueError):
        store.init(DIM + 1)


def test_hybrid_fusion_over_any_backend(store):
    seed(store)
    fused = rrf([store.vector_search([1, 0, 0], 3), store.keyword_search("pajak", 3)], top_k=2)
    assert fused[0].chunk.id in {"d1:0", "d2:0"}
    assert len(fused) == 2
