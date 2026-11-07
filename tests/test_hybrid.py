"""Hybrid search and reranking with fakes: no database, no model download."""

import pytest

from ragmcp.ingest.pipeline import IngestPipeline
from ragmcp.models import Filters
from ragmcp.retrieve.search import Retriever
from ragmcp.store.memory import MemoryStore

DOCS = {
    "a": "Pajak penghasilan dihitung dari penghasilan neto.",
    "b": "Cukai rokok dikenakan untuk mengendalikan konsumsi.",
    "c": "Nasi goreng memakai kecap manis.",
}


class VecEmbedder:
    """Maps text to a fixed vector by topic word, so vector rank is controllable."""
    dim = 3

    def embed(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            out.append([1.0 if "nasi" in t or "goreng" in t else 0.0,
                        1.0 if "cukai" in t or "rokok" in t else 0.0,
                        1.0 if "pajak" in t else 0.01])
        return out


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def score(self, query, texts):
        self.calls += 1
        return [10.0 if "nasi" in t.lower() else 0.0 for t in texts]


def make(reranker=None, rerank_default=False):
    store = MemoryStore()
    store.init(3)
    emb = VecEmbedder()
    pipe = IngestPipeline(store, emb)
    for name, text in DOCS.items():
        pipe.ingest_text(text, source=name)
    return Retriever(store, emb, reranker, rerank_default)


def test_hybrid_is_default_and_combines_both_signals():
    r = make()
    # Keyword hits only doc b ("rokok"); vector query embeds toward "nasi" (doc c).
    hits = r.search("rokok nasi")
    ids = [h.source for h in hits]
    assert {"b", "c"} <= set(ids)
    assert r.search("rokok", "keyword")[0].source == "b"
    # Hybrid scores are RRF scores, not raw cosine/keyword scores.
    assert all(h.score <= 2 / 61 + 1e-9 for h in hits)


def test_modes_select_single_signal():
    r = make()
    assert [h.source for h in r.search("zzzz", "keyword")] == []
    assert r.search("zzzz", "hybrid")  # vector side still answers
    assert r.search("nasi", "vector")[0].source == "c"


def test_invalid_mode_and_top_k():
    r = make()
    with pytest.raises(ValueError, match="hybrid"):
        r.search("x", "semantic")
    with pytest.raises(ValueError):
        r.search("x", top_k=0)


def test_filters_apply_in_hybrid():
    r = make()
    hits = r.search("pajak rokok nasi", "hybrid", 5, Filters(doc_ids=[r.search("rokok", "keyword")[0].chunk.doc_id]))
    assert {h.source for h in hits} == {"b"}


def test_rerank_off_by_default_and_never_called():
    rr = FakeReranker()
    r = make(rr)
    r.search("rokok")
    assert rr.calls == 0


def test_rerank_reorders_and_respects_top_k():
    rr = FakeReranker()
    r = make(rr)
    hits = r.search("rokok", "keyword", 1, rerank=False)
    assert hits[0].source == "b"
    hits = r.search("pajak rokok nasi", "hybrid", 1, rerank=True)
    assert rr.calls == 1 and len(hits) == 1 and hits[0].source == "c" and hits[0].score == 10.0


def test_rerank_default_on_when_configured_and_can_be_disabled():
    rr = FakeReranker()
    r = make(rr, rerank_default=True)
    assert r.search("pajak rokok nasi", top_k=1)[0].source == "c"
    assert rr.calls == 1
    r.search("pajak rokok nasi", top_k=1, rerank=False)
    assert rr.calls == 1


def test_rerank_requested_without_reranker_errors():
    r = make()
    with pytest.raises(ValueError, match="RAGMCP_RERANK"):
        r.search("rokok", rerank=True)
