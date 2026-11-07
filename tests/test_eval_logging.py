"""Eval metrics (pure), the shipped eval set (real embedder, in-memory store), log redaction."""

import logging

import pytest

from ragmcp.evalset import DEFAULT_DIR, build_retriever, evaluate, load_questions, score_ranking
from ragmcp.ingest.embedder import SentenceTransformerEmbedder
from ragmcp.logging_config import RedactFilter, redact, setup_logging


def test_score_ranking_recall_and_mrr():
    ranked = [["a", "b", "c"], ["x", "y", "z"], ["q", "r", "s"], ["m", "n", "t"]]
    expected = [["a"], ["z"], ["nope"], ["t"]]
    recall, mrr = score_ranking(ranked, expected, k=3)
    assert recall == 3 / 4
    assert mrr == pytest.approx((1 + 1 / 3 + 0 + 1 / 3) / 4)
    recall2, _ = score_ranking(ranked, expected, k=2)
    assert recall2 == 1 / 4
    assert score_ranking([], [], 5) == (0.0, 0.0)


def test_questions_file_is_well_formed():
    qs = load_questions(DEFAULT_DIR / "questions.jsonl")
    assert len(qs) >= 10
    files = {p.name for p in (DEFAULT_DIR / "corpus").iterdir()}
    for q in qs:
        assert q["q"].strip() and q["expected"]
        assert all(lab.split("#", 1)[0] in files and "#" in lab for lab in q["expected"])


def test_shipped_eval_set_meets_floor():
    r = build_retriever(DEFAULT_DIR / "corpus", SentenceTransformerEmbedder())
    qs = load_questions(DEFAULT_DIR / "questions.jsonl")
    hybrid = evaluate(r, qs, "hybrid", 5)
    vector = evaluate(r, qs, "vector", 5)
    assert hybrid.recall >= 0.85 and hybrid.mrr >= 0.7
    assert vector.recall >= 0.9


@pytest.mark.parametrize("raw,secret", [
    ("Authorization: Bearer abcDEF123456789secret", "abcDEF123456789secret"),
    ("connect postgresql://ragmcp:hunter2pw@localhost/ragmcp", "hunter2pw"),
    ("RAGMCP_ADMIN_TOKENS=tok-aaaaaaaaaaaaaaaaaaaaaaaa,tok-b", "tok-aaaaaaaaaaaaaaaaaaaaaaaa"),
])
def test_redact(raw, secret):
    assert secret not in redact(raw) and "[redacted]" in redact(raw)


def test_redact_filter_covers_args_and_tracebacks(tmp_path):
    logfile = tmp_path / "x.log"
    setup_logging("INFO", str(logfile))
    log = logging.getLogger("ragmcp.test")
    log.info("header %s", "Bearer supersecrettoken1234567890")
    try:
        raise RuntimeError("db url postgresql://u:pw123456@h/db")
    except RuntimeError:
        log.exception("boom")
    for h in logging.getLogger().handlers:
        h.flush()
    text = logfile.read_text()
    assert "supersecrettoken" not in text and "pw123456" not in text
    assert "boom" in text and "RuntimeError" in text
    assert any(isinstance(f, RedactFilter) for h in logging.getLogger().handlers for f in h.filters)
