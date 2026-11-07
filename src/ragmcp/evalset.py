"""Retrieval eval: labeled questions mapped to expected chunks, scored by recall@k and MRR.

A label is "<file name>#<heading path>" (chunk ids are content-derived, so headings are the
stable handle). Run:  uv run ragmcp-eval [--k 5] [--rerank] [--min-recall 0.8]
Uses the real embedder on an in-memory store, so it never touches your database.
Re-run whenever chunk size, embedding model or reranker changes.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import settings
from .ingest.embedder import Embedder, SentenceTransformerEmbedder
from .ingest.pipeline import IngestPipeline
from .models import SearchHit
from .retrieve.rerank import CrossEncoderReranker, Reranker
from .retrieve.search import Retriever
from .store.memory import MemoryStore

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "eval"


def label_of(hit: SearchHit) -> str:
    return f"{Path(hit.source).name}#{hit.chunk.heading_path}"


@dataclass
class Scores:
    recall: float
    mrr: float
    n: int
    misses: list[str]


def score_ranking(ranked: list[list[str]], expected: list[list[str]], k: int) -> tuple[float, float]:
    """ranked[i] is the label list returned for question i. Recall@k: fraction of questions with
    any expected label in the top k. MRR: mean reciprocal rank of the first expected label."""
    hits, rr = 0, 0.0
    for got, want in zip(ranked, expected):
        pos = next((i for i, lab in enumerate(got) if lab in want), None)
        if pos is not None and pos < k:
            hits += 1
        if pos is not None:
            rr += 1.0 / (pos + 1)
    n = len(expected)
    return (hits / n if n else 0.0), (rr / n if n else 0.0)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate(retriever: Retriever, questions: list[dict], mode: str, k: int = 5,
             rerank: bool | None = None) -> Scores:
    ranked = [[label_of(h) for h in retriever.search(q["q"], mode, k, rerank=rerank)] for q in questions]
    expected = [q["expected"] for q in questions]
    recall, mrr = score_ranking(ranked, expected, k)
    misses = [q["q"] for q, got in zip(questions, ranked) if not set(got[:k]) & set(q["expected"])]
    return Scores(recall, mrr, len(questions), misses)


def build_retriever(corpus: Path, embedder: Embedder, reranker: Reranker | None = None) -> Retriever:
    store = MemoryStore()
    store.init(embedder.dim)
    pipe = IngestPipeline(store, embedder, settings.chunk_tokens, settings.chunk_overlap)
    for f in sorted(corpus.iterdir()):
        if f.is_file():
            pipe.ingest_file(f)
    return Retriever(store, embedder, reranker)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ragmcp-eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="folder with corpus/ and questions.jsonl")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", default="vector,keyword,hybrid")
    ap.add_argument("--rerank", action="store_true", help="also score each mode with the cross-encoder")
    ap.add_argument("--min-recall", type=float, default=None, help="exit 1 if hybrid recall@k is below this")
    ap.add_argument("-v", "--verbose", action="store_true", help="list missed questions")
    args = ap.parse_args(argv)

    questions = load_questions(args.dir / "questions.jsonl")
    reranker = CrossEncoderReranker(settings.rerank_model) if args.rerank else None
    retriever = build_retriever(args.dir / "corpus", SentenceTransformerEmbedder(settings.embed_model), reranker)

    print(f"{len(questions)} questions, k={args.k}")
    print(f"{'config':<18}{'recall@' + str(args.k):>10}{'MRR':>8}")
    hybrid_recall = None
    for mode in args.modes.split(","):
        for rr in ([False, True] if args.rerank else [False]):
            s = evaluate(retriever, questions, mode, args.k, rerank=rr if args.rerank else None)
            name = mode + ("+rerank" if rr else "")
            print(f"{name:<18}{s.recall:>10.3f}{s.mrr:>8.3f}")
            if args.verbose:
                for q in s.misses:
                    print(f"    miss: {q}")
            if mode == "hybrid" and not rr:
                hybrid_recall = s.recall
    if args.min_recall is not None and (hybrid_recall or 0.0) < args.min_recall:
        print(f"FAIL: hybrid recall@{args.k} {hybrid_recall} < {args.min_recall}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
