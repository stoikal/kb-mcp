"""Backend-independent hybrid fusion (Reciprocal Rank Fusion)."""

from ..models import SearchHit


def rrf(result_lists: list[list[SearchHit]], top_k: int, k: int = 60) -> list[SearchHit]:
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(cid, hit)
    ranked = sorted(scores, key=lambda cid: -scores[cid])[:top_k]
    return [SearchHit(best[c].chunk, scores[c], best[c].source, best[c].title) for c in ranked]
