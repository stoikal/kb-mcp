"""Postgres + pgvector backend. Keyword search uses tsvector with the `simple`
config (no usable Indonesian stemmer in Postgres)."""

import re
from collections.abc import Sequence
from contextlib import contextmanager

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from ..config import Settings
from ..models import Chunk, Document, Filters, SearchHit
from .base import KnowledgeStore

_WORD = re.compile(r"\w+", re.UNICODE)

_DOC_COLS = "d.id, d.source, d.title, d.content_hash, d.collection, d.tags, d.metadata, d.ingested_at"
_CHUNK_COLS = "c.id, c.doc_id, c.ordinal, c.text, c.heading_path, c.metadata"


def _doc(r) -> Document:
    return Document(id=r[0], source=r[1], title=r[2], content_hash=r[3], collection=r[4],
                    tags=list(r[5]), metadata=r[6] or {}, ingested_at=r[7])


def _chunk(r) -> Chunk:
    return Chunk(id=r[0], doc_id=r[1], ordinal=r[2], text=r[3], heading_path=r[4], metadata=r[5] or {})


class PostgresStore(KnowledgeStore):
    def __init__(self, settings: Settings) -> None:
        if not settings.store_url:
            raise ValueError("RAGMCP_STORE_URL is required for the postgres backend")
        self._url = settings.store_url
        self._dim: int | None = None

    @contextmanager
    def _conn(self, vectors: bool = True):
        with psycopg.connect(self._url) as conn:  # commits on success, rolls back on error
            if vectors:
                register_vector(conn)
            yield conn

    def init(self, embedding_dim: int) -> None:
        with self._conn(vectors=False) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            existing = conn.execute(
                "SELECT a.atttypmod FROM pg_attribute a WHERE a.attrelid = to_regclass('chunks') "
                "AND a.attname = 'embedding' AND NOT a.attisdropped"
            ).fetchone()
            if existing is not None:
                if existing[0] != embedding_dim:
                    raise ValueError(
                        f"store built with dim {existing[0]}, got {embedding_dim}; reindex required"
                    )
            else:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS documents (
                        id text PRIMARY KEY, source text NOT NULL, title text NOT NULL,
                        content_hash text NOT NULL, collection text NOT NULL,
                        tags text[] NOT NULL DEFAULT '{}', metadata jsonb NOT NULL DEFAULT '{}',
                        ingested_at text NOT NULL DEFAULT '',
                        UNIQUE (collection, source))"""
                )
                conn.execute(
                    f"""CREATE TABLE IF NOT EXISTS chunks (
                        id text PRIMARY KEY,
                        doc_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                        ordinal integer NOT NULL, text text NOT NULL, heading_path text NOT NULL,
                        metadata jsonb NOT NULL DEFAULT '{{}}',
                        embedding vector({int(embedding_dim)}),
                        tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED,
                        UNIQUE (doc_id, ordinal))"""
                )
                conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS chunks_emb_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
                )
        self._dim = embedding_dim

    # --- writes -----------------------------------------------------------
    def replace_document(self, doc: Document, chunks: Sequence[Chunk]) -> None:
        for c in chunks:
            if c.embedding is not None and self._dim is not None and len(c.embedding) != self._dim:
                raise ValueError("embedding dimension mismatch")
        with self._conn() as conn:  # single transaction
            conn.execute("DELETE FROM documents WHERE id = %s", (doc.id,))
            conn.execute(
                "INSERT INTO documents (id, source, title, content_hash, collection, tags, metadata, ingested_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                (doc.id, doc.source, doc.title, doc.content_hash, doc.collection, doc.tags,
                 Jsonb(doc.metadata), doc.ingested_at),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (id, doc_id, ordinal, text, heading_path, metadata, embedding) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [(c.id, c.doc_id, c.ordinal, c.text, c.heading_path, Jsonb(c.metadata),
                     None if c.embedding is None else Vector(c.embedding))
                     for c in chunks],
                )

    def delete_document(self, doc_id: str) -> bool:
        with self._conn(vectors=False) as conn:
            return conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,)).rowcount > 0

    # --- reads ------------------------------------------------------------
    def get_document(self, doc_id: str) -> Document | None:
        with self._conn(vectors=False) as conn:
            r = conn.execute(f"SELECT {_DOC_COLS} FROM documents d WHERE d.id = %s", (doc_id,)).fetchone()
        return _doc(r) if r else None

    def find_by_source(self, source: str, collection: str) -> Document | None:
        with self._conn(vectors=False) as conn:
            r = conn.execute(
                f"SELECT {_DOC_COLS} FROM documents d WHERE d.source = %s AND d.collection = %s",
                (source, collection),
            ).fetchone()
        return _doc(r) if r else None

    def list_documents(self, collection: str | None = None) -> list[Document]:
        with self._conn(vectors=False) as conn:
            rows = conn.execute(
                f"SELECT {_DOC_COLS} FROM documents d WHERE (%s::text IS NULL OR d.collection = %s) "
                "ORDER BY d.ingested_at, d.id",
                (collection, collection),
            ).fetchall()
        return [_doc(r) for r in rows]

    def list_collections(self) -> list[str]:
        with self._conn(vectors=False) as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT collection FROM documents ORDER BY 1")]

    def get_chunk(self, chunk_id: str, neighbors: int = 0) -> list[Chunk]:
        with self._conn(vectors=False) as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks c JOIN chunks t ON t.doc_id = c.doc_id "
                "WHERE t.id = %s AND c.ordinal BETWEEN t.ordinal - %s AND t.ordinal + %s ORDER BY c.ordinal",
                (chunk_id, neighbors, neighbors),
            ).fetchall()
        return [_chunk(r) for r in rows]

    def iter_chunks(self, doc_id: str) -> list[Chunk]:
        with self._conn(vectors=False) as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks c WHERE c.doc_id = %s ORDER BY c.ordinal", (doc_id,)
            ).fetchall()
        return [_chunk(r) for r in rows]

    # --- retrieval primitives --------------------------------------------
    @staticmethod
    def _where(f: Filters | None) -> tuple[str, list]:
        conds, params = [], []
        if f:
            if f.collection:
                conds.append("d.collection = %s")
                params.append(f.collection)
            if f.doc_ids:
                conds.append("d.id = ANY(%s)")
                params.append(list(f.doc_ids))
            if f.tags:
                conds.append("d.tags && %s")
                params.append(list(f.tags))
        return (" AND ".join(conds) if conds else "TRUE"), params

    @staticmethod
    def _hits(rows) -> list[SearchHit]:
        # row = chunk cols (6) + score, source, title
        return [SearchHit(chunk=_chunk(r), score=float(r[6]), source=r[7], title=r[8]) for r in rows]

    def vector_search(
        self, embedding: Sequence[float], top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]:
        where, params = self._where(filters)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT {_CHUNK_COLS}, 1 - (c.embedding <=> %s::vector) AS score, d.source, d.title "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id "
                f"WHERE c.embedding IS NOT NULL AND {where} ORDER BY c.embedding <=> %s::vector LIMIT %s",
                [Vector(embedding), *params, Vector(embedding), top_k],
            ).fetchall()
        return self._hits(rows)

    def keyword_search(
        self, query: str, top_k: int, filters: Filters | None = None
    ) -> list[SearchHit]:
        terms = [t.lower() for t in _WORD.findall(query)]
        if not terms:
            return []
        where, params = self._where(filters)
        with self._conn(vectors=False) as conn:
            # OR semantics (any term matches), ranked by ts_rank; same intent as MemoryStore.
            rows = conn.execute(
                f"WITH q AS (SELECT to_tsquery('simple', %s) AS q) "
                f"SELECT {_CHUNK_COLS}, ts_rank(c.tsv, q.q) AS score, d.source, d.title "
                f"FROM chunks c JOIN documents d ON d.id = c.doc_id, q "
                f"WHERE c.tsv @@ q.q AND {where} ORDER BY score DESC, c.id LIMIT %s",
                [" | ".join(terms), *params, top_k],
            ).fetchall()
        return self._hits(rows)

    # --- admin ------------------------------------------------------------
    def stats(self) -> dict:
        with self._conn(vectors=False) as conn:
            docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"backend": "postgres", "documents": docs, "chunks": chunks}
