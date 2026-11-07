"""Runtime settings, overridable via RAGMCP_* environment variables."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("RAGMCP_DATA_DIR", Path.cwd() / "data")).expanduser()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)
    embed_model: str = os.environ.get("RAGMCP_EMBED_MODEL", "BAAI/bge-m3")
    store_backend: str = os.environ.get("RAGMCP_STORE", "postgres")
    store_url: str = os.environ.get("RAGMCP_STORE_URL", "")  # e.g. postgresql://...
    chunk_tokens: int = int(os.environ.get("RAGMCP_CHUNK_TOKENS", "500"))
    chunk_overlap: int = int(os.environ.get("RAGMCP_CHUNK_OVERLAP", "60"))

    # Reranking is off by default; enabling it downloads the cross-encoder on first use.
    rerank: bool = os.environ.get("RAGMCP_RERANK", "0").lower() in ("1", "true", "yes", "on")
    rerank_model: str = os.environ.get("RAGMCP_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_candidates: int = int(os.environ.get("RAGMCP_RERANK_CANDIDATES", "30"))

    # Service layer (hosted mode). Tokens are comma-separated; keep them in a root-only env file.
    admin_tokens: str = os.environ.get("RAGMCP_ADMIN_TOKENS", "")
    reader_tokens: str = os.environ.get("RAGMCP_READER_TOKENS", "")
    rate_per_minute: int = int(os.environ.get("RAGMCP_RATE_PER_MIN", "60"))
    rate_burst: int = int(os.environ.get("RAGMCP_RATE_BURST", "20"))
    max_request_bytes: int = int(os.environ.get("RAGMCP_MAX_REQUEST_BYTES", str(16 * 1024 * 1024)))
    max_upload_bytes: int = int(os.environ.get("RAGMCP_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    max_doc_chars: int = int(os.environ.get("RAGMCP_MAX_DOC_CHARS", str(2_000_000)))

    @property
    def lance_dir(self) -> Path:
        return self.data_dir / "lancedb"

    @property
    def meta_db(self) -> Path:
        return self.data_dir / "meta.sqlite"


settings = Settings()
