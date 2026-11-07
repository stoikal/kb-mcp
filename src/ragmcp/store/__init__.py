"""Backend registry. Backends are imported lazily so unused ones need no deps."""

from importlib import import_module

from ..config import Settings
from .base import KnowledgeStore

# name -> (module, class). Add a backend by adding one line here.
_BACKENDS: dict[str, tuple[str, str]] = {
    "memory": ("ragmcp.store.memory", "MemoryStore"),
    "postgres": ("ragmcp.store.postgres", "PostgresStore"),  # first real backend: phase 2
    "lance": ("ragmcp.store.lance", "LanceStore"),  # optional later
}


def create_store(settings: Settings) -> KnowledgeStore:
    try:
        module, cls = _BACKENDS[settings.store_backend]
    except KeyError:
        raise ValueError(
            f"Unknown RAGMCP_STORE={settings.store_backend!r}; options: {sorted(_BACKENDS)}"
        ) from None
    return getattr(import_module(module), cls)(settings)


__all__ = ["KnowledgeStore", "create_store"]
