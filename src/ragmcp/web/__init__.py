"""Small admin web UI: upload, browse and search. Talks to the MCP server over HTTP,
so it has exactly the permissions of the token it is given (admin)."""

import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..ingest.loaders import SUPPORTED
from .mcp_client import McpError, McpGateway, McpUnavailable

log = logging.getLogger("ragmcp.web")
TEMPLATES = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
TEMPLATES.env.filters["urlq"] = lambda s: quote(str(s), safe="")
DEFAULT_MCP_URL = "http://127.0.0.1:8000/mcp"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
WILDCARDS = {"0.0.0.0", "::", ""}
SNIPPET = 300


def snippet(text: str, n: int = SNIPPET) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


TEMPLATES.env.filters["snippet"] = snippet


def _tags(raw: str) -> list[str] | None:
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    return tags or None


def _origin(url: str) -> tuple[str, str, int]:
    """Scheme + lowercase host + effective port, so quirks like a stripped default
    port or case differences still count as same-origin."""
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower() or "http"
    host = (parts.hostname or "").lower()
    port = parts.port or (443 if scheme == "https" else 80)
    return (scheme, host, port)

def _hostname(netloc: str) -> str:
    return urlsplit(f"//{netloc}").hostname or ""


def create_app(
    gateway: McpGateway | None = None,
    allowed_hosts: set[str] | None = None,
    max_upload_bytes: int = 10 * 1024 * 1024,
) -> FastAPI:
    """allowed_hosts: Host header names accepted (blocks DNS rebinding); None disables the check."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if app.state.gateway is not None and hasattr(app.state.gateway, "aclose"):
            await app.state.gateway.aclose()

    app = FastAPI(title="ragmcp web", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.gateway = gateway

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if allowed_hosts is not None and _hostname(host) not in allowed_hosts:
            return HTMLResponse("Host not allowed", status_code=400)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            # Browsers always send Origin on cross-site POSTs; require same-origin when present (CSRF).
            origin = request.headers.get("origin")
            if origin and _origin(origin) != _origin(f"http://{host}"):
                log.warning("cross-origin refused: origin=%r host=%r", origin, host)
                return HTMLResponse("Cross-origin request refused", status_code=403)
        response = await call_next(request)
        response.headers["X-Frame-Options"] = "DENY"
        # Not "no-referrer": with that policy Chrome sends `Origin: null` even on same-origin form
        # POSTs, which the CSRF check above must (and does) refuse. "same-origin" keeps the real
        # Origin on our own forms and still sends no referrer to other sites.
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        return response

    def gw(request: Request) -> McpGateway:
        g = request.app.state.gateway
        if g is None:
            raise McpUnavailable("web UI is not configured with an MCP server")
        return g

    def page(request: Request, name: str, status: int = 200, **ctx):
        return TEMPLATES.TemplateResponse(request, name, ctx, status_code=status)

    @app.exception_handler(McpError)
    async def _mcp_error(request: Request, exc: McpError):
        return page(request, "error.html", 400, message=str(exc))

    @app.exception_handler(McpUnavailable)
    async def _mcp_down(request: Request, exc: McpUnavailable):
        log.warning("%s", exc)
        return page(request, "error.html", 502, message=str(exc))

    # ---- upload ----

    def upload_page(request: Request, result=None, error=None, status: int = 200):
        return page(request, "upload.html", status, result=result, error=error,
                    accept=",".join(sorted(SUPPORTED)))

    async def ingest(request: Request, tool: str, args: dict):
        try:
            return upload_page(request, result=await gw(request).call(tool, args))
        except McpError as e:  # keep the form on screen with the message
            return upload_page(request, error=str(e), status=400)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/upload", response_class=HTMLResponse)
    async def upload_get(request: Request):
        return upload_page(request)

    @app.post("/upload/file", response_class=HTMLResponse)
    async def upload_file(request: Request, file: UploadFile = File(...),
                          collection: str = Form("default"), tags: str = Form("")):
        name = file.filename or ""
        if Path(name).suffix.lower() not in SUPPORTED:
            return upload_page(request, error=f"Unsupported file type; use one of {sorted(SUPPORTED)}", status=400)
        data = await file.read(max_upload_bytes + 1)
        if len(data) > max_upload_bytes:
            return upload_page(request, error=f"File too large (max {max_upload_bytes} bytes)", status=400)
        return await ingest(request, "ingest_upload", {
            "filename": name, "content_base64": base64.b64encode(data).decode("ascii"),
            "collection": collection.strip() or "default", "tags": _tags(tags)})

    @app.post("/upload/url", response_class=HTMLResponse)
    async def upload_url(request: Request, url: str = Form(...),
                         collection: str = Form("default"), tags: str = Form("")):
        return await ingest(request, "ingest_url", {
            "url": url.strip(), "collection": collection.strip() or "default", "tags": _tags(tags)})

    @app.post("/upload/text", response_class=HTMLResponse)
    async def upload_text(request: Request, text: str = Form(...), source: str = Form(...),
                          title: str = Form(""), collection: str = Form("default"), tags: str = Form("")):
        return await ingest(request, "ingest_text", {
            "text": text, "source": source.strip(), "title": title.strip() or None,
            "collection": collection.strip() or "default", "tags": _tags(tags)})

    # ---- browse ----

    @app.get("/browse", response_class=HTMLResponse)
    async def browse(request: Request, collection: str | None = None, deleted: str | None = None):
        g = gw(request)
        collections = await g.call("list_collections", many=True)
        docs = await g.call("list_documents", many=True)
        counts = {c: 0 for c in collections}
        for d in docs:
            counts[d["collection"]] = counts.get(d["collection"], 0) + 1
        shown = [d for d in docs if d["collection"] == collection] if collection else []
        return page(request, "browse.html", counts=counts, collection=collection, docs=shown, deleted=deleted)

    @app.get("/doc/{doc_id}", response_class=HTMLResponse)
    async def document(request: Request, doc_id: str):
        data = await gw(request).call("list_chunks", {"doc_id": doc_id})
        return page(request, "document.html", doc=data["document"], chunks=data["chunks"])

    @app.post("/doc/{doc_id}/delete")
    async def delete(request: Request, doc_id: str):
        g = gw(request)
        collection = (await g.call("list_chunks", {"doc_id": doc_id}))["document"]["collection"]
        await g.call("delete_document", {"doc_id": doc_id})
        return RedirectResponse(f"/browse?collection={quote(collection, safe='')}&deleted=1", status_code=303)

    @app.get("/chunk/{chunk_id}", response_class=HTMLResponse)
    async def chunk(request: Request, chunk_id: str):
        chunks = await gw(request).call("get_chunk", {"chunk_id": chunk_id, "neighbors": 2}, many=True)
        return page(request, "chunk.html", chunks=chunks, current=chunk_id,
                    doc_id=chunks[0]["doc_id"] if chunks else None)

    # ---- search ----

    @app.get("/search", response_class=HTMLResponse)
    async def search(request: Request, q: str = "", mode: str = "hybrid", top_k: int = 10,
                     collection: str = ""):
        hits = None
        if q.strip():
            args = {"query": q, "mode": mode, "top_k": max(1, min(top_k, 50))}
            if collection.strip():
                args["collection"] = collection.strip()
            hits = await gw(request).call("search", args, many=True)
        return page(request, "search.html", q=q, mode=mode, top_k=top_k, collection=collection, hits=hits)

    return app


def main() -> None:
    import uvicorn

    from ..logging_config import setup_logging

    setup_logging()
    token = os.environ.get("RAGMCP_ADMIN_TOKEN", "").strip()
    if not token:
        raise SystemExit("ragmcp-web: set RAGMCP_ADMIN_TOKEN (an admin token of the MCP server)")
    host = os.environ.get("RAGMCP_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("RAGMCP_WEB_PORT", "8001"))
    url = os.environ.get("RAGMCP_MCP_URL", DEFAULT_MCP_URL)
    extra = {h.strip() for h in os.environ.get("RAGMCP_WEB_ALLOWED_HOSTS", "").split(",") if h.strip()}
    allowed: set[str] | None = LOOPBACK | extra | ({host} if host not in WILDCARDS else set())
    if host in WILDCARDS and not extra:
        allowed = None
    if host not in LOOPBACK:
        log.warning("web UI bound to %s with admin rights and no login: restrict access to trusted machines", host)
    max_upload = int(os.environ.get("RAGMCP_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    app = create_app(McpGateway(url, token), allowed, max_upload)
    uvicorn.run(app, host=host, port=port, log_config=None)
