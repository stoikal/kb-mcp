"""Bearer-token auth, roles and rate limiting for the HTTP transport.

Tokens are never logged or stored in the clear after load: each is kept as a
SHA-256 digest and looked up in constant time. Rate limiting is per token.
stdio has no HTTP request and is treated as a trusted local admin.
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from mcp.types import CallToolResult, TextContent

MIN_TOKEN_LEN = 24

# Tools only admins may see or call. Everything else is available to readers.
ADMIN_TOOLS = frozenset({"ingest_file", "ingest_text", "ingest_url", "ingest_upload", "delete_document", "reindex"})
# Reads the server's disk: only offered over stdio (local development).
STDIO_ONLY_TOOLS = frozenset({"ingest_file"})

PRINCIPAL_KEY = "ragmcp.principal"


class Role(str, Enum):
    READER = "reader"
    ADMIN = "admin"


@dataclass(frozen=True)
class Principal:
    role: Role
    key_id: str  # short digest prefix, safe to log; identifies the token without revealing it


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


class TokenRegistry:
    def __init__(self, admin: list[str], reader: list[str]) -> None:
        self._entries: list[tuple[bytes, Principal]] = []
        seen: set[bytes] = set()
        for role, tokens in ((Role.ADMIN, admin), (Role.READER, reader)):
            for t in tokens:
                if len(t) < MIN_TOKEN_LEN:
                    raise ValueError(f"tokens must be at least {MIN_TOKEN_LEN} characters")
                d = _digest(t)
                if d in seen:
                    raise ValueError("the same token appears twice (or in both roles)")
                seen.add(d)
                self._entries.append((d, Principal(role, d.hex()[:8])))

    @classmethod
    def from_settings(cls, settings) -> "TokenRegistry":
        split = lambda raw: [t.strip() for t in raw.split(",") if t.strip()]  # noqa: E731
        return cls(split(settings.admin_tokens), split(settings.reader_tokens))

    def __bool__(self) -> bool:
        return bool(self._entries)

    def verify(self, token: str) -> Principal | None:
        d = _digest(token)
        found = None
        for known, principal in self._entries:  # no early exit: constant work per attempt
            if hmac.compare_digest(known, d):
                found = principal
        return found


class RateLimiter:
    """Token bucket per key: `burst` immediate requests, refilled at `per_minute`."""

    def __init__(self, per_minute: int, burst: int, clock=time.monotonic) -> None:
        self.rate = per_minute / 60.0
        self.burst = burst
        self.clock = clock
        self._state: dict[str, tuple[float, float]] = {}  # key -> (tokens, last)

    def check(self, key: str) -> float:
        """Return 0 if allowed, else seconds until a request would be allowed."""
        now = self.clock()
        tokens, last = self._state.get(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.rate)
        if tokens >= 1:
            self._state[key] = (tokens - 1, now)
            return 0.0
        self._state[key] = (tokens, now)
        return (1 - tokens) / self.rate if self.rate > 0 else 60.0


async def _reply(send, status: int, message: str, headers: list[tuple[bytes, bytes]] | None = None) -> None:
    body = json.dumps({"error": message}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()), *(headers or [])]})
    await send({"type": "http.response.body", "body": body})


class BearerAuthMiddleware:
    """ASGI middleware: 401 without a valid token, 429 over the rate limit."""

    def __init__(self, app, registry: TokenRegistry, limiter: RateLimiter) -> None:
        self.app, self.registry, self.limiter = app, registry, limiter

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        auth = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
        scheme, _, token = auth.partition(" ")
        principal = self.registry.verify(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if principal is None:
            await _reply(send, 401, "missing or invalid bearer token",
                         [(b"www-authenticate", b'Bearer realm="ragmcp"')])
            return
        wait = self.limiter.check(principal.key_id)
        if wait > 0:
            await _reply(send, 429, "rate limit exceeded",
                         [(b"retry-after", str(max(1, int(wait + 0.999))).encode())])
            return
        scope[PRINCIPAL_KEY] = principal
        await self.app(scope, receive, send)


def principal_of(ctx) -> Principal | None:
    """The caller's principal; stdio (no HTTP request) is a trusted local admin."""
    request = getattr(ctx, "request", None)
    if request is None:
        return Principal(Role.ADMIN, "stdio")
    scope = getattr(request, "scope", None) or {}
    return scope.get(PRINCIPAL_KEY)  # None => fail closed


def _visible(name: str, principal: Principal | None, over_http: bool) -> bool:
    if principal is None:
        return False
    if name in STDIO_ONLY_TOOLS and over_http:
        return False
    return name not in ADMIN_TOOLS or principal.role is Role.ADMIN


async def role_policy(ctx, call_next) -> Any:
    """ServerMiddleware: hide admin tools from readers and refuse their calls."""
    if ctx.method not in ("tools/list", "tools/call"):
        return await call_next(ctx)
    principal = principal_of(ctx)
    over_http = ctx.request is not None
    if ctx.method == "tools/call":
        name = (ctx.params or {}).get("name", "")
        if not _visible(name, principal, over_http):
            # Same answer as a tool that does not exist, so readers learn nothing.
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")], is_error=True)
        return await call_next(ctx)
    result = await call_next(ctx)
    if isinstance(result, dict):
        result = {**result, "tools": [t for t in result["tools"] if _visible(t["name"], principal, over_http)]}
    else:
        result.tools = [t for t in result.tools if _visible(t.name, principal, over_http)]
    return result
