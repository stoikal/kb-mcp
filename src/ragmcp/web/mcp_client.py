"""Thin async gateway to the running ragmcp server over Streamable HTTP.

One short-lived MCP session per call: simple and robust for a low-traffic admin UI.
The bearer token is held in memory only and is never logged or rendered.
"""

import asyncio
import json
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class McpError(Exception):
    """The server answered with a tool error (message is safe to show)."""


class McpUnavailable(Exception):
    """Could not reach or authenticate with the MCP server."""


def parse_result(res, many: bool = False) -> Any:
    """Tool results arrive as structured content or as JSON text blocks. A list result is one
    block per item, so a one-item list looks like a dict: callers say which they expect
    (`many=True` returns a list, otherwise the single value)."""
    texts = [c.text for c in res.content if getattr(c, "type", "") == "text"]
    if res.is_error:
        raise McpError(texts[0] if texts else "tool call failed")
    sc = res.structured_content
    if isinstance(sc, dict):
        sc = sc["result"] if set(sc) == {"result"} else sc
        return sc if not many or isinstance(sc, list) else [sc]
    values = []
    for t in texts:
        try:
            values.append(json.loads(t) if t[:1] in "{[" else t)
        except ValueError:
            values.append(t)
    if many:
        return values
    return values[0] if values else {}


class McpGateway:
    """One long-lived MCP session shared by all requests (a session per call would cost ~5 HTTP
    requests and trip the server's per-token rate limit). A background task owns the session's
    async context managers, so they are entered and exited in the same task; if the session
    dies (server restart) the next call reconnects once."""

    CONNECT_TIMEOUT = 30.0

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self._token = token
        self._session: ClientSession | None = None
        self._owner: asyncio.Task | None = None
        self._stop: asyncio.Event | None = None
        self._lock: asyncio.Lock | None = None

    async def _own(self, ready: asyncio.Future, stop: asyncio.Event) -> None:
        client = httpx2.AsyncClient(headers={"Authorization": f"Bearer {self._token}"}, timeout=120)
        try:
            async with client, streamable_http_client(self.url, http_client=client) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    self._session = session
                    ready.set_result(None)
                    await stop.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError("cancelled"))
            if not isinstance(e, Exception):
                raise
        finally:
            self._session = None

    async def _connect(self) -> ClientSession:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._session is not None:
                return self._session
            ready = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._owner = asyncio.create_task(self._own(ready, self._stop))
            try:
                await asyncio.wait_for(ready, self.CONNECT_TIMEOUT)
            except BaseException:
                await self._disconnect()
                raise
            return self._session  # type: ignore[return-value]

    async def _disconnect(self) -> None:
        owner, stop = self._owner, self._stop
        self._owner = self._stop = self._session = None
        if stop is not None:
            stop.set()
        if owner is not None:
            try:
                await asyncio.wait_for(owner, 5)
            except BaseException:  # already failed or stuck: nothing more to do
                owner.cancel()

    async def aclose(self) -> None:
        await self._disconnect()

    async def call(self, tool: str, arguments: dict | None = None, many: bool = False) -> Any:
        last: Exception | None = None
        for _ in range(2):  # second attempt = fresh session
            try:
                session = await self._connect()
                return parse_result(await session.call_tool(tool, arguments or {}), many)
            except McpError:
                raise
            except Exception as e:  # incl. ExceptionGroup, httpx errors, 401s, a dead session
                last = e
                await self._disconnect()
        raise McpUnavailable(
            f"cannot reach the MCP server at {self.url} ({type(last).__name__}); check that it is running, "
            "that RAGMCP_ADMIN_TOKEN is a valid admin token, and the server's rate limit") from None
