"""Spawn the server over stdio and check the tool list and `stats` respond.

Needs neither database nor model: those are built lazily on first search.
"""

import anyio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_lists_tools_and_calls_stats():
    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "ragmcp.server"])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                res = await s.call_tool("stats", {})
                return names, res

    names, res = anyio.run(run)
    assert {"stats", "search", "get_chunk"} <= names
    assert not res.is_error
