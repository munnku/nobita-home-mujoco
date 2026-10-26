"""TD から作った道具を MCP (Model Context Protocol) のサーバーとして出す (Claude Code / Claude Desktop から使う)。

Claude の API キーなしで、Claude Code (Pro などのプラン) を LLM として家電とアヒルを動かすため。
道具は app/llm_agent.py と同じもの (home/wot_tools.py)。WoT サーバー (app/wot_server.py) が動いている必要がある。
.mcp.json (このフォルダ) に登録してあるので、このフォルダで Claude Code を開くと道具が見える。
"""
import asyncio
import json
import os
import sys

import mcp_types as types
from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from home.wot_tools import SYSTEM, THINGS, WoTTools  # noqa: E402

_tools = None


def tools():
    """TD は起動時ではなく初めて使うときに読む (WoT サーバーが後から起動しても使えるように)。"""
    global _tools
    if _tools is None:
        _tools = WoTTools(os.environ.get("WOT_THINGS", THINGS))
    return _tools


async def list_tools(ctx, params):
    try:
        ts = await asyncio.to_thread(lambda: tools().tools)
    except Exception as e:  # noqa: BLE001
        return types.ListToolsResult(tools=[types.Tool(
            name="wot_unavailable", description=f"WoT サーバーに届かない: {e}",
            inputSchema={"type": "object", "properties": {}})])
    return types.ListToolsResult(tools=[types.Tool(name=t["name"], description=t["description"],
                                                   inputSchema=t["input_schema"]) for t in ts])


async def call_tool(ctx, params):
    try:
        out = await asyncio.to_thread(tools().call, params.name, params.arguments or {})
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(out, ensure_ascii=False))])
    except Exception as e:  # noqa: BLE001
        return types.CallToolResult(content=[types.TextContent(type="text", text=f"error: {e}")], isError=True)


server = Server("orusuban-home", instructions=SYSTEM, on_list_tools=list_tools, on_call_tool=call_tool)


async def main():
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
