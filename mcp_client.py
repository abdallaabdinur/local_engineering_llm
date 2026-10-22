"""
Synchronous adapter for calling the Claims MCP server from the existing
synchronous agent.py code.

The MCP server process and session are kept alive in a background thread so
the agent does not start a new server process for every database tool call.
"""
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path

# StdioServerParameters and stdio_client are confirmed present in
# mcp.client.stdio on the installed SDK.
from mcp.client.stdio import StdioServerParameters, stdio_client

# ClientSession is exported at the top level in 1.x. Fall back to its module
# path in case the top-level export differs in your installed 2.x version.
try:
    from mcp import ClientSession
except ImportError:
    from mcp.client.session import ClientSession

logger = logging.getLogger("claims_agent.mcp_client")

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_FILE = PROJECT_DIR / "mcp_server.py"

_loop = None
_thread = None
_ready = threading.Event()
_start_error = None
_session = None
_exit_stack = None


async def _serve_forever():
    global _session, _exit_stack, _start_error

    try:
        _exit_stack = AsyncExitStack()
        await _exit_stack.__aenter__()

        server_params = StdioServerParameters(
            command=os.environ.get("PYTHON_EXECUTABLE", sys.executable),
            args=[str(SERVER_FILE)],
            env=os.environ.copy(),
        )
        read_stream, write_stream = await _exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        _session = await _exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await _session.initialize()
        _ready.set()

        # Keep the client session and server subprocess alive.
        await asyncio.Event().wait()

    except Exception as exc:
        _start_error = exc
        _ready.set()
        logger.exception("MCP client/server connection failed")
    finally:
        # Drop the stale session so the next call restarts cleanly.
        _session = None
        if _exit_stack is not None:
            await _exit_stack.aclose()
            _exit_stack = None


def _start_client():
    global _loop, _thread, _start_error

    if _thread and _thread.is_alive() and _session is not None:
        return

    _ready.clear()
    _start_error = None
    _loop = asyncio.new_event_loop()

    def runner():
        asyncio.set_event_loop(_loop)
        _loop.run_until_complete(_serve_forever())

    _thread = threading.Thread(
        target=runner,
        name="claims-mcp-client",
        daemon=True,
    )
    _thread.start()

    if not _ready.wait(timeout=20):
        raise TimeoutError("Timed out while starting the Claims MCP server.")
    if _start_error is not None:
        raise RuntimeError(
            f"Could not start MCP server: {_start_error}"
        ) from _start_error


def _call_tool(name, arguments, timeout=60):
    _start_client()
    if _session is None or _loop is None:
        raise RuntimeError("MCP session is not available.")

    future = asyncio.run_coroutine_threadsafe(
        _session.call_tool(name, arguments),
        _loop,
    )
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        # Stop the call from continuing to run in the background.
        future.cancel()
        raise TimeoutError(
            f"MCP tool '{name}' timed out after {timeout} seconds."
        )

    if getattr(result, "isError", False):
        message = "; ".join(
            getattr(item, "text", "")
            for item in (getattr(result, "content", None) or [])
        )
        raise RuntimeError(message or f"MCP tool '{name}' failed.")

    # The server returns JSON strings from tools, normally in a text content item.
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        return structured
    return {}


def mcp_list_tables():
    return _call_tool("list_tables", {})


def mcp_get_schema(table_name):
    return _call_tool("get_schema", {"table_name": table_name})


def mcp_validate_sql(sql):
    return _call_tool("validate_sql", {"sql": sql})


def mcp_run_select(sql, max_rows=None):
    arguments = {"sql": sql}
    if max_rows is not None:
        arguments["max_rows"] = max_rows
    return _call_tool("execute_sql", arguments, timeout=90)


def mcp_explain_query(sql):
    return _call_tool("explain_sql", {"sql": sql}, timeout=60)