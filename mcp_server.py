"""
MCP server for the Claims Intelligence project.

Exposes the existing read-only PostgreSQL tools in tools.py through MCP.
Run this file directly using the project's active virtual environment.

Targets the MCP Python SDK 2.x (MCPServer). The 1.x FastMCP class in
mcp.server.fastmcp no longer exists in 2.x.
"""
import json
import logging
from decimal import Decimal
from datetime import date, datetime

from mcp.server.mcpserver import MCPServer

from tools import (
    list_tables as db_list_tables,
    get_schema as db_get_schema,
    validate_sql as db_validate_sql,
    validate_sql_identifiers as db_validate_sql_identifiers,
    run_select as db_run_select,
    explain_query as db_explain_query,
)

# With the stdio transport, stdout carries the protocol messages.
# Logging goes to stderr by default, which is what we want.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claims_agent.mcp_server")

MAX_ROWS_LIMIT = 1000

mcp = MCPServer("claims-database-tools")


def _json_default(value):
    """Serialise Decimal, date and datetime (and anything else) as strings."""
    if isinstance(value, (Decimal, date, datetime)):
        return str(value)
    return str(value)


@mcp.tool()
def list_tables() -> str:
    """List public base tables available in the claims PostgreSQL database."""
    return json.dumps(db_list_tables(), default=_json_default)


@mcp.tool()
def get_schema(table_name: str) -> str:
    """Return the columns, PostgreSQL data types, and nullability for a public table."""
    if not table_name or not table_name.replace("_", "").isalnum():
        raise ValueError("Provide a valid table name.")
    columns = db_get_schema(table_name)
    return json.dumps(
        [
            {
                "column": row[0],
                "data_type": row[1],
                "nullable": row[2] == "YES",
            }
            for row in columns
        ],
        default=_json_default,
    )


@mcp.tool()
def validate_sql(sql: str) -> str:
    """Validate that SQL is a single read-only SELECT query using known public tables and columns."""
    clean_sql = db_validate_sql(sql)
    db_validate_sql_identifiers(clean_sql)
    return json.dumps({"valid": True, "sql": clean_sql})


@mcp.tool()
def execute_sql(sql: str, max_rows: int | None = None) -> str:
    """Execute one validated read-only SELECT query and return JSON results. Writes are prohibited."""
    # Validate at the MCP boundary too. run_select performs its own
    # validation as defence in depth.
    clean_sql = db_validate_sql(sql)
    db_validate_sql_identifiers(clean_sql)

    if max_rows is None:
        # Use the default row limit defined in tools.run_select.
        result = db_run_select(clean_sql)
    else:
        # Keep an explicit row limit within a sane range.
        max_rows = min(max(int(max_rows), 1), MAX_ROWS_LIMIT)
        result = db_run_select(clean_sql, max_rows=max_rows)
    return json.dumps(result, default=_json_default)


@mcp.tool()
def explain_sql(sql: str) -> str:
    """Return the PostgreSQL query plan cost estimate for one validated read-only SELECT query."""
    clean_sql = db_validate_sql(sql)
    db_validate_sql_identifiers(clean_sql)
    return json.dumps(db_explain_query(clean_sql), default=_json_default)


if __name__ == "__main__":
    # MCP uses stdio by default when launched as a subprocess by an MCP client.
    mcp.run(transport="stdio")