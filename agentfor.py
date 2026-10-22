from openai import OpenAI
import json
import logging
import os
import random
import threading
import time

import psycopg2
from dotenv import load_dotenv

from tools import (
    list_tables,
    get_schema as get_table_schema,
    run_select,
)


load_dotenv()
logger = logging.getLogger("claims_agent.generator")




INSERT_CLAIM_SQL = """
INSERT INTO public.claims (
    claim_id, creation, modified, bundle_id, claim_status,
    claim_type, claim_subtype, claim_amount, date_start, date_end,
    client, facility, facility_name, facility_level, facility_category,
    county, sub_county, scheme_id, insurer, claim_use, is_cancelled,
    diagnoses_code, interventions_code, sub_interventions_code,
    linked_claim, claim_upstream_error_group, claim_upstream_errors,
    claim_verdict, dbt_loaded_at
)
VALUES (
    %(claim_id)s, %(creation)s, %(modified)s, %(bundle_id)s, %(claim_status)s,
    %(claim_type)s, %(claim_subtype)s, %(claim_amount)s, %(date_start)s, %(date_end)s,
    %(client)s, %(facility)s, %(facility_name)s, %(facility_level)s, %(facility_category)s,
    %(county)s, %(sub_county)s, %(scheme_id)s, %(insurer)s, %(claim_use)s, %(is_cancelled)s,
    %(diagnoses_code)s, %(interventions_code)s, %(sub_interventions_code)s,
    %(linked_claim)s, %(claim_upstream_error_group)s, %(claim_upstream_errors)s,
    %(claim_verdict)s, %(dbt_loaded_at)s
)
"""


def _writer_connection():
    """Open only the dedicated writer connection for ingestion."""

    user = os.getenv("PG_WRITER_USER")
    password = os.getenv("PG_WRITER_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "Set PG_WRITER_USER and PG_WRITER_PASSWORD for agentfor ingestion."
        )

    return psycopg2.connect(
        host=os.getenv("PG_WRITER_HOST", os.getenv("PGHOST")),
        port=os.getenv("PG_WRITER_PORT", os.getenv("PGPORT")),
        dbname=os.getenv("PG_WRITER_DATABASE", os.getenv("PGDATABASE")),
        user=user,
        password=password,
        connect_timeout=10,
    )


def populate_claim(claim: dict) -> str:
    """Insert one payload produced by live_claim_generator."""

    connection = _writer_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(INSERT_CLAIM_SQL, claim)
        connection.commit()
        return str(claim["claim_id"])
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def populate_claims(claims: list[dict]) -> list[str]:
    """Insert a batch of producer payloads through one transaction."""

    if not claims:
        return []

    connection = _writer_connection()
    try:
        with connection.cursor() as cursor:
            ids = []
            for claim in claims:
                cursor.execute(INSERT_CLAIM_SQL, claim)
                ids.append(str(claim["claim_id"]))
        connection.commit()
        return ids
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def run_live_ingestion(
    interval_seconds: float = 10.0,
    batch_min: int = 1,
    batch_max: int = 5,
    stop_event: threading.Event | None = None,
) -> None:
    """Connect the data-only producer to the database writer."""

    from live_claim_generator import generate_claims

    print("Agentfor database ingestion started.")
    try:
        while stop_event is None or not stop_event.is_set():
            try:
                claims = generate_claims(random.randint(batch_min, batch_max))
                ids = populate_claims(claims)
                print(f"Inserted {len(ids)} generated claims into PostgreSQL")
            except Exception:
                logger.exception("Generated claim batch could not be inserted")

            wait_seconds = max(0.1, interval_seconds)
            if stop_event is None:
                time.sleep(wait_seconds)
            elif stop_event.wait(wait_seconds):
                break
    except KeyboardInterrupt:
        print("Agentfor database ingestion stopped.")


def run_select_query(sql):
    """Read-only adapter for the agent tool contract."""

    result = run_select(sql)

    if isinstance(result, dict) and result.get("error"):
        return {
            "success": False,
            "error": result["error"],
            "sql": sql,
        }

    return {
        "success": True,
        "result": result,
    }


# ============================================================
# QWEN / OLLAMA CONNECTION
# ============================================================

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODEL = "qwen3.5:4b"


# ============================================================
# TOOL DEFINITIONS
# ============================================================

tools = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": (
                "List all available tables in the PostgreSQL database. "
                "Use this when you need to discover what tables are available."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": (
                "Get the columns and data types of a specific PostgreSQL table. "
                "Use this before writing SQL when you need to know the table structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "The name of the PostgreSQL table."
                    }
                },
                "required": ["table"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "run_select_query",
            "description": (
                "Execute a read-only SELECT SQL query against PostgreSQL "
                "and return the results. Only use SELECT queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A PostgreSQL SELECT query."
                    }
                },
                "required": ["sql"]
            }
        }
    }
]


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(tool_name, arguments):

    if tool_name == "list_tables":
        return list_tables()

    elif tool_name == "get_table_schema":
        return get_table_schema(arguments["table"])

    elif tool_name == "run_select_query":
        return run_select_query(arguments["sql"])

    else:
        return {
            "error": f"Unknown tool: {tool_name}"
        }


# ============================================================
# SCHEMA INJECTION
# ============================================================

def build_schema_context():
    """
    Retrieve the database schema and format it as text
    that can be injected into Qwen's system prompt.
    """

    try:

        tables = list_tables()

        schema_text = ""

        for table in tables:

            schema = get_table_schema(table)

            schema_text += f"TABLE: {table}\n"

            for column in schema["columns"]:

                schema_text += (
                    f"  - {column['column']} "
                    f"({column['data_type']}, "
                    f"nullable={column['nullable']})\n"
                )

            schema_text += "\n"

        return schema_text

    except Exception as e:

        print(f"[Schema loading error: {e}]")

        return "Database schema could not be loaded."


# ============================================================
# LOAD SCHEMA ONCE WHEN PROGRAM STARTS
# ============================================================

DATABASE_SCHEMA = "Ingestion service; schema reads are owned by the read-only agent."


# ============================================================
# AGENT LOOP
# ============================================================

def ask_agent(question):
    raise RuntimeError(
        "agentfor.py is the database ingestion service. "
        "Use agent.py for read-only database questions."
    )

    # --------------------------------------------------------
    # Maximum SQL attempts
    # --------------------------------------------------------

    MAX_SQL_ATTEMPTS = 3

    sql_attempts = 0

    # --------------------------------------------------------
    # System prompt with schema injection
    # --------------------------------------------------------

    system_prompt = f"""
You are a PostgreSQL data analysis agent.

Your job is to answer the user's questions using the
available database tools.

============================================================
DATABASE SCHEMA
============================================================

{DATABASE_SCHEMA}

============================================================
RULES
============================================================

1. Use the database tools when the user asks about
   database data.

2. Only use tables and columns that appear in the
   database schema above.

3. Do not invent table names or column names.

4. Use list_tables when you need to discover available
   tables.

5. Use get_table_schema when you need to verify the
   structure of a specific table.

6. Use run_select_query to retrieve actual database
   results.

7. Only generate SELECT queries.

8. Never generate INSERT, UPDATE, DELETE, DROP, ALTER,
   TRUNCATE, CREATE, GRANT, or REVOKE statements.

9. Do not modify the database.

10. When a question requires multiple tables, use
    appropriate SQL JOINs based on the available schema.

11. Do not assume relationships that are not supported
    by the database schema.

12. If a SQL query fails, examine the database error
    and correct the SQL query before trying again.

13. You have a maximum of 3 SQL execution attempts.

14. After receiving successful query results, explain
    the answer clearly.

15. Do not expose unnecessary technical details to
    the user.

16. If no records are returned, clearly tell the user
    that no matching records were found.
17. For amounts if currency is not specified,use Kes currency symbol. 

============================================================
SQL SAFETY
============================================================

All SQL is executed through run_select_query(), which
has additional safety controls.

Do not attempt to bypass those controls.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    # --------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------

    for _ in range(10):

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # ----------------------------------------------------
        # If Qwen has a normal answer, return it
        # ----------------------------------------------------

        if not message.tool_calls:

            return message.content

        # ----------------------------------------------------
        # Add Qwen's message to conversation
        # ----------------------------------------------------

        messages.append(message)

        # ----------------------------------------------------
        # Execute every requested tool
        # ----------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:

                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                arguments = {}

            print(f"\n[Tool called: {tool_name}]")

            print(f"[Arguments: {arguments}]")

            # ------------------------------------------------
            # Execute tool
            # ------------------------------------------------

            result = execute_tool(
                tool_name,
                arguments
            )

            print(f"[Tool result: {result}]")

            # =================================================
            # SQL RETRY HANDLING
            # =================================================

            if tool_name == "run_select_query":

                sql_attempts += 1

                print(
                    f"[SQL attempt "
                    f"{sql_attempts}/{MAX_SQL_ATTEMPTS}]"
                )

                # ------------------------------------------------
                # Check whether SQL failed
                # ------------------------------------------------

                if (
                    isinstance(result, dict)
                    and result.get("success") is False
                ):

                    error_message = result.get(
                        "error",
                        "Unknown SQL/database error."
                    )

                    print(
                        f"[SQL ERROR] {error_message}"
                    )

                    # --------------------------------------------
                    # Hard maximum reached
                    # --------------------------------------------

                    if sql_attempts >= MAX_SQL_ATTEMPTS:

                        print(
                            "[SQL RETRY LIMIT REACHED]"
                        )

                        return (
                            "I could not complete the database "
                            "query after 3 SQL attempts."
                        )

                    # --------------------------------------------
                    # Send error back to Qwen
                    # --------------------------------------------

                    retry_result = {
                        "success": False,
                        "error": error_message,
                        "sql_attempt": sql_attempts,
                        "remaining_attempts": (
                            MAX_SQL_ATTEMPTS - sql_attempts
                        ),
                        "instruction": (
                            "The SQL query failed. Review the "
                            "database error and correct the SQL. "
                            "Check the injected schema and use "
                            "get_table_schema if necessary. "
                            "Then try the corrected SELECT query."
                        )
                    }

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                retry_result,
                                default=str
                            )
                        }
                    )

                    print(
                        f"[SQL failed. Remaining attempts: "
                        f"{MAX_SQL_ATTEMPTS - sql_attempts}]"
                    )

                    continue

            # =================================================
            # NORMAL TOOL RESULT
            # =================================================

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        result,
                        default=str
                    )
                }
            )

    return "The agent reached the maximum number of tool calls."


# ============================================================
# TERMINAL CHAT
# ============================================================

if __name__ == "__main__":
    run_live_ingestion()