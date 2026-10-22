import os
import re
import time
from decimal import Decimal
from datetime import date, datetime

import psycopg2
from psycopg2 import pool
import sqlglot
from sqlglot import exp
from dotenv import load_dotenv
import logging
from metrics import SQL_EXECUTIONS, SQL_LATENCY, SQL_ERRORS, SQL_CACHE_HITS, SQL_CACHE_MISSES
from cachetools import TTLCache
from worker import submit_task, get_task_result

# In-memory cache for recent SELECT results (keyed by (sql_clean, max_rows))
SQL_RESULT_CACHE = TTLCache(maxsize=1024, ttl=int(os.getenv('SQL_RESULT_CACHE_TTL', '120')))


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("claims_agent")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(LOG_LEVEL)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": os.getenv("PGHOST"),
    "port": os.getenv("PGPORT"),
    "dbname": os.getenv("PGDATABASE"),
    "user": os.getenv("PGUSER"),
    "password": os.getenv("PGPASSWORD"),
}


# ============================================================
# QUERY LIMITS
# ============================================================

# Keep this reasonably low for interactive AI analytics.
# Override with PG_STATEMENT_TIMEOUT_MS in .env if required.
STATEMENT_TIMEOUT_MS = int(
    os.getenv(
        "PG_STATEMENT_TIMEOUT_MS",
        "30000",
    )
)

# EXPLAIN cost threshold (if estimated total cost exceeds this, consider query expensive)
EXPLAIN_COST_THRESHOLD = float(os.getenv("EXPLAIN_COST_THRESHOLD", "100000.0"))

DEFAULT_MAX_ROWS = int(
    os.getenv(
        "MAX_QUERY_ROWS",
        "100",
    )
)

HARD_MAX_ROWS = 1000

LOCK_TIMEOUT_MS = int(
    os.getenv(
        "PG_LOCK_TIMEOUT_MS",
        "5000",
    )
)


# ============================================================
# SCHEMA CACHE
# ============================================================

SCHEMA_CACHE_SECONDS = int(
    os.getenv(
        "SCHEMA_CACHE_SECONDS",
        "300",
    )
)

_schema_cache = None
_schema_cache_time = 0


# ============================================================
# CONNECTION POOL
# ============================================================

_connection_pool = None


def get_connection_pool():
    """
    Create and reuse the PostgreSQL connection pool.
    """

    global _connection_pool

    if _connection_pool is None:

        # Validate configuration early to provide actionable errors
        missing = [k for k, v in DB_CONFIG.items() if not v]

        if missing:
            raise ValueError(
                "PostgreSQL configuration incomplete. Missing: "
                + ", ".join(missing)
                + ". Set PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD in your environment."
            )

        try:
            logger.info("Creating new PostgreSQL connection pool to %s", DB_CONFIG.get("host"))
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                **DB_CONFIG,
            )
        except Exception as e:
            logger.exception("Failed to create PostgreSQL connection pool")
            raise RuntimeError(
                f"Failed to create PostgreSQL connection pool: {e}"
            )

    return _connection_pool


def reset_connection_pool():
    """
    Completely reset the PostgreSQL connection pool.
    """

    global _connection_pool

    try:
        if _connection_pool is not None:
            logger.info("Closing PostgreSQL connection pool")
            _connection_pool.closeall()
    except Exception:
        logger.exception("Error closing connection pool")

    _connection_pool = None


def get_connection():
    """
    Get a healthy PostgreSQL connection.

    If the pool or connection is broken, recreate the pool once.
    """

    global _connection_pool

    try:
        pool_obj = get_connection_pool()
    except Exception as e:
        logger.exception("Unable to obtain PostgreSQL connection pool")
        raise RuntimeError(
            f"Unable to obtain PostgreSQL connection pool: {e}"
        )

    conn = None

    try:

        conn = pool_obj.getconn()

        if conn is None:

            raise psycopg2.InterfaceError(
                "PostgreSQL pool returned no connection."
            )

        if conn.closed:

            raise psycopg2.InterfaceError(
                "PostgreSQL connection is closed."
            )

        logger.debug("Acquired database connection from pool")
        return conn

    except Exception:

        if conn is not None:

            try:
                conn.close()
            except Exception:
                pass

        logger.exception("Connection retrieval failed; resetting pool and retrying")
        reset_connection_pool()

        try:
            pool_obj = get_connection_pool()
        except Exception as e:
            raise RuntimeError(
                f"Unable to recreate PostgreSQL connection pool: {e}"
            )

        logger.debug("Recreated connection pool and obtained connection")
        return pool_obj.getconn()


def release_connection(conn):
    """
    Return a connection to the pool.

    Rollback before returning it so aborted transactions
    never leak into another request.
    """

    if conn is None:
        return

    try:

        if conn.closed:
            return

        try:
            conn.rollback()
        except Exception:
            pass

        pool_obj = get_connection_pool()

        pool_obj.putconn(conn)

    except Exception:

        try:
            conn.close()
        except Exception:
            pass


# ============================================================
# SCHEMA CACHE
# ============================================================

def clear_schema_cache():
    """
    Force schema metadata to be refreshed.
    """

    global _schema_cache
    global _schema_cache_time

    _schema_cache = None
    _schema_cache_time = 0


def get_schema_map(force_refresh=False):
    """
    Return:

        {
            "claims": {
                "claim_id",
                "county",
                ...
            }
        }

    Schema metadata is cached to avoid querying
    information_schema on every AI request.
    """

    global _schema_cache
    global _schema_cache_time

    now = time.time()

    if (
        not force_refresh
        and _schema_cache is not None
        and (
            now - _schema_cache_time
            < SCHEMA_CACHE_SECONDS
        )
    ):
        logger.debug("Returning schema from cache (age=%.1fs)", now - _schema_cache_time)
        return _schema_cache

    conn = get_connection()
    cur = None

    try:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY
                table_name,
                ordinal_position
            """
        )

        schema = {}

        for table_name, column_name in cur.fetchall():

            schema.setdefault(
                table_name,
                set(),
            ).add(column_name)

        conn.commit()

        _schema_cache = schema
        _schema_cache_time = now

        logger.info("Schema map refreshed (%d tables)", len(schema))

        return schema

    finally:

        if cur is not None:

            try:
                cur.close()
            except Exception:
                pass

        release_connection(conn)


# ============================================================
# TABLE DISCOVERY
# ============================================================

def list_tables():
    """
    Return all public PostgreSQL base tables.
    """

    conn = get_connection()
    cur = None

    try:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )

        tables = [row[0] for row in cur.fetchall()]

        conn.commit()
        logger.debug("Discovered %d tables", len(tables))
        return tables

    finally:

        if cur is not None:

            try:
                cur.close()
            except Exception:
                pass

        release_connection(conn)


# ============================================================
# TABLE SCHEMA
# ============================================================

def get_schema(table_name):
    """
    Return detailed schema information for one table.
    """

    if not table_name:

        raise ValueError(
            "table_name cannot be empty."
        )

    conn = get_connection()
    cur = None

    try:

        cur = conn.cursor()

        cur.execute(
            """
            SELECT
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
            ORDER BY ordinal_position
            """,
            (table_name,),
        )

        columns = cur.fetchall()

        conn.commit()

        return columns

    finally:

        if cur is not None:

            try:
                cur.close()
            except Exception:
                pass

        release_connection(conn)


# ============================================================
# SQL PARSING
# ============================================================

def parse_sql(sql):
    """
    Parse exactly one PostgreSQL SQL statement.
    """

    if not sql:

        raise ValueError(
            "SQL query cannot be empty."
        )

    sql_clean = (
        str(sql)
        .strip()
        .rstrip(";")
        .strip()
    )

    if not sql_clean:

        raise ValueError(
            "SQL query cannot be empty."
        )

    try:

        statements = sqlglot.parse(
            sql_clean,
            read="postgres",
        )

    except Exception as error:

        raise ValueError(
            f"Invalid SQL syntax: {error}"
        )

    if len(statements) != 1:

        raise ValueError(
            "Multiple SQL statements are not allowed."
        )

    return statements[0]


# ============================================================
# READ-ONLY SQL VALIDATION
# ============================================================

def validate_sql(sql):
    """
    Strict read-only SQL validation.

    Allows:

        SELECT
        CTEs
        UNION
        INTERSECT
        EXCEPT
        GROUP BY
        ORDER BY
        HAVING
        JOIN
        CASE
        FILTER
        window functions
        subqueries
        aggregates
        PostgreSQL functions

    Rejects all write operations.
    """

    tree = parse_sql(sql)

    clean = (
        str(sql)
        .strip()
        .rstrip(";")
        .strip()
    )

    # --------------------------------------------------------
    # Allowed roots
    # --------------------------------------------------------

    allowed_roots = (
        exp.Select,
        exp.Union,
        exp.Intersect,
        exp.Except,
    )

    if not isinstance(
        tree,
        allowed_roots,
    ):

        raise ValueError(
            "Only read-only SELECT queries are allowed."
        )

    # --------------------------------------------------------
    # Forbidden AST nodes
    # --------------------------------------------------------

    forbidden_nodes = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.TruncateTable,
        exp.Command,
        exp.Merge,
        exp.Grant,
        exp.Revoke,
    )

    for node in tree.walk():

        if isinstance(
            node,
            forbidden_nodes,
        ):

            raise ValueError(
                "Forbidden SQL operation: "
                + type(node).__name__
            )

    # --------------------------------------------------------
    # SELECT INTO
    # --------------------------------------------------------

    for select in tree.find_all(exp.Select):

        if select.args.get("into") is not None:

            raise ValueError(
                "SELECT INTO is not allowed."
            )

    # --------------------------------------------------------
    # Dangerous keywords
    #
    # Secondary defense only.
    # AST validation remains primary.
    # --------------------------------------------------------

    dangerous_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
        "COPY",
        "VACUUM",
        "REINDEX",
        "CALL",
        "DO",
        "MERGE",
    ]

    upper_sql = clean.upper()

    for keyword in dangerous_keywords:

        if re.search(
            rf"\b{keyword}\b",
            upper_sql,
        ):

            raise ValueError(
                "Forbidden SQL operation detected: "
                + keyword
            )

    # --------------------------------------------------------
    # Multiple statements
    # --------------------------------------------------------

    if ";" in clean:

        raise ValueError(
            "Multiple SQL statements are not allowed."
        )

    return clean


# ============================================================
# SQL IDENTIFIER VALIDATION
# ============================================================

def validate_sql_identifiers(sql):
    """
    Validate physical PostgreSQL tables and qualified
    physical columns.

    Derived namespaces such as:

        CTEs
        subqueries
        derived tables

    are allowed to be resolved by PostgreSQL itself.
    """

    tree = parse_sql(sql)

    schema = get_schema_map()

    # --------------------------------------------------------
    # CTE names
    # --------------------------------------------------------

    cte_names = set()

    for cte in tree.find_all(exp.CTE):

        name = cte.alias_or_name

        if name:
            cte_names.add(name)

    # --------------------------------------------------------
    # Physical tables
    # --------------------------------------------------------

    physical_tables = set()

    for table in tree.find_all(exp.Table):

        table_name = table.name

        if table_name in schema:

            physical_tables.add(
                table_name
            )

            continue

        if table_name in cte_names:

            continue

        raise ValueError(
            f"Unknown table '{table_name}'. "
            f"Available tables: "
            f"{', '.join(sorted(schema))}"
        )

    # --------------------------------------------------------
    # Table aliases
    # --------------------------------------------------------

    table_aliases = {}

    for table in tree.find_all(exp.Table):

        table_name = table.name
        alias = table.alias

        if alias:

            table_aliases[
                alias
            ] = table_name

    # --------------------------------------------------------
    # Physical columns
    # --------------------------------------------------------

    available_columns = set()

    for table_name in physical_tables:

        available_columns.update(
            schema[table_name]
        )

    # --------------------------------------------------------
    # SELECT aliases
    # --------------------------------------------------------

    select_aliases = set()

    for select in tree.find_all(exp.Select):

        for projection in select.expressions:

            alias = projection.alias

            if alias:

                select_aliases.add(
                    alias
                )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    for column in tree.find_all(exp.Column):

        column_name = column.name

        if not column_name:
            continue

        # COUNT(*)
        if column_name == "*":
            continue

        # SELECT aliases
        if column_name in select_aliases:
            continue

        # ----------------------------------------------------
        # Qualified column
        # ----------------------------------------------------

        if column.table:

            table_name = column.table

            resolved_table = table_aliases.get(
                table_name,
                table_name,
            )

            # CTE / derived namespace
            if resolved_table in cte_names:
                continue

            if resolved_table not in schema:

                raise ValueError(
                    f"Unknown table or alias "
                    f"'{table_name}'."
                )

            if (
                column_name
                not in schema[resolved_table]
            ):

                raise ValueError(
                    f"Unknown column "
                    f"'{table_name}.{column_name}'."
                )

            continue

        # ----------------------------------------------------
        # Unqualified column
        #
        # Unknown columns are allowed because they can
        # originate from CTEs/subqueries.
        # PostgreSQL performs final semantic validation.
        # ----------------------------------------------------

        if column_name in available_columns:
            continue

        if column_name in select_aliases:
            continue

    return True


# ============================================================
# RESULT SERIALIZATION
# ============================================================

def serialize_value(value):
    """
    Convert PostgreSQL values into JSON/Streamlit-friendly
    Python values.
    """

    if isinstance(
        value,
        Decimal,
    ):

        return float(value)

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):

        return value.isoformat()

    if isinstance(
        value,
        bytes,
    ):

        return value.decode(
            "utf-8",
            errors="replace",
        )

    return value


def serialize_rows(rows):
    """
    Serialize database rows.
    """

    return [
        tuple(
            serialize_value(value)
            for value in row
        )
        for row in rows
    ]


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_repairable_error(error_text):
    """
    Determine whether an SQL error is worth sending
    back to the LLM for repair.
    """

    error_text = str(
        error_text
    ).lower()

    # --------------------------------------------------------
    # Errors where another database attempt is dangerous
    # --------------------------------------------------------

    non_repairable = [
        "statement timeout",
        "canceling statement due to statement timeout",
        "lock timeout",
        "canceling statement due to lock timeout",
        "out of memory",
        "connection",
        "server closed",
        "terminating connection",
        "too many connections",
    ]

    for phrase in non_repairable:

        if phrase in error_text:

            return False

    # --------------------------------------------------------
    # Errors commonly repairable by SQL generation
    # --------------------------------------------------------

    repairable = [
        "syntax",
        "aggregate function calls cannot be nested",
        "column",
        "relation",
        "function",
        "operator",
        "argument",
        "round",
        "abs",
        "undefined",
        "does not exist",
        "invalid input",
        "cannot cast",
        "group",
    ]

    for phrase in repairable:

        if phrase in error_text:

            return True

    return False


# ============================================================
# QUERY EXECUTION
# ============================================================

def run_select(
    sql,
    max_rows=DEFAULT_MAX_ROWS,
):
    """
    Execute one read-only analytical query.

    IMPORTANT:

    Validation errors are returned as structured results
    rather than raised. This allows the agent's SQL repair
    mechanism to repair malformed LLM-generated SQL.
    """

    # --------------------------------------------------------
    # Validate max_rows
    # --------------------------------------------------------

    if not isinstance(
        max_rows,
        int,
    ):

        raise ValueError(
            "max_rows must be an integer."
        )

    if max_rows <= 0:

        raise ValueError(
            "max_rows must be positive."
        )

    max_rows = min(
        max_rows,
        HARD_MAX_ROWS,
    )

    # --------------------------------------------------------
    # Validate SQL
    #
    # IMPORTANT:
    # Catch validation errors so execute_sql_with_repair()
    # can see them.
    # --------------------------------------------------------

    try:

        sql_clean = validate_sql(
            sql
        )

        validate_sql_identifiers(
            sql_clean
        )

    except Exception as error:

        return {
            "error": str(error),
            "sql": str(sql),
            "execution_seconds": 0.0,
            "validation_error": True,
        }

    # --------------------------------------------------------
    # Result cache: avoid re-running identical analytical queries
    # --------------------------------------------------------

    cache_key = (sql_clean, max_rows)
    cacheable = not re.search(
        r"\b(?:NOW\s*\(\s*\)|CURRENT_TIMESTAMP\b|CURRENT_DATE\b)",
        sql_clean,
        flags=re.IGNORECASE,
    )

    try:
        cached = SQL_RESULT_CACHE.get(cache_key) if cacheable else None
    except Exception:
        cached = None
    if cached is not None:
        try:
            SQL_CACHE_HITS.inc()
        except Exception:
            pass
        logger.debug("Serving result from SQL cache")
        return cached
    else:
        try:
            SQL_CACHE_MISSES.inc()
        except Exception:
            pass

    # --------------------------------------------------------
    # Database connection
    # --------------------------------------------------------

    conn = get_connection()

    cur = None

    started = time.perf_counter()

    logger.info("Running SELECT (max_rows=%d)", max_rows)
    SQL_EXECUTIONS.inc()
    try:

        cur = conn.cursor()

        # ----------------------------------------------------
        # Statement timeout
        # ----------------------------------------------------

        cur.execute(
            "SET LOCAL statement_timeout = %s",
            (
                STATEMENT_TIMEOUT_MS,
            ),
        )

        # ----------------------------------------------------
        # Lock timeout
        # ----------------------------------------------------

        cur.execute(
            "SET LOCAL lock_timeout = %s",
            (
                LOCK_TIMEOUT_MS,
            ),
        )

        # ----------------------------------------------------
        # Result limit
        #
        # The original analytical query remains intact.
        # The outer LIMIT controls returned rows.
        # ----------------------------------------------------

        limited_sql = f"""
            SELECT *
            FROM (
                {sql_clean}
            ) AS agent_query
            LIMIT {max_rows}
        """

        logger.debug("Executing SQL: %s", limited_sql)
        start = time.perf_counter()
        cur.execute(limited_sql)

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        columns = [
            description[0]
            for description in cur.description
        ]

        # ----------------------------------------------------
        # Rows
        # ----------------------------------------------------

        rows = cur.fetchmany(
            max_rows
        )

        elapsed = time.perf_counter() - started
        try:
            SQL_LATENCY.observe(elapsed)
        except Exception:
            pass
        result = {
            "columns": columns,
            "rows": serialize_rows(rows),
            "row_count": len(rows),
            "execution_seconds": round(elapsed, 4),
        }

        if cacheable:
            try:
                SQL_RESULT_CACHE[cache_key] = result
            except Exception:
                logger.debug("Failed to set SQL result cache")

        conn.commit()

        logger.info("Query returned %d rows (elapsed=%.3fs)", len(rows), elapsed)
        return result

    except psycopg2.Error as error:

        try:
            conn.rollback()
        except Exception:
            pass

        elapsed = time.perf_counter() - started

        error_lines = str(error).splitlines()

        error_message = error_lines[0].strip() if error_lines else "Unknown PostgreSQL error."

        logger.warning("Database error: %s (elapsed=%.3fs)", error_message, elapsed)
        try:
            SQL_ERRORS.inc()
        except Exception:
            pass

        return {
            "error": error_message,
            "sql": sql_clean,
            "execution_seconds": round(elapsed, 4),
            "database_error": True,
            "repairable": is_repairable_error(error_message),
        }

    except Exception as error:

        try:
            conn.rollback()
        except Exception:
            pass

        elapsed = time.perf_counter() - started

        logger.exception("Unexpected error during query execution")

        return {
            "error": str(error),
            "sql": sql_clean,
            "execution_seconds": round(elapsed, 4),
            "repairable": False,
        }

    finally:

        if cur is not None:

            try:
                cur.close()
            except Exception:
                pass

        release_connection(
            conn
        )


def run_select_background(sql, max_rows=HARD_MAX_ROWS):
    """Submit a run_select job to the worker pool and return a job id."""
    return submit_task(run_select, sql, max_rows)


# ============================================================
# DATABASE HEALTH
# ============================================================

def database_health():
    """
    Lightweight PostgreSQL connectivity test.
    """

    started = time.perf_counter()

    conn = None
    cur = None

    try:

        conn = get_connection()

        cur = conn.cursor()

        cur.execute(
            "SELECT 1"
        )

        result = cur.fetchone()

        elapsed = (
            time.perf_counter()
            - started
        )

        conn.commit()

        return {
            "connected": result == (1,),
            "response_seconds": round(
                elapsed,
                4,
            ),
        }

    except Exception as error:

        if conn is not None:

            try:
                conn.rollback()
            except Exception:
                pass

        return {
            "connected": False,
            "error": str(error),
        }

    finally:

        if cur is not None:

            try:
                cur.close()
            except Exception:
                pass

        if conn is not None:

            release_connection(
                conn
            )


# ============================================================
# QUERY BENCHMARK
# ============================================================

def benchmark_query(sql):
    """
    Execute a query and measure total Python-side latency.
    """

    started = time.perf_counter()

    result = run_select(
        sql
    )

    result["total_seconds"] = round(
        time.perf_counter()
        - started,
        4,
    )

    return result


def explain_query(sql):
    """
    Run EXPLAIN (FORMAT JSON) for the given SQL and return a small summary:

    {"total_cost": float, "plan": <json>, "estimated_rows": float}

    This does not execute the query itself.
    """

    try:
        sql_clean = validate_sql(sql)
    except Exception as e:
        return {"error": str(e)}

    conn = get_connection()
    cur = None

    try:
        cur = conn.cursor()

        explain_sql = f"EXPLAIN (FORMAT JSON) {sql_clean}"
        cur.execute(explain_sql)

        row = cur.fetchone()

        # psycopg2 returns a one-element tuple containing the JSON plan
        plan_json = row[0]

        # plan_json is typically a list with one plan dict
        if isinstance(plan_json, (list, tuple)) and len(plan_json) > 0:
            plan = plan_json[0]
        else:
            plan = plan_json

        # Walk plan to find total cost and estimated rows
        total_cost = None
        est_rows = None

        def walk(node):
            nonlocal total_cost, est_rows
            if not isinstance(node, dict):
                return
            if "Plan" in node:
                p = node["Plan"]
            else:
                p = node
            if total_cost is None and "Total Cost" in p:
                try:
                    total_cost = float(p.get("Total Cost") or p.get("Total Cost", 0))
                except Exception:
                    pass
            if est_rows is None and "Plan Rows" in p:
                try:
                    est_rows = float(p.get("Plan Rows") or p.get("Plan Rows", 0))
                except Exception:
                    pass
            # recurse
            for k, v in (p.items() if isinstance(p, dict) else []):
                if isinstance(v, dict):
                    walk(v)
                if isinstance(v, list):
                    for item in v:
                        walk(item)

        walk(plan)

        return {
            "total_cost": total_cost or 0.0,
            "estimated_rows": est_rows or 0.0,
            "plan": plan,
        }

    except Exception as e:
        logger.exception("EXPLAIN failed: %s", e)
        return {"error": str(e)}

    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        release_connection(conn)


# ============================================================
# BASIC ANALYTICS TEST
# ============================================================

def test_basic_query():
    """
    Test county claim volume.
    """

    sql = """
        SELECT
            county,
            COUNT(*) AS claim_count

        FROM claims

        WHERE county IS NOT NULL

        GROUP BY county

        ORDER BY claim_count DESC

        LIMIT 10
    """

    return run_select(
        sql,
        max_rows=10,
    )


# ============================================================
# CTE TEST
# ============================================================

def test_cte_query():
    """
    Test CTE support.
    """

    sql = """
        WITH county_stats AS (

            SELECT
                county,
                COUNT(*) AS claim_count

            FROM claims

            WHERE county IS NOT NULL

            GROUP BY county
        )

        SELECT
            county,
            claim_count

        FROM county_stats

        ORDER BY claim_count DESC

        LIMIT 10
    """

    return run_select(
        sql,
        max_rows=10,
    )


# ============================================================
# WINDOW FUNCTION TEST
# ============================================================

def test_window_query():
    """
    Test PostgreSQL window functions.
    """

    sql = """
        SELECT
            county,
            claim_type,
            COUNT(*) AS claim_count,

            RANK() OVER (
                PARTITION BY county
                ORDER BY COUNT(*) DESC
            ) AS claim_type_rank

        FROM claims

        WHERE county IS NOT NULL
          AND claim_type IS NOT NULL

        GROUP BY
            county,
            claim_type

        ORDER BY
            county,
            claim_type_rank

        LIMIT 50
    """

    return run_select(
        sql,
        max_rows=50,
    )


# ============================================================
# ROBUST ANOMALY TEST
# ============================================================

def test_anomaly_query():
    """
    Test multi-stage robust anomaly analysis.

    IMPORTANT:

    Statistical stages are deliberately separated.

        county_stats
            ↓
        eligible
            ↓
        medians
            ↓
        deviations
            ↓
        MAD
            ↓
        scores

    This prevents nested aggregate errors.
    """

    sql = """
        WITH county_stats AS (

            SELECT

                county,

                COUNT(*) AS claim_count,

                AVG(claim_amount)
                    AS average_claim_amount,

                SUM(claim_amount)
                    AS total_claim_amount,

                COUNT(*) FILTER (
                    WHERE is_cancelled = TRUE
                ) AS cancelled_claims,

                100.0 *
                COUNT(*) FILTER (
                    WHERE is_cancelled = TRUE
                )
                /
                NULLIF(
                    COUNT(*),
                    0
                ) AS cancellation_rate

            FROM claims

            WHERE county IS NOT NULL

            GROUP BY county
        ),

        eligible AS (

            SELECT *

            FROM county_stats

            WHERE claim_count >= 100
        ),

        medians AS (

            SELECT

                PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY average_claim_amount
                ) AS median_avg_amount,

                PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY cancellation_rate
                ) AS median_cancellation_rate

            FROM eligible
        ),

        deviations AS (

            SELECT

                e.*,

                ABS(
                    e.average_claim_amount
                    - m.median_avg_amount
                ) AS amount_deviation,

                ABS(
                    e.cancellation_rate
                    - m.median_cancellation_rate
                ) AS cancellation_deviation

            FROM eligible e

            CROSS JOIN medians m
        ),

        mad AS (

            SELECT

                PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY amount_deviation
                ) AS mad_amount,

                PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY cancellation_deviation
                ) AS mad_cancellation

            FROM deviations
        ),

        scored AS (

            SELECT

                d.county,

                d.claim_count,

                d.average_claim_amount,

                d.total_claim_amount,

                d.cancelled_claims,

                d.cancellation_rate,

                med.median_avg_amount,

                med.median_cancellation_rate,

                CASE

                    WHEN
                        mad.mad_amount > 0
                        AND d.average_claim_amount
                            IS NOT NULL

                    THEN

                        ABS(
                            d.average_claim_amount
                            - med.median_avg_amount
                        )
                        /
                        (
                            1.4826
                            * mad.mad_amount
                        )

                    ELSE 0

                END AS amount_anomaly,

                CASE

                    WHEN
                        mad.mad_cancellation > 0
                        AND d.cancellation_rate
                            IS NOT NULL

                    THEN

                        ABS(
                            d.cancellation_rate
                            - med.median_cancellation_rate
                        )
                        /
                        (
                            1.4826
                            * mad.mad_cancellation
                        )

                    ELSE 0

                END AS cancellation_anomaly,

                CASE

                    WHEN
                        mad.mad_amount > 0
                        AND d.average_claim_amount
                            IS NOT NULL

                    THEN

                        (
                            d.average_claim_amount
                            - med.median_avg_amount
                        )
                        /
                        (
                            1.4826
                            * mad.mad_amount
                        )

                    ELSE 0

                END AS amount_zscore,

                CASE

                    WHEN
                        mad.mad_cancellation > 0
                        AND d.cancellation_rate
                            IS NOT NULL

                    THEN

                        (
                            d.cancellation_rate
                            - med.median_cancellation_rate
                        )
                        /
                        (
                            1.4826
                            * mad.mad_cancellation
                        )

                    ELSE 0

                END AS cancellation_zscore

            FROM deviations d

            CROSS JOIN medians med

            CROSS JOIN mad
        )

        SELECT

            county,

            claim_count,

            ROUND(
                CAST(average_claim_amount AS numeric),
                2
            ) AS average_claim_amount,

            ROUND(
                CAST(total_claim_amount AS numeric),
                2
            ) AS total_claim_amount,

            cancelled_claims,

            ROUND(
                CAST(cancellation_rate AS numeric),
                2
            ) AS cancellation_rate,

            ROUND(
                CAST(
                    (
                        amount_anomaly * 0.60
                        +
                        cancellation_anomaly * 0.40
                    ) AS numeric
                ),
                2
            ) AS anomaly_score,

            ROUND(
                CAST(amount_zscore AS numeric),
                2
            ) AS amount_anomaly_zscore,

            ROUND(
                CAST(cancellation_zscore AS numeric),
                2
            ) AS cancellation_anomaly_zscore

        FROM scored

        ORDER BY anomaly_score DESC

        LIMIT 10
    """

    return run_select(
        sql,
        max_rows=10,
    )


# ============================================================
# COMMAND-LINE TEST SUITE
# ============================================================

def run_test_suite():
    """
    Execute the complete database-layer test suite.
    """

    print()
    print("=" * 70)
    print("CLAIMS INTELLIGENCE DATABASE TEST SUITE")
    print("=" * 70)

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    print()
    print("[1] DATABASE HEALTH")
    print("-" * 70)

    health = database_health()

    print(health)

    if not health.get("connected"):

        print()
        print(
            "Database connection failed."
        )

        return

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    print()
    print("[2] TABLE DISCOVERY")
    print("-" * 70)

    try:

        tables = list_tables()

        for table_name in tables:

            print(
                f" - {table_name}"
            )

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    # --------------------------------------------------------
    # Claims schema
    # --------------------------------------------------------

    print()
    print("[3] CLAIMS SCHEMA")
    print("-" * 70)

    try:

        columns = get_schema(
            "claims"
        )

        for column in columns:

            print(
                column
            )

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    # --------------------------------------------------------
    # Basic query
    # --------------------------------------------------------

    print()
    print("[4] BASIC QUERY")
    print("-" * 70)

    try:

        result = test_basic_query()

        print(result)

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    # --------------------------------------------------------
    # CTE
    # --------------------------------------------------------

    print()
    print("[5] CTE QUERY")
    print("-" * 70)

    try:

        result = test_cte_query()

        print(result)

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    print()
    print("[6] WINDOW FUNCTION")
    print("-" * 70)

    try:

        result = test_window_query()

        print(result)

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    # --------------------------------------------------------
    # Anomaly
    # --------------------------------------------------------

    print()
    print("[7] ROBUST ANOMALY ANALYSIS")
    print("-" * 70)

    try:

        result = test_anomaly_query()

        print(result)

    except Exception as error:

        print(
            "ERROR:",
            error,
        )

    print()
    print("=" * 70)
    print("DATABASE TEST SUITE COMPLETE")
    print("=" * 70)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_test_suite()