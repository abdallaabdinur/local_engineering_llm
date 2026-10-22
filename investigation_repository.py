"""
Investigation Repository

Persistent SQLite storage for investigation cases
and structured audit events.

This module is responsible for persistence only.
Business rules remain in investigation_workflow.py.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


DEFAULT_DB_PATH = "investigation.db"


# ============================================================
# TIME
# ============================================================

def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# CONNECTION
# ============================================================

def _connect(
    db_path: str = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    """
    Open SQLite connection with:

    - Row access by column name
    - Foreign-key enforcement
    """

    connection = sqlite3.connect(db_path)

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database(
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Create investigation database tables and indexes.
    """

    with _connect(db_path) as connection:

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS investigation_cases (
                case_id TEXT PRIMARY KEY,
                claim_id TEXT NOT NULL UNIQUE,
                county TEXT NOT NULL,

                facility TEXT,
                facility_name TEXT,
                insurer TEXT,
                claim_type TEXT,
                creation TEXT,

                claim_amount REAL NOT NULL DEFAULT 0,

                priority_level TEXT NOT NULL,
                priority_score REAL NOT NULL DEFAULT 0,
                priority_action TEXT,

                anomaly_score INTEGER NOT NULL DEFAULT 0,
                context_score INTEGER NOT NULL DEFAULT 0,
                benchmark_score INTEGER NOT NULL DEFAULT 0,

                evidence_json TEXT NOT NULL DEFAULT '[]',
                context_flags_json TEXT NOT NULL DEFAULT '[]',
                statistical_signals_json TEXT NOT NULL DEFAULT '{}',

                investigation_status TEXT NOT NULL DEFAULT 'OPEN',

                assigned_investigator TEXT,

                investigator_notes_json TEXT NOT NULL DEFAULT '[]',
                investigator_findings_json TEXT NOT NULL DEFAULT '[]',

                fraud_determination TEXT,
                determination_reason TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                closed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS investigation_audit (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,

                case_id TEXT NOT NULL,

                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,

                investigator TEXT,
                status TEXT,
                previous_status TEXT,

                determination TEXT,

                previous_investigator TEXT,

                note TEXT,

                FOREIGN KEY (case_id)
                    REFERENCES investigation_cases(case_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_cases_county
                ON investigation_cases(county);

            CREATE INDEX IF NOT EXISTS idx_cases_status
                ON investigation_cases(investigation_status);

            CREATE INDEX IF NOT EXISTS idx_cases_priority
                ON investigation_cases(priority_level);

            CREATE INDEX IF NOT EXISTS idx_cases_investigator
                ON investigation_cases(assigned_investigator);

            CREATE INDEX IF NOT EXISTS idx_audit_case
                ON investigation_audit(case_id);

            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON investigation_audit(timestamp);
            """
        )


# ============================================================
# JSON HELPERS
# ============================================================

def _json_dump(value: Any) -> str:
    """
    Safely serialize Python values into JSON.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def _json_load(
    value: Optional[str],
    default: Any,
) -> Any:
    """
    Safely deserialize JSON.

    Returns default if the database value is empty
    or malformed.
    """

    if value is None:
        return default

    try:
        return json.loads(value)
    except (
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return default


# ============================================================
# SAVE / UPDATE CASE
# ============================================================

def save_case(
    case: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """
    Insert or update an investigation case.

    IMPORTANT:
    This function persists the case itself.
    Audit events are persisted separately using
    add_audit_event().
    """

    if not case:
        raise ValueError(
            "Case is required."
        )

    case_id = case.get("case_id")
    claim_id = case.get("claim_id")
    county = case.get("county")

    if not case_id:
        raise ValueError(
            "case_id is required."
        )

    if not claim_id:
        raise ValueError(
            "claim_id is required."
        )

    if not county:
        raise ValueError(
            "county is required."
        )

    priority = dict(
        case.get("priority") or {}
    )

    scores = dict(
        case.get("scores") or {}
    )

    evidence = list(
        case.get("evidence") or []
    )

    context_flags = list(
        case.get("context_flags") or []
    )

    statistical_signals = dict(
        case.get("statistical_signals") or {}
    )

    investigator_notes = list(
        case.get("investigator_notes") or []
    )

    investigator_findings = list(
        case.get("investigator_findings") or []
    )

    now = _utc_now()

    with _connect(db_path) as connection:

        existing = connection.execute(
            """
            SELECT created_at
            FROM investigation_cases
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()

        created_at = (
            existing["created_at"]
            if existing
            else now
        )

        # ----------------------------------------------------
        # EXACTLY 27 COLUMNS
        # EXACTLY 27 VALUES
        # ----------------------------------------------------

        connection.execute(
            """
            INSERT INTO investigation_cases (
                case_id,
                claim_id,
                county,

                facility,
                facility_name,
                insurer,
                claim_type,
                creation,

                claim_amount,

                priority_level,
                priority_score,
                priority_action,

                anomaly_score,
                context_score,
                benchmark_score,

                evidence_json,
                context_flags_json,
                statistical_signals_json,

                investigation_status,

                assigned_investigator,

                investigator_notes_json,
                investigator_findings_json,

                fraud_determination,
                determination_reason,

                created_at,
                updated_at,
                closed_at
            )
            VALUES (
                ?,
                ?,
                ?,

                ?,
                ?,
                ?,
                ?,
                ?,

                ?,

                ?,
                ?,
                ?,

                ?,
                ?,
                ?,

                ?,
                ?,
                ?,

                ?,

                ?,

                ?,
                ?,

                ?,
                ?,

                ?,
                ?,
                ?
            )

            ON CONFLICT(case_id)
            DO UPDATE SET

                claim_id = excluded.claim_id,
                county = excluded.county,

                facility = excluded.facility,
                facility_name = excluded.facility_name,
                insurer = excluded.insurer,
                claim_type = excluded.claim_type,
                creation = excluded.creation,

                claim_amount = excluded.claim_amount,

                priority_level = excluded.priority_level,
                priority_score = excluded.priority_score,
                priority_action = excluded.priority_action,

                anomaly_score = excluded.anomaly_score,
                context_score = excluded.context_score,
                benchmark_score = excluded.benchmark_score,

                evidence_json = excluded.evidence_json,
                context_flags_json =
                    excluded.context_flags_json,
                statistical_signals_json =
                    excluded.statistical_signals_json,

                investigation_status =
                    excluded.investigation_status,

                assigned_investigator =
                    excluded.assigned_investigator,

                investigator_notes_json =
                    excluded.investigator_notes_json,

                investigator_findings_json =
                    excluded.investigator_findings_json,

                fraud_determination =
                    excluded.fraud_determination,

                determination_reason =
                    excluded.determination_reason,

                updated_at =
                    excluded.updated_at,

                closed_at =
                    excluded.closed_at
            """,
            (
                # 1-3
                case_id,
                claim_id,
                county,

                # 4-8
                case.get("facility"),
                case.get("facility_name"),
                case.get("insurer"),
                case.get("claim_type"),
                case.get("creation"),

                # 9
                float(
                    case.get(
                        "claim_amount",
                        0,
                    ) or 0
                ),

                # 10-12
                priority.get(
                    "level",
                    "LOW",
                ),
                float(
                    priority.get(
                        "score",
                        0,
                    ) or 0
                ),
                priority.get(
                    "action"
                ),

                # 13-15
                int(
                    scores.get(
                        "anomaly",
                        0,
                    ) or 0
                ),
                int(
                    scores.get(
                        "context",
                        0,
                    ) or 0
                ),
                int(
                    scores.get(
                        "benchmark",
                        0,
                    ) or 0
                ),

                # 16-18
                _json_dump(evidence),
                _json_dump(context_flags),
                _json_dump(statistical_signals),

                # 19
                case.get(
                    "investigation_status",
                    "OPEN",
                ),

                # 20
                case.get(
                    "assigned_investigator"
                ),

                # 21-22
                _json_dump(
                    investigator_notes
                ),
                _json_dump(
                    investigator_findings
                ),

                # 23-24
                case.get(
                    "fraud_determination"
                ),
                case.get(
                    "determination_reason"
                ),

                # 25-27
                created_at,
                now,
                case.get("closed_at"),
            ),
        )


# ============================================================
# UPDATE CASE
# ============================================================

def update_case(
    case_id: str,
    updates: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Update selected case fields.

    This function exists so workflow code can explicitly
    persist lifecycle changes.
    """

    if not case_id:
        raise ValueError(
            "case_id is required."
        )

    case = get_case(
        case_id,
        db_path=db_path,
    )

    if case is None:
        return None

    case.update(updates)

    save_case(
        case,
        db_path=db_path,
    )

    return get_case(
        case_id,
        db_path=db_path,
    )


# ============================================================
# ADD AUDIT EVENT
# ============================================================

def add_audit_event(
    case_id: str,
    event: Dict[str, Any],
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """
    Persist one structured audit event.

    The case MUST already exist before this function
    is called because investigation_audit.case_id has
    a foreign-key constraint.
    """

    if not case_id:
        raise ValueError(
            "case_id is required."
        )

    if not event:
        raise ValueError(
            "Audit event is required."
        )

    event_type = event.get(
        "event_type"
    )

    if not event_type:
        raise ValueError(
            "event_type is required."
        )

    timestamp = event.get(
        "timestamp"
    ) or _utc_now()

    with _connect(db_path) as connection:

        # Explicit FK safety check.
        exists = connection.execute(
            """
            SELECT 1
            FROM investigation_cases
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()

        if exists is None:
            raise ValueError(
                f"Cannot add audit event: "
                f"case '{case_id}' does not exist."
            )

        cursor = connection.execute(
            """
            INSERT INTO investigation_audit (
                case_id,
                event_type,
                timestamp,
                investigator,
                status,
                previous_status,
                determination,
                previous_investigator,
                note
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                case_id,
                event_type,
                timestamp,
                event.get(
                    "investigator"
                ),
                event.get(
                    "status"
                ),
                event.get(
                    "previous_status"
                ),
                event.get(
                    "determination"
                ),
                event.get(
                    "previous_investigator"
                ),
                event.get(
                    "note"
                ),
            ),
        )

        return int(
            cursor.lastrowid
        )


# ============================================================
# GET CASE
# ============================================================

def get_case(
    case_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a case by case_id.
    """

    if not case_id:
        return None

    with _connect(db_path) as connection:

        row = connection.execute(
            """
            SELECT *
            FROM investigation_cases
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_case(row)


# ============================================================
# GET CASE BY CLAIM ID
# ============================================================

def get_case_by_claim_id(
    claim_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a case by claim_id.
    """

    if not claim_id:
        return None

    with _connect(db_path) as connection:

        row = connection.execute(
            """
            SELECT *
            FROM investigation_cases
            WHERE claim_id = ?
            """,
            (claim_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_case(row)


# ============================================================
# GET AUDIT TRAIL
# ============================================================

def get_audit_trail(
    case_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    Return structured audit events ordered chronologically.
    """

    with _connect(db_path) as connection:

        rows = connection.execute(
            """
            SELECT
                event_id,
                case_id,
                event_type,
                timestamp,
                investigator,
                status,
                previous_status,
                determination,
                previous_investigator,
                note
            FROM investigation_audit
            WHERE case_id = ?
            ORDER BY event_id ASC
            """,
            (case_id,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# LIST CASES
# ============================================================

def list_cases(
    county: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    investigator: Optional[str] = None,
    limit: int = 100,
    db_path: str = DEFAULT_DB_PATH,
) -> List[Dict[str, Any]]:
    """
    List investigation cases with optional filters.
    """

    limit = max(
        1,
        min(
            int(limit),
            1000,
        ),
    )

    query = """
        SELECT *
        FROM investigation_cases
        WHERE 1 = 1
    """

    parameters = []

    if county:
        query += """
            AND county = ?
        """
        parameters.append(
            str(county).upper().strip()
        )

    if status:
        query += """
            AND investigation_status = ?
        """
        parameters.append(status)

    if priority:
        query += """
            AND priority_level = ?
        """
        parameters.append(priority)

    if investigator:
        query += """
            AND assigned_investigator = ?
        """
        parameters.append(investigator)

    query += """
        ORDER BY
            priority_score DESC,
            claim_amount DESC,
            created_at ASC
        LIMIT ?
    """

    parameters.append(limit)

    with _connect(db_path) as connection:

        rows = connection.execute(
            query,
            parameters,
        ).fetchall()

    return [
        _row_to_case(row)
        for row in rows
    ]


# ============================================================
# DELETE CASE
# ============================================================

def delete_case(
    case_id: str,
    db_path: str = DEFAULT_DB_PATH,
) -> bool:
    """
    Delete a case.

    Associated audit events are automatically deleted
    because the database uses ON DELETE CASCADE.
    """

    with _connect(db_path) as connection:

        cursor = connection.execute(
            """
            DELETE FROM investigation_cases
            WHERE case_id = ?
            """,
            (case_id,),
        )

        return cursor.rowcount > 0


# ============================================================
# ROW -> CASE
# ============================================================

def _row_to_case(
    row: sqlite3.Row,
) -> Dict[str, Any]:
    """
    Convert a SQLite row into the application case structure.
    """

    case = {
        "case_id": row["case_id"],
        "claim_id": row["claim_id"],
        "county": row["county"],

        "facility": row["facility"],
        "facility_name": row["facility_name"],
        "insurer": row["insurer"],
        "claim_type": row["claim_type"],
        "creation": row["creation"],

        "claim_amount": row["claim_amount"],

        "priority": {
            "level": row["priority_level"],
            "score": row["priority_score"],
            "action": row["priority_action"],
        },

        "scores": {
            "anomaly": row["anomaly_score"],
            "context": row["context_score"],
            "benchmark": row["benchmark_score"],
            "total_priority": row["priority_score"],
        },

        "evidence": _json_load(
            row["evidence_json"],
            [],
        ),

        "context_flags": _json_load(
            row["context_flags_json"],
            [],
        ),

        "statistical_signals": _json_load(
            row["statistical_signals_json"],
            {},
        ),

        "investigation_status":
            row["investigation_status"],

        "assigned_investigator":
            row["assigned_investigator"],

        "investigator_notes":
            _json_load(
                row["investigator_notes_json"],
                [],
            ),

        "investigator_findings":
            _json_load(
                row["investigator_findings_json"],
                [],
            ),

        "fraud_determination":
            row["fraud_determination"],

        "determination_reason":
            row["determination_reason"],

        "created_at":
            row["created_at"],

        "updated_at":
            row["updated_at"],

        "closed_at":
            row["closed_at"],
    }

    return case