"""
Investigation Workflow

Manages the investigation lifecycle:

BUILD
    ->
ASSIGN
    ->
UNDER_REVIEW / ESCALATED
    ->
NOTES / FINDINGS
    ->
DETERMINATION
    ->
CLOSED

The workflow does NOT automatically determine fraud.
A determination is explicitly recorded by an investigator.
"""

from datetime import datetime, timezone

from claim_evidence_engine import (
    build_evidence_packages,
)

from investigation_repository import (
    initialize_database,
    save_case,
    update_case,
    add_audit_event,
    get_case as repository_get_case,
    get_audit_trail,
    list_cases,
)


# ============================================================
# CONSTANTS
# ============================================================

VALID_STATUSES = {
    "OPEN",
    "UNDER_REVIEW",
    "ESCALATED",
    "CLOSED",
}


VALID_DETERMINATIONS = {
    "FRAUD_CONFIRMED",
    "NO_FRAUD",
    "INSUFFICIENT_EVIDENCE",
    "FALSE_POSITIVE",
}


STATUS_TRANSITIONS = {
    "OPEN": {
        "UNDER_REVIEW",
        "ESCALATED",
        "CLOSED",
    },

    "UNDER_REVIEW": {
        "ESCALATED",
        "CLOSED",
    },

    "ESCALATED": {
        "UNDER_REVIEW",
        "CLOSED",
    },

    "CLOSED": set(),
}


# ============================================================
# TIME
# ============================================================

def _timestamp():
    """
    Return UTC ISO-8601 timestamp.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


# ============================================================
# AUDIT EVENT
# ============================================================

def _add_audit_event(
    case,
    event_type,
    investigator=None,
    status=None,
    previous_status=None,
    determination=None,
    previous_investigator=None,
    note=None,
    persist=True,
):
    """
    Create a structured audit event.

    If persist=True:
        the case must already exist in SQLite.
    """

    event = {
        "event_type": event_type,
        "timestamp": _timestamp(),
        "investigator": investigator,
        "status": status,
        "previous_status": previous_status,
        "determination": determination,
        "previous_investigator":
            previous_investigator,
        "note": note,
    }

    case.setdefault(
        "audit_trail",
        [],
    ).append(event)

    if persist:
        add_audit_event(
            case["case_id"],
            event,
        )

    return event


# ============================================================
# CASE CREATION
# ============================================================

def _build_workflow_case(
    evidence_package,
):
    """
    Convert an evidence package into an investigation case.

    No fraud determination is made here.
    """

    priority = dict(
        evidence_package.get(
            "priority"
        ) or {}
    )

    case = {
        "case_id":
            evidence_package.get(
                "case_id"
            )
            or evidence_package.get(
                "claim_id"
            ),

        "claim_id":
            evidence_package.get(
                "claim_id"
            ),

        "county":
            evidence_package.get(
                "county"
            ),

        "facility":
            evidence_package.get(
                "facility"
            ),

        "facility_name":
            evidence_package.get(
                "facility_name"
            ),

        "insurer":
            evidence_package.get(
                "insurer"
            ),

        "claim_type":
            evidence_package.get(
                "claim_type"
            ),

        "creation":
            evidence_package.get(
                "creation"
            ),

        "claim_amount":
            evidence_package.get(
                "claim_amount",
                0.0,
            ),

        "priority":
            priority,

        "scores":
            dict(
                evidence_package.get(
                    "scores"
                ) or {}
            ),

        "evidence":
            list(
                evidence_package.get(
                    "evidence"
                ) or []
            ),

        "statistical_evidence":
            list(
                evidence_package.get(
                    "statistical_evidence"
                ) or []
            ),

        "investigation_questions":
            list(
                evidence_package.get(
                    "investigation_questions"
                ) or []
            ),

        "recommended_documents":
            list(
                evidence_package.get(
                    "recommended_documents"
                ) or []
            ),

        "context_flags":
            list(
                evidence_package.get(
                    "context_flags"
                ) or []
            ),

        "statistical_signals":
            dict(
                evidence_package.get(
                    "statistical_signals"
                ) or {}
            ),

        "investigation_status":
            "OPEN",

        "assigned_investigator":
            None,

        "investigator_notes":
            [],

        "investigator_findings":
            [],

        "fraud_determination":
            None,

        "determination_reason":
            None,

        "closed_at":
            None,

        "audit_trail":
            [],
    }

    return case


# ============================================================
# SUMMARY
# ============================================================

def _build_summary(cases):
    """
    Build workflow status summary.
    """

    summary = {
        "open": 0,
        "under_review": 0,
        "escalated": 0,
        "closed": 0,
        "determined": 0,
    }

    for case in cases:

        status = case.get(
            "investigation_status"
        )

        if status == "OPEN":
            summary["open"] += 1

        elif status == "UNDER_REVIEW":
            summary["under_review"] += 1

        elif status == "ESCALATED":
            summary["escalated"] += 1

        elif status == "CLOSED":
            summary["closed"] += 1

        if (
            case.get(
                "fraud_determination"
            )
            is not None
        ):
            summary["determined"] += 1

    return summary


# ============================================================
# BUILD INVESTIGATION CASES
# ============================================================

def build_investigation_cases(
    county,
    limit=25,
):
    """
    Build investigation cases.

    Persistence order:

        1. Build case
        2. Save case to SQLite
        3. Add CASE_CREATED only if this is a genuinely new case

    Rebuilding an existing case must NOT create another
    CASE_CREATED audit event.
    """

    county = str(
        county or ""
    ).upper().strip()

    if not county:
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
            "error": "County is required.",
        }

    try:
        limit = int(limit)
    except (
        TypeError,
        ValueError,
    ):
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
            "error": "Limit must be an integer.",
        }

    if limit <= 0:
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
            "error": "Limit must be greater than zero.",
        }

    initialize_database()

    # --------------------------------------------------------
    # BUILD EVIDENCE
    # --------------------------------------------------------

    result = build_evidence_packages(
        county,
        limit=limit,
    )

    if not result:
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
            "error": "Evidence engine returned no result.",
        }

    if result.get("error"):
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
            "error": result["error"],
        }

    packages = list(
        result.get("packages") or []
    )

    cases = []

    # --------------------------------------------------------
    # CREATE / UPDATE CASES
    # --------------------------------------------------------

    for package in packages:

        case = _build_workflow_case(
            package
        )

        case_id = case["case_id"]

        # Check whether this case already exists.
        existing_case = repository_get_case(
            case_id
        )

        if existing_case is None:

            # ----------------------------------------------
            # NEW CASE
            # ----------------------------------------------

            save_case(case)

            _add_audit_event(
                case,
                event_type="CASE_CREATED",
                status="OPEN",
                note="Investigation case created.",
                persist=True,
            )

        else:

            # ----------------------------------------------
            # EXISTING CASE
            #
            # IMPORTANT:
            # Preserve investigation state.
            # The evidence engine should not reset an
            # investigator's work.
            # ----------------------------------------------

            case["investigation_status"] = (
                existing_case.get(
                    "investigation_status"
                )
                or "OPEN"
            )

            case["assigned_investigator"] = (
                existing_case.get(
                    "assigned_investigator"
                )
            )

            case["investigator_notes"] = list(
                existing_case.get(
                    "investigator_notes"
                ) or []
            )

            case["investigator_findings"] = list(
                existing_case.get(
                    "investigator_findings"
                ) or []
            )

            case["fraud_determination"] = (
                existing_case.get(
                    "fraud_determination"
                )
            )

            case["determination_reason"] = (
                existing_case.get(
                    "determination_reason"
                )
            )

            case["created_at"] = (
                existing_case.get(
                    "created_at"
                )
            )

            case["updated_at"] = (
                existing_case.get(
                    "updated_at"
                )
            )

            case["closed_at"] = (
                existing_case.get(
                    "closed_at"
                )
            )

            # Do NOT add CASE_CREATED.
            save_case(case)

        # ----------------------------------------------------
        # Reload authoritative database state.
        # ----------------------------------------------------

        persisted_case = repository_get_case(
            case_id
        )

        if persisted_case is not None:

            persisted_case["audit_trail"] = (
                get_audit_trail(case_id)
            )

            cases.append(
                persisted_case
            )

        else:
            cases.append(case)

    summary = _build_summary(
        cases
    )

    return {
        "county": county,
        "case_count": len(cases),
        "cases": cases,
        "summary": summary,
    }

# ============================================================
# ASSIGN INVESTIGATOR
# ============================================================

def assign_investigator(
    case,
    investigator,
):
    """
    Assign an investigator.
    """

    if not investigator:
        raise ValueError(
            "Investigator is required."
        )

    if (
        case.get(
            "investigation_status"
        )
        == "CLOSED"
    ):
        raise ValueError(
            "Cannot assign investigator "
            "to a closed case."
        )

    previous_investigator = case.get(
        "assigned_investigator"
    )

    case[
        "assigned_investigator"
    ] = investigator

    save_case(case)

    _add_audit_event(
        case,
        event_type="INVESTIGATOR_ASSIGNED",
        investigator=investigator,
        status=case.get(
            "investigation_status"
        ),
        previous_investigator=
            previous_investigator,
        note=(
            "Previous investigator: "
            f"{previous_investigator}"
        ),
    )

    return case


# ============================================================
# CHANGE STATUS
# ============================================================

def change_status(
    case,
    new_status,
    investigator=None,
    note=None,
):
    """
    Change investigation status using the allowed
    state-transition graph.
    """

    new_status = str(
        new_status or ""
    ).upper().strip()

    if new_status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {new_status}"
        )

    current_status = case.get(
        "investigation_status"
    )

    if current_status == new_status:
        raise ValueError(
            f"Case is already {new_status}."
        )

    allowed = STATUS_TRANSITIONS.get(
        current_status,
        set(),
    )

    if new_status not in allowed:
        raise ValueError(
            "Invalid status transition: "
            f"{current_status} -> "
            f"{new_status}"
        )

    if (
        new_status == "CLOSED"
        and case.get(
            "fraud_determination"
        ) is None
    ):
        raise ValueError(
            "A determination must be "
            "recorded before closing "
            "a case."
        )

    previous_status = current_status

    case[
        "investigation_status"
    ] = new_status

    if new_status == "CLOSED":
        case["closed_at"] = _timestamp()

    save_case(case)

    _add_audit_event(
        case,
        event_type="STATUS_CHANGED",
        investigator=investigator,
        status=new_status,
        previous_status=previous_status,
        note=note,
    )

    return case


# ============================================================
# ADD INVESTIGATOR NOTE
# ============================================================

def add_investigator_note(
    case,
    note,
    investigator,
):
    """
    Add an investigator note.
    """

    if not investigator:
        raise ValueError(
            "Investigator is required."
        )

    if not note:
        raise ValueError(
            "Note is required."
        )

    if (
        case.get(
            "investigation_status"
        )
        == "CLOSED"
    ):
        raise ValueError(
            "Cannot add note to a closed case."
        )

    entry = {
        "timestamp": _timestamp(),
        "investigator": investigator,
        "note": str(note),
    }

    case.setdefault(
        "investigator_notes",
        [],
    ).append(entry)

    save_case(case)

    _add_audit_event(
        case,
        event_type="INVESTIGATOR_NOTE",
        investigator=investigator,
        status=case.get(
            "investigation_status"
        ),
        note=str(note),
    )

    return entry


# ============================================================
# ADD INVESTIGATOR FINDING
# ============================================================

def add_investigator_finding(
    case,
    finding,
    investigator,
):
    """
    Add an investigator finding.
    """

    if not investigator:
        raise ValueError(
            "Investigator is required."
        )

    if not finding:
        raise ValueError(
            "Finding is required."
        )

    if (
        case.get(
            "investigation_status"
        )
        == "CLOSED"
    ):
        raise ValueError(
            "Cannot add finding to "
            "a closed case."
        )

    entry = {
        "timestamp": _timestamp(),
        "investigator": investigator,
        "finding": str(finding),
    }

    case.setdefault(
        "investigator_findings",
        [],
    ).append(entry)

    save_case(case)

    _add_audit_event(
        case,
        event_type="INVESTIGATOR_FINDING",
        investigator=investigator,
        status=case.get(
            "investigation_status"
        ),
        note=str(finding),
    )

    return entry


# ============================================================
# RECORD DETERMINATION
# ============================================================

def record_determination(
    case,
    determination,
    investigator,
    reason,
):
    """
    Record an explicit investigator determination.

    The system does NOT infer this automatically.
    """

    determination = str(
        determination or ""
    ).upper().strip()

    if determination not in VALID_DETERMINATIONS:
        raise ValueError(
            "Invalid determination: "
            f"{determination}"
        )

    if not investigator:
        raise ValueError(
            "Investigator is required."
        )

    if not reason:
        raise ValueError(
            "Determination reason is required."
        )

    if (
        case.get(
            "investigation_status"
        )
        == "CLOSED"
    ):
        raise ValueError(
            "Cannot determine a closed case."
        )

    previous_determination = case.get(
        "fraud_determination"
    )

    case[
        "fraud_determination"
    ] = determination

    case[
        "determination_reason"
    ] = str(reason)

    save_case(case)

    _add_audit_event(
        case,
        event_type="DETERMINATION_RECORDED",
        investigator=investigator,
        status=case.get(
            "investigation_status"
        ),
        determination=determination,
        note=str(reason),
    )

    return case


# ============================================================
# CLOSE CASE
# ============================================================

def close_case(
    case,
    investigator,
    note="Investigation completed.",
):
    """
    Close a case.

    A determination must exist before closure.
    """

    if not investigator:
        raise ValueError(
            "Investigator is required."
        )

    if (
        case.get(
            "fraud_determination"
        )
        is None
    ):
        raise ValueError(
            "Cannot close case without "
            "a determination."
        )

    if (
        case.get(
            "investigation_status"
        )
        == "CLOSED"
    ):
        raise ValueError(
            "Case is already closed."
        )

    return change_status(
        case,
        "CLOSED",
        investigator=investigator,
        note=note,
    )


# ============================================================
# GET OPEN CASES
# ============================================================

def get_open_cases(
    county=None,
    limit=100,
):
    """
    Return open investigation cases.
    """

    return list_cases(
        county=county,
        status="OPEN",
        limit=limit,
    )


# ============================================================
# GET CASE
# ============================================================

def get_case(
    case_id,
):
    """
    Retrieve a persisted investigation case.

    Audit trail is attached to the returned object.
    """

    case = repository_get_case(
        case_id
    )

    if case is None:
        return None

    case["audit_trail"] = (
        get_audit_trail(
            case_id
        )
    )

    return case