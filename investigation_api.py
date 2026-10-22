from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from investigation_repository import (
    initialize_database,
    get_case,
    get_audit_trail,
    list_cases,
)
from investigation_workflow import (
    build_investigation_cases,
    assign_investigator,
    change_status,
    add_investigator_note,
    add_investigator_finding,
    record_determination,
    close_case,
)


app = FastAPI(
    title="Claims Investigation API",
    description="Investigation workflow and audit API for healthcare claims.",
    version="1.0.0",
)


# ------------------------------------------------------------------
# DATABASE INITIALIZATION
# ------------------------------------------------------------------

initialize_database()


# ------------------------------------------------------------------
# REQUEST MODELS
# ------------------------------------------------------------------

class AssignRequest(BaseModel):
    investigator: str


class StatusRequest(BaseModel):
    status: str
    investigator: Optional[str] = None
    note: Optional[str] = None


class NoteRequest(BaseModel):
    note: str
    investigator: str


class FindingRequest(BaseModel):
    finding: str
    investigator: str


class DeterminationRequest(BaseModel):
    determination: str
    investigator: str
    reason: str


class CloseRequest(BaseModel):
    investigator: str
    note: str = "Investigation completed."


# ------------------------------------------------------------------
# HEALTH
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "claims-investigation-api",
    }


# ------------------------------------------------------------------
# BUILD / REFRESH CASES
# ------------------------------------------------------------------

@app.post("/cases/build")
def build_cases(
    county: str,
    limit: int = Query(25, ge=1, le=500),
):
    result = build_investigation_cases(
        county,
        limit=limit,
    )

    return result


# ------------------------------------------------------------------
# LIST CASES
# ------------------------------------------------------------------

@app.get("/cases")
def cases(
    county: Optional[str] = None,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    investigator: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    results = list_cases(
        county=county,
        status=status,
        priority=priority,
        investigator=investigator,
        limit=limit,
    )

    return {
        "count": len(results),
        "cases": results,
    }


# ------------------------------------------------------------------
# GET CASE
# ------------------------------------------------------------------

@app.get("/cases/{case_id}")
def case_detail(case_id: str):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    return case


# ------------------------------------------------------------------
# AUDIT TRAIL
# ------------------------------------------------------------------

@app.get("/cases/{case_id}/audit")
def case_audit(case_id: str):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    audit = get_audit_trail(case_id)

    return {
        "case_id": case_id,
        "event_count": len(audit),
        "audit_trail": audit,
    }


# ------------------------------------------------------------------
# ASSIGN INVESTIGATOR
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/assign")
def assign_case(
    case_id: str,
    request: AssignRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    assign_investigator(
        case,
        request.investigator,
    )

    return case


# ------------------------------------------------------------------
# CHANGE STATUS
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/status")
def update_status(
    case_id: str,
    request: StatusRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    try:
        change_status(
            case,
            request.status,
            investigator=request.investigator,
            note=request.note,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return case


# ------------------------------------------------------------------
# ADD NOTE
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/notes")
def add_note(
    case_id: str,
    request: NoteRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    try:
        add_investigator_note(
            case,
            request.note,
            request.investigator,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return case


# ------------------------------------------------------------------
# ADD FINDING
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/findings")
def add_finding(
    case_id: str,
    request: FindingRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    try:
        add_investigator_finding(
            case,
            request.finding,
            request.investigator,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return case


# ------------------------------------------------------------------
# RECORD DETERMINATION
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/determination")
def determine_case(
    case_id: str,
    request: DeterminationRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    try:
        record_determination(
            case,
            request.determination,
            request.investigator,
            request.reason,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return case


# ------------------------------------------------------------------
# CLOSE CASE
# ------------------------------------------------------------------

@app.post("/cases/{case_id}/close")
def close_investigation(
    case_id: str,
    request: CloseRequest,
):

    case = get_case(case_id)

    if case is None:
        raise HTTPException(
            status_code=404,
            detail="Investigation case not found.",
        )

    try:
        close_case(
            case,
            request.investigator,
            request.note,
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    return case
