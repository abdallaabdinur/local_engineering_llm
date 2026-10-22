import requests


API_BASE_URL = "http://127.0.0.1:8000"


class InvestigationAPIError(Exception):
    pass


def _request(
    method,
    path,
    json=None,
    params=None,
    timeout=10,
):
    url = f"{API_BASE_URL}{path}"

    try:
        response = requests.request(
            method,
            url,
            json=json,
            params=params,
            timeout=timeout,
        )

    except requests.RequestException as error:
        raise InvestigationAPIError(
            f"Investigation API unavailable: {error}"
        )

    if not response.ok:
        raise InvestigationAPIError(
            f"API error {response.status_code}: "
            f"{response.text}"
        )

    return response.json()


def health():
    return _request(
        "GET",
        "/health",
    )


def list_cases():
    return _request(
        "GET",
        "/cases",
    )


def get_case(case_id):
    return _request(
        "GET",
        f"/cases/{case_id}",
    )


def get_audit_trail(case_id):
    return _request(
        "GET",
        f"/cases/{case_id}/audit",
    )


def build_cases(county, limit=25):
    return _request(
        "POST",
        "/cases/build",
        params={
            "county": county,
            "limit": limit,
        },
    )


def assign_case(
    case_id,
    investigator,
):
    return _request(
        "POST",
        f"/cases/{case_id}/assign",
        json={
            "investigator": investigator,
        },
    )


def change_case_status(
    case_id,
    status,
    investigator=None,
    note=None,
):
    return _request(
        "POST",
        f"/cases/{case_id}/status",
        json={
            "status": status,
            "investigator": investigator,
            "note": note,
        },
    )


def add_note(
    case_id,
    note,
    investigator,
):
    return _request(
        "POST",
        f"/cases/{case_id}/notes",
        json={
            "note": note,
            "investigator": investigator,
        },
    )


def add_finding(
    case_id,
    finding,
    investigator,
):
    return _request(
        "POST",
        f"/cases/{case_id}/findings",
        json={
            "finding": finding,
            "investigator": investigator,
        },
    )


def record_determination(
    case_id,
    determination,
    investigator,
    reason,
):
    return _request(
        "POST",
        f"/cases/{case_id}/determination",
        json={
            "determination": determination,
            "investigator": investigator,
            "reason": reason,
        },
    )


def close_case(
    case_id,
    investigator,
    note="Investigation completed.",
):
    return _request(
        "POST",
        f"/cases/{case_id}/close",
        json={
            "investigator": investigator,
            "note": note,
        },
    )
