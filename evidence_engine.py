"""
Evidence Engine
---------------
Converts investigation signals into concrete database evidence.

The engine is intentionally deterministic:
- SQL does the measurement.
- Python interprets the measurements.
- The LLM is not trusted to invent evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List

from tools import run_select


def _first_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a one-row database result into a dictionary."""
    columns = result.get("columns", [])
    rows = result.get("rows", [])

    if not columns or not rows:
        return {}

    return dict(zip(columns, rows[0]))


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert database numeric values to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Safely convert database numeric values to int."""
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def get_basic_evidence(county: str) -> Dict[str, Any]:
    """
    Collect the core monetary evidence for a county.
    """

    sql = """
    SELECT
        COUNT(*) AS claim_count,

        COUNT(*) FILTER (
            WHERE claim_amount > 0
        ) AS positive_claims,

        COUNT(*) FILTER (
            WHERE claim_amount = 0
        ) AS zero_claims,

        COALESCE(SUM(claim_amount), 0) AS total_claim_amount,

        COALESCE(AVG(claim_amount), 0) AS average_claim_amount,

        COALESCE(
            MAX(claim_amount),
            0
        ) AS maximum_claim_amount

    FROM claims

    WHERE county = %s
    """

    # run_select currently accepts SQL directly, so safely quote
    # the normalized county value.
    safe_county = county.replace("'", "''")

    sql = sql.replace("%s", f"'{safe_county}'")

    result = run_select(sql)

    row = _first_row(result)

    claim_count = _safe_int(row.get("claim_count"))
    positive_claims = _safe_int(row.get("positive_claims"))
    zero_claims = _safe_int(row.get("zero_claims"))

    positive_percentage = (
        positive_claims / claim_count * 100
        if claim_count
        else 0.0
    )

    zero_percentage = (
        zero_claims / claim_count * 100
        if claim_count
        else 0.0
    )

    return {
        "claim_count": claim_count,
        "positive_claims": positive_claims,
        "zero_claims": zero_claims,
        "positive_percentage": round(
            positive_percentage,
            2,
        ),
        "zero_percentage": round(
            zero_percentage,
            2,
        ),
        "total_claim_amount": _safe_float(
            row.get("total_claim_amount")
        ),
        "average_claim_amount": _safe_float(
            row.get("average_claim_amount")
        ),
        "maximum_claim_amount": _safe_float(
            row.get("maximum_claim_amount")
        ),
    }


def get_insurer_concentration(
    county: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Identify insurer concentration inside a county.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        insurer,
        COUNT(*) AS claim_count,
        COALESCE(
            SUM(claim_amount),
            0
        ) AS total_claim_amount,
        COALESCE(
            AVG(claim_amount),
            0
        ) AS average_claim_amount

    FROM claims

    WHERE county = '{safe_county}'

    GROUP BY insurer

    ORDER BY claim_count DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def get_facility_concentration(
    county: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Identify facilities responsible for the greatest
    monetary exposure in the county.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        facility,
        facility_name,
        COUNT(*) AS claim_count,

        COALESCE(
            SUM(claim_amount),
            0
        ) AS total_claim_amount,

        COALESCE(
            AVG(claim_amount),
            0
        ) AS average_claim_amount,

        COALESCE(
            MAX(claim_amount),
            0
        ) AS maximum_claim_amount

    FROM claims

    WHERE county = '{safe_county}'

    GROUP BY
        facility,
        facility_name

    ORDER BY total_claim_amount DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def get_claim_type_concentration(
    county: str,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Identify dominant claim types.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        claim_type,
        COUNT(*) AS claim_count,

        COALESCE(
            SUM(claim_amount),
            0
        ) AS total_claim_amount,

        COALESCE(
            AVG(claim_amount),
            0
        ) AS average_claim_amount

    FROM claims

    WHERE county = '{safe_county}'

    GROUP BY claim_type

    ORDER BY total_claim_amount DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def get_high_value_claims(
    county: str,
    limit: int = 25,
) -> Dict[str, Any]:
    """
    Retrieve the highest-value claims in the county.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        claim_id,
        claim_amount,
        facility,
        facility_name,
        insurer,
        claim_type,
        creation

    FROM claims

    WHERE county = '{safe_county}'
      AND claim_amount > 0

    ORDER BY claim_amount DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def get_amount_repetition(
    county: str,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    Detect repeated exact claim amounts.

    Repeated amounts can be legitimate, but unusually
    concentrated amounts are useful investigative signals.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        claim_amount,
        COUNT(*) AS claim_count,
        COALESCE(
            SUM(claim_amount),
            0
        ) AS total_claim_amount

    FROM claims

    WHERE county = '{safe_county}'
      AND claim_amount > 0

    GROUP BY claim_amount

    ORDER BY claim_count DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def get_daily_concentration(
    county: str,
    limit: int = 15,
) -> Dict[str, Any]:
    """
    Detect unusually concentrated claim activity by day.
    """

    safe_county = county.replace("'", "''")

    sql = f"""
    SELECT
        DATE(creation) AS claim_date,
        COUNT(*) AS claim_count,

        COALESCE(
            SUM(claim_amount),
            0
        ) AS total_claim_amount,

        COALESCE(
            AVG(claim_amount),
            0
        ) AS average_claim_amount

    FROM claims

    WHERE county = '{safe_county}'
      AND creation IS NOT NULL

    GROUP BY DATE(creation)

    ORDER BY claim_count DESC

    LIMIT {int(limit)}
    """

    return run_select(sql)


def calculate_concentration(
    rows_result: Dict[str, Any],
    total_exposure: float,
) -> List[Dict[str, Any]]:
    """
    Calculate exposure concentration for returned rows.

    Works with insurer, facility and claim-type results.
    """

    columns = rows_result.get("columns", [])
    rows = rows_result.get("rows", [])

    if (
        not columns
        or not rows
        or total_exposure <= 0
    ):
        return []

    evidence = []

    for row in rows:

        item = dict(
            zip(columns, row)
        )

        exposure = _safe_float(
            item.get("total_claim_amount")
        )

        share = (
            exposure / total_exposure * 100
        )

        item["exposure_percentage"] = round(
            share,
            2,
        )

        evidence.append(item)

    return evidence


def build_evidence(county: str) -> Dict[str, Any]:
    """
    Run the complete evidence collection process.
    """

    county = county.strip().upper()

    basic = get_basic_evidence(county)

    total_exposure = basic[
        "total_claim_amount"
    ]

    insurers = get_insurer_concentration(
        county
    )

    facilities = get_facility_concentration(
        county
    )

    claim_types = get_claim_type_concentration(
        county
    )

    high_value_claims = get_high_value_claims(
        county
    )

    repeated_amounts = get_amount_repetition(
        county
    )

    daily_activity = get_daily_concentration(
        county
    )

    insurer_evidence = calculate_concentration(
        insurers,
        total_exposure,
    )

    facility_evidence = calculate_concentration(
        facilities,
        total_exposure,
    )

    claim_type_evidence = calculate_concentration(
        claim_types,
        total_exposure,
    )

    return {
        "county": county,

        "basic": basic,

        "insurers": insurer_evidence,

        "facilities": facility_evidence,

        "claim_types": claim_type_evidence,

        "high_value_claims": high_value_claims,

        "repeated_amounts": repeated_amounts,

        "daily_activity": daily_activity,
    }


def summarize_evidence(
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert raw evidence into concise deterministic findings.
    """

    basic = evidence.get(
        "basic",
        {},
    )

    findings: List[str] = []

    positive_percentage = _safe_float(
        basic.get("positive_percentage")
    )

    zero_percentage = _safe_float(
        basic.get("zero_percentage")
    )

    if zero_percentage >= 90:
        findings.append(
            f"{zero_percentage:.1f}% of claims "
            "have zero monetary value."
        )

    if positive_percentage <= 10:
        findings.append(
            f"Only {positive_percentage:.1f}% "
            "of claims have positive monetary value."
        )

    facilities = evidence.get(
        "facilities",
        [],
    )

    if facilities:
        top_facility = facilities[0]

        facility_share = _safe_float(
            top_facility.get(
                "exposure_percentage"
            )
        )

        if facility_share >= 50:
            findings.append(
                "The dominant facility accounts "
                f"for {facility_share:.1f}% of "
                "county monetary exposure."
            )

    insurers = evidence.get(
        "insurers",
        [],
    )

    if insurers:
        top_insurer = insurers[0]

        insurer_share = _safe_float(
            top_insurer.get(
                "exposure_percentage"
            )
        )

        if insurer_share >= 80:
            findings.append(
                "The dominant insurer accounts "
                f"for {insurer_share:.1f}% of "
                "county monetary exposure."
            )

    repeated_amounts = evidence.get(
        "repeated_amounts",
        {}
    )

    repeated_rows = repeated_amounts.get(
        "rows",
        [],
    )

    repeated_columns = repeated_amounts.get(
        "columns",
        [],
    )

    if repeated_rows and repeated_columns:

        repeated = [
            dict(
                zip(
                    repeated_columns,
                    row,
                )
            )
            for row in repeated_rows
        ]

        if repeated:

            highest_frequency = _safe_int(
                repeated[0].get(
                    "claim_count"
                )
            )

            if highest_frequency >= 10:
                findings.append(
                    "A repeated exact claim amount "
                    f"appears {highest_frequency} "
                    "or more times among positive claims."
                )

    return {
        "findings": findings,
        "finding_count": len(findings),
    }


def investigate_evidence(
    county: str,
) -> Dict[str, Any]:
    """
    Public entry point.

    Collect evidence and produce deterministic findings.
    """

    evidence = build_evidence(county)

    summary = summarize_evidence(
        evidence
    )

    return {
        "county": county.strip().upper(),
        "evidence": evidence,
        "summary": summary,
    }
