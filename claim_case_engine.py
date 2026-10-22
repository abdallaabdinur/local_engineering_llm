from claim_decision_engine import decide_claims


def _safe_float(value):
    """
    Safely convert a value to float.
    """
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value):
    """
    Safely convert a value to int.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _determine_action(priority_level):
    """
    Convert investigation priority into an operational action.

    This is an investigation workflow decision,
    NOT a fraud determination.
    """

    if priority_level == "CRITICAL":
        return "IMMEDIATE_INVESTIGATION"

    if priority_level == "HIGH":
        return "PRIORITY_INVESTIGATION"

    if priority_level == "MEDIUM":
        return "SECONDARY_REVIEW"

    return "ROUTINE_MONITORING"


def _build_case(claim, county):
    """
    Convert a scored claim into an auditable investigation case.
    """

    priority_level = (
        claim.get("priority_level")
        or "LOW"
    )

    priority_score = _safe_float(
        claim.get("priority_score")
    )

    anomaly_score = _safe_int(
        claim.get("anomaly_score")
    )

    context_score = _safe_int(
        claim.get("context_score")
    )

    benchmark_score = _safe_int(
        claim.get("benchmark_score")
    )

    claim_amount = _safe_float(
        claim.get("claim_amount")
    )

    amount_frequency = _safe_int(
        claim.get("amount_frequency")
    )

    facility_exposure = _safe_float(
        claim.get("facility_exposure_percentage")
    )

    insurer_exposure = _safe_float(
        claim.get("insurer_exposure_percentage")
    )

    daily_zscore = _safe_float(
        claim.get("daily_amount_zscore")
    )

    national_percentile = _safe_float(
        claim.get("national_percentile")
    )

    # Support both field names:
    # peer_percentile and claim_type_percentile.
    peer_percentile = _safe_float(
        claim.get(
            "peer_percentile",
            claim.get("claim_type_percentile")
        )
    )

    evidence = list(
        claim.get("evidence") or []
    )

    context_flags = list(
        claim.get("context_flags") or []
    )

    action = _determine_action(
        priority_level
    )

    return {
        "case_id": claim.get(
            "claim_id"
        ),

        "claim_id": claim.get(
            "claim_id"
        ),

        "county": county,

        "facility": claim.get(
            "facility"
        ),

        "facility_name": claim.get(
            "facility_name"
        ),

        "insurer": claim.get(
            "insurer"
        ),

        "claim_type": claim.get(
            "claim_type"
        ),

        "creation": claim.get(
            "creation"
        ),

        "claim_amount": claim_amount,

        "priority": {
            "level": priority_level,
            "score": round(
                priority_score,
                2,
            ),
            "action": action,
        },

        "scores": {
            "anomaly": anomaly_score,
            "context": context_score,
            "benchmark": benchmark_score,
            "total_priority": round(
                priority_score,
                2,
            ),
        },

        "evidence": evidence,

        "context_flags": context_flags,

        "statistical_signals": {
            "amount_frequency": amount_frequency,

            "facility_exposure_percentage": round(
                facility_exposure,
                2,
            ),

            "insurer_exposure_percentage": round(
                insurer_exposure,
                2,
            ),

            "daily_amount_zscore": round(
                daily_zscore,
                2,
            ),

            "national_percentile": round(
                national_percentile,
                2,
            ),

            "peer_percentile": round(
                peer_percentile,
                2,
            ),
        },

        "investigation_status": "OPEN",

        # No fraud determination is made
        # by this engine.
        "fraud_determination": None,
    }


def build_claim_cases(county, limit=25):
    """
    Build auditable investigation cases for a county.

    The underlying decision engine determines priority.
    This engine converts those decisions into structured
    investigation cases.

    No fraud determination is made here.
    """

    county = (
        county.upper()
        .strip()
    )

    decision_result = decide_claims(
        county,
        limit=limit,
    )

    if not decision_result:
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "summary": {},
        }

    if decision_result.get("error"):
        return {
            "county": county,
            "case_count": 0,
            "cases": [],
            "error": decision_result["error"],
            "summary": {},
        }

    decision_claims = (
        decision_result.get("claims")
        or []
    )

    cases = []

    for claim in decision_claims:
        case = _build_case(
            claim,
            county,
        )

        cases.append(case)

    critical_count = sum(
        1
        for case in cases
        if case["priority"]["level"] == "CRITICAL"
    )

    high_count = sum(
        1
        for case in cases
        if case["priority"]["level"] == "HIGH"
    )

    medium_count = sum(
        1
        for case in cases
        if case["priority"]["level"] == "MEDIUM"
    )

    low_count = sum(
        1
        for case in cases
        if case["priority"]["level"] == "LOW"
    )

    immediate_count = sum(
        1
        for case in cases
        if case["priority"]["action"]
        == "IMMEDIATE_INVESTIGATION"
    )

    priority_count = sum(
        1
        for case in cases
        if case["priority"]["action"]
        == "PRIORITY_INVESTIGATION"
    )

    secondary_count = sum(
        1
        for case in cases
        if case["priority"]["action"]
        == "SECONDARY_REVIEW"
    )

    monitoring_count = sum(
        1
        for case in cases
        if case["priority"]["action"]
        == "ROUTINE_MONITORING"
    )

    return {
        "county": county,

        "case_count": len(cases),

        "cases": cases,

        "summary": {
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,

            "immediate_investigation": immediate_count,
            "priority_investigation": priority_count,
            "secondary_review": secondary_count,
            "routine_monitoring": monitoring_count,
        },
    }


def get_critical_cases(county, limit=25):
    """
    Return only CRITICAL investigation cases.
    """

    result = build_claim_cases(
        county,
        limit=limit,
    )

    if result.get("error"):
        return result

    critical_cases = [
        case
        for case in result["cases"]
        if case["priority"]["level"] == "CRITICAL"
    ]

    return {
        "county": result["county"],

        "case_count": len(
            critical_cases
        ),

        "cases": critical_cases,

        "summary": {
            "critical": len(
                critical_cases
            ),
        },
    }


def get_investigation_queue(
    county,
    limit=25,
):
    """
    Return cases ordered by investigation priority.

    CRITICAL cases appear first,
    followed by HIGH, MEDIUM, and LOW.
    """

    result = build_claim_cases(
        county,
        limit=limit,
    )

    if result.get("error"):
        return result

    priority_order = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    queue = sorted(
        result["cases"],
        key=lambda case: (
            priority_order.get(
                case["priority"]["level"],
                99,
            ),
            -case["priority"]["score"],
            -case["claim_amount"],
        ),
    )

    return {
        "county": result["county"],

        "queue_count": len(queue),

        "queue": queue,

        "summary": result["summary"],
    }

