import math
from datetime import datetime

from tools import run_select
from anomaly_investigator import analyze_claim_distribution


def _first_row(result):
    if not result:
        return {}

    rows = result.get("rows", [])

    if not rows:
        return {}

    columns = result.get("columns", [])

    return dict(zip(columns, rows[0]))


def _safe_float(value, default=0.0):
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def investigate_county(county):
    """
    Perform a multi-dimensional investigation of an anomalous county.

    Returns evidence rather than just a single anomaly score.
    """

    county = county.strip().upper()

    # ---------------------------------------------------------
    # 1. Distribution analysis
    # ---------------------------------------------------------

    distribution = analyze_claim_distribution(county)

    quantiles = _first_row(
        distribution.get("quantiles")
    )

    # ---------------------------------------------------------
    # 2. Overall county metrics
    # ---------------------------------------------------------

    overview = run_select(
        """
        SELECT
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0) AS total_claim_amount,
            COALESCE(AVG(claim_amount), 0) AS average_claim_amount,
            COALESCE(MIN(claim_amount), 0) AS minimum_claim_amount,
            COALESCE(MAX(claim_amount), 0) AS maximum_claim_amount,
            COUNT(*) FILTER (
                WHERE claim_amount = 0
            ) AS zero_amount_claims,
            COUNT(*) FILTER (
                WHERE claim_amount > 0
            ) AS positive_amount_claims
        FROM claims
        WHERE UPPER(county) = %s
        """.replace("%s", f"'{county}'")
    )

    overview_row = _first_row(overview)

    # ---------------------------------------------------------
    # 3. National comparison
    # ---------------------------------------------------------

    national = run_select(
        f"""
        SELECT
            COUNT(*) AS national_claim_count,
            COALESCE(AVG(claim_amount), 0)
                AS national_average_amount
        FROM claims
        """
    )

    national_row = _first_row(national)

    county_count = _safe_float(
        overview_row.get("claim_count")
    )

    county_avg = _safe_float(
        overview_row.get("average_claim_amount")
    )

    national_count = _safe_float(
        national_row.get("national_claim_count")
    )

    national_avg = _safe_float(
        national_row.get("national_average_amount")
    )

    # ---------------------------------------------------------
    # 4. Concentration by insurer
    # ---------------------------------------------------------

    insurers = run_select(
        f"""
        SELECT
            insurer,
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount,
            COALESCE(AVG(claim_amount), 0)
                AS average_claim_amount
        FROM claims
        WHERE UPPER(county) = '{county}'
        GROUP BY insurer
        ORDER BY claim_count DESC
        LIMIT 10
        """
    )

    # ---------------------------------------------------------
    # 5. Concentration by facility
    # ---------------------------------------------------------

    facilities = run_select(
        f"""
        SELECT
            facility,
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount,
            COALESCE(AVG(claim_amount), 0)
                AS average_claim_amount
        FROM claims
        WHERE UPPER(county) = '{county}'
        GROUP BY facility
        ORDER BY total_claim_amount DESC
        LIMIT 10
        """
    )

    # ---------------------------------------------------------
    # 6. Claim type concentration
    # ---------------------------------------------------------

    claim_types = run_select(
        f"""
        SELECT
            claim_type,
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount,
            COALESCE(AVG(claim_amount), 0)
                AS average_claim_amount
        FROM claims
        WHERE UPPER(county) = '{county}'
        GROUP BY claim_type
        ORDER BY claim_count DESC
        LIMIT 10
        """
    )

    # ---------------------------------------------------------
    # 7. Temporal concentration
    # ---------------------------------------------------------

    monthly = run_select(
        f"""
        SELECT
            DATE_TRUNC(
                'month',
                creation
            ) AS month,
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount,
            COALESCE(AVG(claim_amount), 0)
                AS average_claim_amount
        FROM claims
        WHERE UPPER(county) = '{county}'
          AND creation IS NOT NULL
        GROUP BY DATE_TRUNC('month', creation)
        ORDER BY month ASC
        """
    )

    # ---------------------------------------------------------
    # 8. Positive-amount concentration
    # ---------------------------------------------------------

    positive = run_select(
        f"""
        SELECT
            COUNT(*) AS positive_claims,
            COALESCE(SUM(claim_amount), 0)
                AS positive_total,
            COALESCE(AVG(claim_amount), 0)
                AS positive_average,
            COALESCE(MAX(claim_amount), 0)
                AS positive_max
        FROM claims
        WHERE UPPER(county) = '{county}'
          AND claim_amount > 0
        """
    )

    positive_row = _first_row(positive)

    positive_count = _safe_float(
        positive_row.get("positive_claims")
    )

    # ---------------------------------------------------------
    # 9. Calculate concentration metrics
    # ---------------------------------------------------------

    zero_claims = _safe_float(
        overview_row.get("zero_amount_claims")
    )

    zero_percentage = (
        (zero_claims / county_count) * 100
        if county_count
        else 0
    )

    positive_percentage = (
        (positive_count / county_count) * 100
        if county_count
        else 0
    )

    average_multiplier = (
        county_avg / national_avg
        if national_avg
        else 0
    )

    claim_share = (
        (county_count / national_count) * 100
        if national_count
        else 0
    )

    # ---------------------------------------------------------
    # 10. Identify evidence signals
    # ---------------------------------------------------------

    signals = []

    if average_multiplier >= 5:
        signals.append({
            "type": "AMOUNT_ANOMALY",
            "severity": "HIGH",
            "message": (
                f"Average claim amount is "
                f"{average_multiplier:.1f}x the national average."
            ),
        })

    elif average_multiplier >= 2:
        signals.append({
            "type": "AMOUNT_ANOMALY",
            "severity": "MEDIUM",
            "message": (
                f"Average claim amount is "
                f"{average_multiplier:.1f}x the national average."
            ),
        })

    if zero_percentage >= 80:
        signals.append({
            "type": "ZERO_INFLATION",
            "severity": "HIGH",
            "message": (
                f"{zero_percentage:.1f}% of claims have "
                "zero claim amounts."
            ),
        })

    if positive_percentage <= 10:
        signals.append({
            "type": "POSITIVE_AMOUNT_CONCENTRATION",
            "severity": "HIGH",
            "message": (
                f"Only {positive_percentage:.1f}% of claims "
                "have positive amounts."
            ),
        })

    if county_count >= 0.10 * national_count:
        signals.append({
            "type": "VOLUME_CONCENTRATION",
            "severity": "HIGH",
            "message": (
                f"County represents {claim_share:.2f}% "
                "of all claims."
            ),
        })

    # ---------------------------------------------------------
    # 11. Facility concentration
    # ---------------------------------------------------------

    facility_rows = facilities.get("rows", [])

    if facility_rows and county_count:

        top_facility_count = _safe_float(
            facility_rows[0][1]
        )

        facility_share = (
            top_facility_count / county_count
        ) * 100

        if facility_share >= 50:
            signals.append({
                "type": "FACILITY_CONCENTRATION",
                "severity": "HIGH",
                "message": (
                    f"Top facility accounts for "
                    f"{facility_share:.1f}% of county claims."
                ),
            })

    # ---------------------------------------------------------
    # 12. Insurer concentration
    # ---------------------------------------------------------

    insurer_rows = insurers.get("rows", [])

    if insurer_rows and county_count:

        top_insurer_count = _safe_float(
            insurer_rows[0][1]
        )

        insurer_share = (
            top_insurer_count / county_count
        ) * 100

        if insurer_share >= 90:
            signals.append({
                "type": "INSURER_CONCENTRATION",
                "severity": "HIGH",
                "message": (
                    f"Top insurer accounts for "
                    f"{insurer_share:.1f}% of county claims."
                ),
            })

    # ---------------------------------------------------------
    # 13. Calculate investigation score
    # ---------------------------------------------------------

    score = 0

    severity_weights = {
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    for signal in signals:
        score += severity_weights.get(
            signal["severity"],
            1,
        )

    if score >= 10:
        risk_level = "CRITICAL"

    elif score >= 7:
        risk_level = "HIGH"

    elif score >= 4:
        risk_level = "MEDIUM"

    elif score >= 2:
        risk_level = "LOW"

    else:
        risk_level = "NORMAL"

    # ---------------------------------------------------------
    # 14. Build investigation conclusion
    # ---------------------------------------------------------

    conclusions = []

    if average_multiplier >= 5:
        conclusions.append(
            "The county's average claim amount is "
            "materially abnormal relative to the national baseline."
        )

    if zero_percentage >= 80:
        conclusions.append(
            "The distribution is heavily zero-inflated, "
            "meaning the mean is being driven by a small "
            "subset of positive-value claims."
        )

    if positive_percentage <= 10:
        conclusions.append(
            "A small minority of claims account for "
            "the overwhelming majority of monetary exposure."
        )

    if not conclusions:
        conclusions.append(
            "No single dominant structural anomaly was identified."
        )

    return {
        "county": county,

        "risk_level": risk_level,

        "investigation_score": score,

        "signals": signals,

        "conclusions": conclusions,

        "metrics": {
            "claim_count": county_count,
            "total_claim_amount": _safe_float(
                overview_row.get(
                    "total_claim_amount"
                )
            ),
            "average_claim_amount": county_avg,
            "national_average_amount": national_avg,
            "average_multiplier": average_multiplier,
            "zero_claims": zero_claims,
            "zero_percentage": zero_percentage,
            "positive_claims": positive_count,
            "positive_percentage": positive_percentage,
            "claim_share_percentage": claim_share,
            "maximum_claim_amount": _safe_float(
                overview_row.get(
                    "maximum_claim_amount"
                )
            ),
        },

        "distribution": distribution,

        "insurers": insurers,

        "facilities": facilities,

        "claim_types": claim_types,

        "monthly": monthly,

        "positive_amount_analysis": positive,
    }


if __name__ == "__main__":

    import json

    result = investigate_county("KITUI")

    print(
        json.dumps(
            result,
            default=str,
            indent=2,
        )
    )
