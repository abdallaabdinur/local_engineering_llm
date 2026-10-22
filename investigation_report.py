"""
Investigation Report Layer
---------------------------

Converts the raw output from investigation_engine.py into
a clean, reliable executive investigation report.

This module intentionally reads the actual structure returned by:

    investigate_engine.investigate_county()

Expected structure:

{
    "county": ...,
    "risk_level": ...,
    "investigation_score": ...,
    "signals": [...],
    "conclusions": [...],
    "metrics": {...},
    "distribution": {
        "quantiles": {...},
        "buckets": {...},
        "top_claims": {...}
    },
    "insurers": {...},
    "facilities": {...},
    "claim_types": {...},
    "monthly": {...},
    "positive_amount_analysis": {...}
}
"""

from typing import Any, Dict, List, Optional


# ============================================================
# SAFE HELPERS
# ============================================================

def _first_row(section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract the first row from a database-result dictionary.
    Returns an empty dictionary if the structure is missing.
    """

    if not isinstance(section, dict):
        return {}

    columns = section.get("columns", [])
    rows = section.get("rows", [])

    if not columns or not rows:
        return {}

    row = rows[0]

    return dict(zip(columns, row))


def _rows(section: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a database-result dictionary into a list of dictionaries.
    """

    if not isinstance(section, dict):
        return []

    columns = section.get("columns", [])
    rows = section.get("rows", [])

    if not columns or not rows:
        return []

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    """

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_percentage(
    numerator: float,
    denominator: float,
) -> float:
    """
    Calculate a percentage safely.
    """

    if denominator == 0:
        return 0.0

    return (numerator / denominator) * 100.0


def _round(value: Any, digits: int = 2) -> float:
    """
    Safe rounding helper.
    """

    return round(_safe_float(value), digits)


# ============================================================
# MAIN REPORT BUILDER
# ============================================================

def build_investigation_report(
    investigation: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build an executive investigation report from the output
    returned by investigate_county().
    """

    if not isinstance(investigation, dict):
        raise TypeError(
            "investigation must be a dictionary"
        )

    # --------------------------------------------------------
    # CORE INFORMATION
    # --------------------------------------------------------

    county = investigation.get(
        "county",
        "UNKNOWN",
    )

    risk_level = investigation.get(
        "risk_level",
        "UNKNOWN",
    )

    investigation_score = _safe_float(
        investigation.get(
            "investigation_score",
            0,
        )
    )

    signals = investigation.get(
        "signals",
        [],
    )

    conclusions = investigation.get(
        "conclusions",
        [],
    )

    metrics = investigation.get(
        "metrics",
        {},
    )

    distribution = investigation.get(
        "distribution",
        {},
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    claim_count = _safe_float(
        metrics.get(
            "claim_count",
            0,
        )
    )

    total_claim_amount = _safe_float(
        metrics.get(
            "total_claim_amount",
            0,
        )
    )

    county_average = _safe_float(
        metrics.get(
            "average_claim_amount",
            0,
        )
    )

    national_average = _safe_float(
        metrics.get(
            "national_average_amount",
            0,
        )
    )

    average_multiplier = _safe_float(
        metrics.get(
            "average_multiplier",
            0,
        )
    )

    zero_claims = _safe_float(
        metrics.get(
            "zero_claims",
            0,
        )
    )

    zero_percentage = _safe_float(
        metrics.get(
            "zero_percentage",
            0,
        )
    )

    positive_claims = _safe_float(
        metrics.get(
            "positive_claims",
            0,
        )
    )

    positive_percentage = _safe_float(
        metrics.get(
            "positive_percentage",
            0,
        )
    )

    claim_share_percentage = _safe_float(
        metrics.get(
            "claim_share_percentage",
            0,
        )
    )

    maximum_claim_amount = _safe_float(
        metrics.get(
            "maximum_claim_amount",
            0,
        )
    )

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    quantiles_section = distribution.get(
        "quantiles",
        {},
    )

    buckets_section = distribution.get(
        "buckets",
        {},
    )

    top_claims_section = distribution.get(
        "top_claims",
        {},
    )

    quantiles = _first_row(
        quantiles_section
    )

    buckets = _rows(
        buckets_section
    )

    top_claims = _rows(
        top_claims_section
    )

    minimum_claim_amount = _safe_float(
        quantiles.get(
            "minimum_amount",
            0,
        )
    )

    median_claim_amount = _safe_float(
        quantiles.get(
            "median",
            0,
        )
    )

    p75_claim_amount = _safe_float(
        quantiles.get(
            "p75",
            0,
        )
    )

    p90_claim_amount = _safe_float(
        quantiles.get(
            "p90",
            0,
        )
    )

    p95_claim_amount = _safe_float(
        quantiles.get(
            "p95",
            0,
        )
    )

    p99_claim_amount = _safe_float(
        quantiles.get(
            "p99",
            0,
        )
    )

    # --------------------------------------------------------
    # TOP-10 EXPOSURE
    # --------------------------------------------------------

    top_5_exposure = 0.0
    top_10_exposure = 0.0

    if top_claims:

        amounts = sorted(
            [
                _safe_float(
                    row.get(
                        "claim_amount",
                        0,
                    )
                )
                for row in top_claims
            ],
            reverse=True,
        )

        top_5_exposure = sum(
            amounts[:5]
        )

        top_10_exposure = sum(
            amounts[:10]
        )

    top_10_exposure_percentage = _safe_percentage(
        top_10_exposure,
        total_claim_amount,
    )

    # --------------------------------------------------------
    # POSITIVE EXPOSURE
    # --------------------------------------------------------

    positive_amount_analysis = investigation.get(
        "positive_amount_analysis",
        {},
    )

    positive_analysis_row = _first_row(
        positive_amount_analysis
    )

    positive_claim_exposure = _safe_float(
        positive_analysis_row.get(
            "positive_claim_exposure",
            total_claim_amount,
        )
    )

    if positive_claim_exposure == 0:
        positive_claim_exposure = total_claim_amount

    # --------------------------------------------------------
    # NATIONAL COMPARISON
    # --------------------------------------------------------

    if national_average > 0:

        average_ratio_to_national = (
            county_average / national_average
        )

        percentage_above_national_average = (
            (county_average - national_average)
            / national_average
        ) * 100.0

    else:

        average_ratio_to_national = 0.0
        percentage_above_national_average = 0.0

    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    if risk_level == "CRITICAL":
        priority = "IMMEDIATE"

    elif risk_level == "HIGH":
        priority = "HIGH"

    elif risk_level == "MEDIUM":
        priority = "MONITOR"

    else:
        priority = "ROUTINE"

    # --------------------------------------------------------
    # PRIMARY DRIVER
    # --------------------------------------------------------

    primary_driver = "UNKNOWN"

    signal_types = {
        signal.get("type")
        for signal in signals
        if isinstance(signal, dict)
    }

    if "AMOUNT_ANOMALY" in signal_types:
        primary_driver = "MONETARY ANOMALY"

    if "POSITIVE_AMOUNT_CONCENTRATION" in signal_types:
        primary_driver = "MONETARY CONCENTRATION"

    if "ZERO_INFLATION" in signal_types:
        primary_driver = "ZERO-INFLATED CLAIM DISTRIBUTION"

    if "INSURER_CONCENTRATION" in signal_types:
        if primary_driver == "UNKNOWN":
            primary_driver = "INSURER CONCENTRATION"

    # --------------------------------------------------------
    # EXECUTIVE VERDICT
    # --------------------------------------------------------

    verdict_parts = []

    if risk_level == "CRITICAL":

        verdict_parts.append(
            f"{county} requires immediate investigation."
        )

    elif risk_level == "HIGH":

        verdict_parts.append(
            f"{county} requires priority investigation."
        )

    elif risk_level == "MEDIUM":

        verdict_parts.append(
            f"{county} should be placed under enhanced monitoring."
        )

    else:

        verdict_parts.append(
            f"{county} does not currently show a critical anomaly."
        )

    # Driver

    if primary_driver != "UNKNOWN":

        verdict_parts.append(
            f"The primary driver is {primary_driver.lower()}."
        )

    # Claim share

    verdict_parts.append(
        f"The county represents "
        f"{claim_share_percentage:.2f}% of all claims."
    )

    # National comparison

    if national_average > 0:

        verdict_parts.append(
            f"Its average claim amount is "
            f"{average_ratio_to_national:.1f}x "
            f"the national average."
        )

    else:

        verdict_parts.append(
            "A reliable national-average comparison "
            "could not be calculated."
        )

    # Zero / positive distribution

    verdict_parts.append(
        f"Only {positive_percentage:.1f}% of claims "
        f"carry positive amounts."
    )

    # Exposure concentration

    if positive_claim_exposure > 0:

        verdict_parts.append(
            f"The top 10 positive-value claims account "
            f"for approximately "
            f"{top_10_exposure_percentage:.1f}% "
            f"of positive monetary exposure."
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    recommendations = []

    if top_claims:

        recommendations.append(
            "Review the highest-value claims individually."
        )

    if zero_percentage >= 80:

        recommendations.append(
            "Investigate why most claims have zero monetary value."
        )

    if positive_percentage <= 10:

        recommendations.append(
            "Identify facilities responsible for positive-value claims."
        )

    if "INSURER_CONCENTRATION" in signal_types:

        recommendations.append(
            "Investigate insurer concentration and determine "
            "whether it reflects legitimate business structure."
        )

    recommendations.extend(
        [
            "Check whether repeated high-value claims share "
            "the same facility, provider, insurer, or submission pattern.",
            "Compare anomalous claims against surrounding counties "
            "and comparable facilities.",
            "Validate whether the observed pattern is a legitimate "
            "business process or a data-quality issue.",
        ]
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    return {

        "county": county,

        "risk_level": risk_level,

        "investigation_score": investigation_score,

        "priority": priority,

        "primary_driver": primary_driver,

        # Core metrics

        "claim_count": claim_count,

        "total_claim_amount": _round(
            total_claim_amount
        ),

        "county_claim_share_percentage": _round(
            claim_share_percentage
        ),

        "national_average": _round(
            national_average
        ),

        "county_average": _round(
            county_average
        ),

        "average_ratio_to_national": _round(
            average_ratio_to_national
        ),

        "percentage_above_national_average": _round(
            percentage_above_national_average
        ),

        # Distribution

        "minimum_claim_amount": _round(
            minimum_claim_amount
        ),

        "median_claim_amount": _round(
            median_claim_amount
        ),

        "p75_claim_amount": _round(
            p75_claim_amount
        ),

        "p90_claim_amount": _round(
            p90_claim_amount
        ),

        "p95_claim_amount": _round(
            p95_claim_amount
        ),

        "p99_claim_amount": _round(
            p99_claim_amount
        ),

        "maximum_claim_amount": _round(
            maximum_claim_amount
        ),

        # Positive / zero claims

        "positive_claims": positive_claims,

        "positive_claim_percentage": _round(
            positive_percentage
        ),

        "zero_claims": zero_claims,

        "zero_claim_percentage": _round(
            zero_percentage
        ),

        # Exposure

        "positive_claim_exposure": _round(
            positive_claim_exposure
        ),

        "top_5_claim_exposure": _round(
            top_5_exposure
        ),

        "top_10_claim_exposure": _round(
            top_10_exposure
        ),

        "top_10_exposure_percentage": _round(
            top_10_exposure_percentage
        ),

        # Raw analytical material

        "signals": signals,

        "conclusions": conclusions,

        "recommendations": recommendations,

        "distribution": distribution,

        "insurers": investigation.get(
            "insurers",
            {},
        ),

        "facilities": investigation.get(
            "facilities",
            {},
        ),

        "claim_types": investigation.get(
            "claim_types",
            {},
        ),

        "monthly": investigation.get(
            "monthly",
            {},
        ),

        "positive_amount_analysis": positive_amount_analysis,

        # Executive summary

        "executive_verdict": " ".join(
            verdict_parts
        ),
    }
