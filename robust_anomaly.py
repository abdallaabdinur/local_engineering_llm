"""Robust county-level anomaly detection for claims data.

The detector separates claim volume, zero/positive behavior, and positive
claim amounts. It produces investigation-priority signals, not fraud scores.
"""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from tools import run_select


MIN_COUNTY_CLAIMS = 30
MAD_SCALE = 1.4826


FEATURE_SQL = """
SELECT
    county,
    COUNT(*) AS claim_count,
    COUNT(*) FILTER (WHERE claim_amount > 0) AS positive_claims,
    COUNT(*) FILTER (WHERE claim_amount = 0) AS zero_claims,
    COALESCE(SUM(claim_amount), 0) AS total_claim_amount,
    COALESCE(SUM(claim_amount) FILTER (WHERE claim_amount > 0), 0)
        AS positive_total_amount,
    COALESCE(AVG(claim_amount) FILTER (WHERE claim_amount > 0), 0)
        AS positive_average_amount,
    PERCENTILE_CONT(0.50) WITHIN GROUP (
        ORDER BY claim_amount
    ) FILTER (WHERE claim_amount > 0) AS positive_median_amount,
    PERCENTILE_CONT(0.95) WITHIN GROUP (
        ORDER BY claim_amount
    ) FILTER (WHERE claim_amount > 0) AS positive_p95_amount
FROM claims
WHERE county IS NOT NULL
GROUP BY county
HAVING COUNT(*) >= %s
ORDER BY county
"""


PEER_CLAIM_SQL = """
WITH peer_stats AS (
    SELECT
        COALESCE(claim_type, '__UNKNOWN__') AS claim_type_key,
        COALESCE(facility_level, '__UNKNOWN__') AS facility_level_key,
        COUNT(*) AS peer_claim_count,
        PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY claim_amount
        ) AS peer_median_amount,
        PERCENTILE_CONT(0.95) WITHIN GROUP (
            ORDER BY claim_amount
        ) AS peer_p95_amount
    FROM claims
    WHERE claim_amount > 0
    GROUP BY
        COALESCE(claim_type, '__UNKNOWN__'),
        COALESCE(facility_level, '__UNKNOWN__')
    HAVING COUNT(*) >= %s
)
SELECT
    c.claim_id,
    c.county,
    c.client,
    c.facility,
    c.facility_name,
    c.insurer,
    c.claim_type,
    c.facility_level,
    c.creation,
    c.claim_amount,
    p.peer_claim_count,
    p.peer_median_amount,
    p.peer_p95_amount,
    c.claim_amount / NULLIF(p.peer_median_amount, 0)
        AS peer_median_multiple
FROM claims AS c
JOIN peer_stats AS p
    ON COALESCE(c.claim_type, '__UNKNOWN__') = p.claim_type_key
   AND COALESCE(c.facility_level, '__UNKNOWN__') = p.facility_level_key
WHERE c.claim_amount > 0
  AND c.claim_amount > p.peer_p95_amount
ORDER BY peer_median_multiple DESC
LIMIT %s
"""


MONTHLY_FEATURE_SQL = """
SELECT
    county,
    DATE_TRUNC('month', creation)::date AS claim_month,
    COUNT(*) AS claim_count,
    COUNT(*) FILTER (WHERE claim_amount > 0) AS positive_claims,
    COALESCE(SUM(claim_amount), 0) AS total_claim_amount,
    COALESCE(
        SUM(claim_amount) FILTER (WHERE claim_amount > 0),
        0
    ) AS positive_total_amount
FROM claims
WHERE county IS NOT NULL
  AND creation IS NOT NULL
GROUP BY county, DATE_TRUNC('month', creation)::date
ORDER BY county, claim_month
"""


def _as_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("error"):
        return []

    columns = result.get("columns") or []
    return [dict(zip(columns, row)) for row in result.get("rows") or []]


def _mad(values: list[float], center: float) -> float:
    if not values:
        return 0.0
    return float(median([abs(value - center) for value in values]))


def _robust_z(value: float, center: float, mad: float) -> float:
    if mad <= 0:
        return 0.0
    return abs(value - center) / (MAD_SCALE * mad)


def _temporal_robust_z(value: float, center: float, mad: float) -> float:
    """Use a stable scale when a short history has near-zero MAD."""

    scale = max(
        MAD_SCALE * mad,
        abs(center) * 0.10,
        1.0,
    )
    return abs(value - center) / scale


def _severity(score: float) -> str:
    if score >= 8:
        return "CRITICAL"
    if score >= 5:
        return "HIGH"
    if score >= 3:
        return "MEDIUM"
    if score >= 2:
        return "LOW"
    return "NORMAL"


def analyze_county_anomalies(
    minimum_claims: int = MIN_COUNTY_CLAIMS,
    limit: int = 25,
) -> dict[str, Any]:
    """Rank counties using robust two-part monetary features."""

    minimum_claims = max(int(minimum_claims), 1)
    limit = min(max(int(limit), 1), 100)
    result = run_select(
        FEATURE_SQL.replace("%s", str(minimum_claims)),
        max_rows=1000,
    )

    if isinstance(result, dict) and result.get("error"):
        return {
            "algorithm": "two_part_mad",
            "error": result["error"],
            "rows": [],
        }

    source_rows = _rows(result)
    if not source_rows:
        return {
            "algorithm": "two_part_mad",
            "minimum_claims": minimum_claims,
            "rows": [],
            "summary": "No counties met the minimum claim threshold.",
        }

    features = []
    for row in source_rows:
        claim_count = _as_float(row.get("claim_count"))
        positive_claims = _as_float(row.get("positive_claims"))
        positive_rate = positive_claims / claim_count if claim_count else 0.0
        features.append({
            **row,
            "claim_count": claim_count,
            "positive_rate": positive_rate,
            "log_claim_count": math.log1p(claim_count),
            "log_positive_median": math.log1p(
                _as_float(row.get("positive_median_amount"))
            ),
        })

    baselines = {}
    for feature_name in (
        "positive_rate",
        "log_claim_count",
        "log_positive_median",
    ):
        values = [row[feature_name] for row in features]
        center = float(median(values))
        baselines[feature_name] = {
            "median": center,
            "mad": _mad(values, center),
        }

    ranked = []
    for row in features:
        positive_rate_score = _robust_z(
            row["positive_rate"],
            baselines["positive_rate"]["median"],
            baselines["positive_rate"]["mad"],
        )
        volume_score = _robust_z(
            row["log_claim_count"],
            baselines["log_claim_count"]["median"],
            baselines["log_claim_count"]["mad"],
        )
        amount_score = _robust_z(
            row["log_positive_median"],
            baselines["log_positive_median"]["median"],
            baselines["log_positive_median"]["mad"],
        )

        score = round(
            positive_rate_score + amount_score + volume_score,
            3,
        )
        severity = _severity(score)
        ranked.append({
            "county": row.get("county"),
            "claim_count": int(row["claim_count"]),
            "positive_claims": int(_as_float(row.get("positive_claims"))),
            "zero_claims": int(_as_float(row.get("zero_claims"))),
            "positive_rate": round(row["positive_rate"] * 100, 2),
            "positive_median_amount": row.get("positive_median_amount"),
            "positive_p95_amount": row.get("positive_p95_amount"),
            "total_claim_amount": row.get("total_claim_amount"),
            "positive_total_amount": row.get("positive_total_amount"),
            "anomaly_score": score,
            "severity": severity,
            "components": {
                "positive_rate": round(positive_rate_score, 3),
                "positive_amount": round(amount_score, 3),
                "claim_volume": round(volume_score, 3),
            },
        })

    ranked.sort(key=lambda row: row["anomaly_score"], reverse=True)
    return {
        "algorithm": "two_part_mad",
        "minimum_claims": minimum_claims,
        "county_count": len(ranked),
        "baselines": baselines,
        "rows": ranked[:limit],
        "summary": (
            "Scores combine robust deviations in positive-claim rate, "
            "positive median amount, and log claim volume."
        ),
    }


def analyze_peer_claims(
    minimum_peer_claims: int = 30,
    limit: int = 50,
) -> dict[str, Any]:
    """Find claims unusually high for their type/facility-level peer group."""

    minimum_peer_claims = max(int(minimum_peer_claims), 3)
    limit = min(max(int(limit), 1), 200)
    result = run_select(
        PEER_CLAIM_SQL.replace("%s", str(minimum_peer_claims), 1)
        .replace("%s", str(limit), 1),
        max_rows=limit,
    )

    if isinstance(result, dict) and result.get("error"):
        return {
            "algorithm": "peer_median_p95",
            "error": result["error"],
            "rows": [],
        }

    return {
        "algorithm": "peer_median_p95",
        "minimum_peer_claims": minimum_peer_claims,
        "rows": _rows(result),
        "summary": (
            "Claims are compared with peers sharing claim type and "
            "facility level; only claims above the peer P95 are returned."
        ),
    }


def analyze_temporal_anomalies(
    minimum_history: int = 3,
    limit: int = 50,
) -> dict[str, Any]:
    """Find county-month spikes against each county's prior-month baseline."""

    minimum_history = max(int(minimum_history), 2)
    limit = min(max(int(limit), 1), 200)
    result = run_select(MONTHLY_FEATURE_SQL, max_rows=10000)

    if isinstance(result, dict) and result.get("error"):
        return {
            "algorithm": "rolling_median_mad",
            "error": result["error"],
            "rows": [],
        }

    monthly_rows = _rows(result)
    by_county: dict[str, list[dict[str, Any]]] = {}
    for row in monthly_rows:
        by_county.setdefault(str(row.get("county")), []).append(row)

    anomalies = []
    for county, rows in by_county.items():
        rows.sort(key=lambda row: str(row.get("claim_month")))
        for index, row in enumerate(rows):
            history = rows[max(0, index - 6):index]
            if len(history) < minimum_history:
                continue

            count_values = [
                _as_float(item.get("claim_count"))
                for item in history
            ]
            amount_values = [
                math.log1p(_as_float(item.get("positive_total_amount")))
                for item in history
            ]
            count_center = float(median(count_values))
            amount_center = float(median(amount_values))
            count_score = min(
                _temporal_robust_z(
                    _as_float(row.get("claim_count")),
                    count_center,
                    _mad(count_values, count_center),
                ),
                10.0,
            )
            amount_score = min(
                _temporal_robust_z(
                    math.log1p(_as_float(row.get("positive_total_amount"))),
                    amount_center,
                    _mad(amount_values, amount_center),
                ),
                10.0,
            )
            score = round(count_score + amount_score, 3)
            if score < 2:
                continue

            anomalies.append({
                "county": county,
                "claim_month": row.get("claim_month"),
                "claim_count": int(_as_float(row.get("claim_count"))),
                "positive_total_amount": row.get("positive_total_amount"),
                "anomaly_score": score,
                "severity": _severity(score),
                "components": {
                    "claim_volume": round(count_score, 3),
                    "positive_exposure": round(amount_score, 3),
                },
            })

    anomalies.sort(key=lambda row: row["anomaly_score"], reverse=True)
    return {
        "algorithm": "rolling_median_mad",
        "minimum_history": minimum_history,
        "rows": anomalies[:limit],
        "summary": (
            "County-month activity is compared with up to six prior months "
            "using robust volume and positive-exposure baselines."
        ),
    }
