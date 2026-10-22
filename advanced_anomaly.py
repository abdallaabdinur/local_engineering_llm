"""Multivariate, relationship, and count-dispersion anomaly detectors.

These detectors prioritize records and relationships for investigation. They do
not estimate fraud probability or make a fraud determination.
"""

from __future__ import annotations

import math
from statistics import mean, variance
from typing import Any

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from tools import run_select


CLAIM_FEATURE_SQL = """
WITH sampled AS (
    SELECT
        claim_id,
        county,
        client,
        facility,
        facility_name,
        insurer,
        claim_type,
        facility_level,
        creation,
        claim_amount,
        is_cancelled
    FROM claims
    WHERE claim_id IS NOT NULL
      AND creation IS NOT NULL
      AND claim_amount IS NOT NULL
    ORDER BY creation DESC
    LIMIT %s
),
source AS (
    SELECT
        sampled.*,
        COUNT(*) OVER (
            PARTITION BY client, creation::date
        ) AS client_day_claims,
        COUNT(*) OVER (
            PARTITION BY facility, creation::date
        ) AS facility_day_claims,
        COUNT(*) OVER (
            PARTITION BY claim_amount
        ) AS repeated_amount_claims,
        COUNT(*) OVER (
            PARTITION BY client, facility
        ) AS client_facility_claims
    FROM sampled
)
SELECT * FROM source
"""


RELATIONSHIP_SQL = """
SELECT
    client,
    facility,
    MAX(facility_name) AS facility_name,
    COUNT(*) AS interaction_count,
    COUNT(DISTINCT creation::date) AS active_days,
    COUNT(DISTINCT claim_type) AS claim_type_count,
    COALESCE(SUM(claim_amount), 0) AS total_amount,
    COALESCE(AVG(claim_amount), 0) AS average_amount
FROM claims
WHERE client IS NOT NULL
  AND facility IS NOT NULL
  AND creation IS NOT NULL
GROUP BY client, facility
HAVING COUNT(*) >= %s
ORDER BY interaction_count DESC
LIMIT %s
"""


MONTH_COUNT_SQL = """
SELECT
    DATE_TRUNC('month', creation)::date AS claim_month,
    COUNT(*) AS claim_count
FROM claims
WHERE creation IS NOT NULL
GROUP BY DATE_TRUNC('month', creation)::date
ORDER BY claim_month
"""


def _rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(result, dict) or result.get("error"):
        return []
    columns = result.get("columns") or []
    return [dict(zip(columns, row)) for row in result.get("rows") or []]


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _percentile_ranks(values: list[float]) -> list[float]:
    """Convert arbitrary anomaly scores into stable 0-1 percentile ranks."""

    if not values:
        return []

    ordered = sorted(values)
    denominator = max(len(values) - 1, 1)
    return [
        ordered.index(value) / denominator
        for value in values
    ]


def _feature_rows(limit: int) -> tuple[list[dict[str, Any]], str | None]:
    result = run_select(
        CLAIM_FEATURE_SQL.replace("%s", str(min(max(int(limit), 100), 10000)), 1),
        max_rows=limit,
    )
    if isinstance(result, dict) and result.get("error"):
        return [], str(result["error"])
    return _rows(result), None


def _claim_features(row: dict[str, Any]) -> list[float]:
    amount = max(_float(row.get("claim_amount")), 0.0)
    return [
        math.log1p(amount),
        1.0 if amount == 0 else 0.0,
        1.0 if row.get("is_cancelled") is True else 0.0,
        math.log1p(_float(row.get("client_day_claims"))),
        math.log1p(_float(row.get("facility_day_claims"))),
        math.log1p(_float(row.get("repeated_amount_claims"))),
        math.log1p(_float(row.get("client_facility_claims"))),
    ]


def analyze_claim_ml_anomalies(
    limit: int = 5000,
    contamination: float = 0.02,
    result_limit: int = 100,
) -> dict[str, Any]:
    """Score claim-level feature vectors with Isolation Forest and LOF."""

    limit = min(max(int(limit), 100), 10000)
    result_limit = min(max(int(result_limit), 1), 500)
    contamination = min(max(float(contamination), 0.001), 0.2)
    rows, error = _feature_rows(limit)
    if error:
        return {
            "algorithm": "isolation_forest_lof",
            "error": error,
            "rows": [],
        }
    if len(rows) < 10:
        return {
            "algorithm": "isolation_forest_lof",
            "rows": [],
            "summary": "At least 10 claim rows are required.",
        }

    matrix = [_claim_features(row) for row in rows]
    forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(matrix)
    forest_scores = [-float(score) for score in forest.decision_function(matrix)]

    neighbors = min(35, len(rows) - 1)
    lof = LocalOutlierFactor(
        n_neighbors=neighbors,
        contamination=contamination,
    )
    lof.fit_predict(matrix)
    lof_scores = [-float(score) for score in lof.negative_outlier_factor_]
    lof_percentiles = _percentile_ranks(lof_scores)

    forest_cutoff = sorted(forest_scores, reverse=True)[
        max(0, int(len(rows) * contamination) - 1)
    ]
    lof_cutoff = sorted(lof_scores, reverse=True)[
        max(0, int(len(rows) * contamination) - 1)
    ]

    scored = []
    for row, forest_score, lof_score, lof_percentile in zip(
        rows,
        forest_scores,
        lof_scores,
        lof_percentiles,
    ):
        forest_flag = forest_score >= forest_cutoff
        lof_flag = lof_score >= lof_cutoff
        if not forest_flag and not lof_flag:
            continue
        scored.append({
            "claim_id": row.get("claim_id"),
            "county": row.get("county"),
            "client": row.get("client"),
            "facility": row.get("facility"),
            "facility_name": row.get("facility_name"),
            "insurer": row.get("insurer"),
            "claim_type": row.get("claim_type"),
            "creation": row.get("creation"),
            "claim_amount": row.get("claim_amount"),
            "is_cancelled": row.get("is_cancelled"),
            "isolation_forest_score": round(forest_score, 5),
            "lof_score": round(lof_percentile, 5),
            "isolation_forest_flag": forest_flag,
            "lof_flag": lof_flag,
            "agreement": "BOTH" if forest_flag and lof_flag else "ONE_MODEL",
            "feature_evidence": {
                "client_day_claims": row.get("client_day_claims"),
                "facility_day_claims": row.get("facility_day_claims"),
                "repeated_amount_claims": row.get("repeated_amount_claims"),
                "client_facility_claims": row.get("client_facility_claims"),
            },
        })

    scored.sort(
        key=lambda row: (
            row["agreement"] == "BOTH",
            row["isolation_forest_score"] + row["lof_score"],
        ),
        reverse=True,
    )
    return {
        "algorithm": "isolation_forest_lof",
        "sample_size": len(rows),
        "contamination": contamination,
        "rows": scored[:result_limit],
        "summary": (
            "Isolation Forest detects globally unusual feature combinations; "
            "LOF detects claims unlike their local peer neighborhood. "
            "Agreement between both models is the strongest signal."
        ),
    }


def analyze_relationship_anomalies(
    minimum_interactions: int = 5,
    limit: int = 100,
) -> dict[str, Any]:
    """Rank unusually dense client-facility relationships."""

    minimum_interactions = max(int(minimum_interactions), 2)
    limit = min(max(int(limit), 1), 500)
    result = run_select(
        RELATIONSHIP_SQL.replace("%s", str(minimum_interactions), 1)
        .replace("%s", str(limit), 1),
        max_rows=limit,
    )
    if isinstance(result, dict) and result.get("error"):
        return {
            "algorithm": "client_facility_relationship_density",
            "error": result["error"],
            "rows": [],
        }

    relationships = _rows(result)
    counts = [_float(row.get("interaction_count")) for row in relationships]
    baseline = mean(counts) if counts else 0.0
    spread = math.sqrt(variance(counts)) if len(counts) > 1 else 0.0
    for row in relationships:
        count = _float(row.get("interaction_count"))
        row["density_score"] = round(
            (count - baseline) / spread if spread else 0.0,
            3,
        )
        row["investigation_signal"] = (
            "HIGH" if row["density_score"] >= 3 else
            "MEDIUM" if row["density_score"] >= 2 else
            "NORMAL"
        )

    relationships.sort(key=lambda row: row["density_score"], reverse=True)
    return {
        "algorithm": "client_facility_relationship_density",
        "minimum_interactions": minimum_interactions,
        "rows": relationships,
        "summary": (
            "Ranks client-facility edges with unusually high interaction "
            "counts; use with amount, date, bundle, and linked-claim review."
        ),
    }


def analyze_count_dispersion() -> dict[str, Any]:
    """Recommend Poisson or Negative Binomial monitoring from count dispersion."""

    result = run_select(MONTH_COUNT_SQL, max_rows=1000)
    if isinstance(result, dict) and result.get("error"):
        return {
            "algorithm": "poisson_vs_negative_binomial",
            "error": result["error"],
        }
    rows = _rows(result)
    counts = [_float(row.get("claim_count")) for row in rows]
    if len(counts) < 2:
        return {
            "algorithm": "poisson_vs_negative_binomial",
            "recommendation": "INSUFFICIENT_HISTORY",
            "months": len(counts),
        }

    average = mean(counts)
    dispersion = variance(counts) / average if average else 0.0
    model = "POISSON" if dispersion <= 1.5 else "NEGATIVE_BINOMIAL"
    return {
        "algorithm": "poisson_vs_negative_binomial",
        "months": len(counts),
        "mean_monthly_claims": round(average, 2),
        "variance_to_mean_ratio": round(dispersion, 3),
        "recommended_model": model,
        "explanation": (
            "Poisson is suitable when variance is close to the mean; "
            "Negative Binomial is safer for overdispersed counts."
        ),
    }
