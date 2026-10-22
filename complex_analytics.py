import math
from decimal import Decimal
from statistics import mean, median, stdev, StatisticsError

from tools import run_select


# ============================================================
# CONFIGURATION
# ============================================================

ANALYSIS_MAX_ROWS = 1000

# Minimum number of observations required before using
# z-score based anomaly detection.
MIN_STATISTICAL_SAMPLE = 3

# Standard z-score threshold.
OUTLIER_Z_THRESHOLD = 2.0

# Number of extreme claims to inspect.
EXTREME_CLAIM_LIMIT = 25

# Number of facilities to inspect.
FACILITY_LIMIT = 100


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    """
    Convert PostgreSQL numeric values into Python floats.

    Returns None when conversion is impossible.
    """

    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def safe_int(value):
    """
    Convert a value to integer safely.
    """

    if value is None:
        return None

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def clean_rows(result):
    """
    Convert a database result into a list of dictionaries.

    Example:

        columns = ["county", "claim_count"]

        rows = [
            ("NAIROBI", 1000),
            ("MOMBASA", 500),
        ]

    becomes:

        [
            {
                "county": "NAIROBI",
                "claim_count": 1000
            },
            ...
        ]
    """

    if not result:
        return []

    if not isinstance(result, dict):
        return []

    if "error" in result:
        return []

    columns = result.get(
        "columns",
        [],
    )

    rows = result.get(
        "rows",
        [],
    )

    if not columns or not rows:
        return []

    cleaned = []

    for row in rows:

        try:

            cleaned.append(
                dict(
                    zip(
                        columns,
                        row,
                    )
                )
            )

        except Exception:
            continue

    return cleaned


def execute_analysis(
    name,
    sql,
):
    """
    Execute one analytical query safely.

    Every analytical operation returns a consistent structure.

    This prevents one failed analysis from destroying the
    entire complex-analysis pipeline.
    """

    try:

        result = run_select(
            sql,
            max_rows=ANALYSIS_MAX_ROWS,
        )

        if not isinstance(
            result,
            dict,
        ):

            return {
                "name": name,
                "success": False,
                "sql": sql,
                "result": None,
                "rows": [],
                "error": (
                    "Database returned an "
                    "invalid result."
                ),
            }

        if "error" in result:

            return {
                "name": name,
                "success": False,
                "sql": sql,
                "result": result,
                "rows": [],
                "error": str(
                    result.get(
                        "error",
                        "Unknown database error.",
                    )
                ),
            }

        return {
            "name": name,
            "success": True,
            "sql": sql,
            "result": result,
            "rows": clean_rows(result),
            "error": None,
        }

    except Exception as error:

        return {
            "name": name,
            "success": False,
            "sql": sql,
            "result": None,
            "rows": [],
            "error": str(error),
        }


# ============================================================
# CORE ANALYTICS
# ============================================================

def analyze_claim_volume():
    """
    Identify counties with unusually high or low claim volume.
    """

    sql = """
        SELECT
            county,
            COUNT(*) AS claim_count
        FROM claims
        WHERE county IS NOT NULL
        GROUP BY county
        ORDER BY claim_count DESC
    """

    return execute_analysis(
        "claim_volume_by_county",
        sql,
    )


def analyze_claim_amounts():
    """
    Analyze claim amount distribution by county.
    """

    sql = """
        SELECT
            county,

            COUNT(*) AS claim_count,

            SUM(claim_amount) AS total_claim_amount,

            AVG(claim_amount) AS average_claim_amount,

            MIN(claim_amount) AS minimum_claim_amount,

            MAX(claim_amount) AS maximum_claim_amount

        FROM claims

        WHERE county IS NOT NULL
          AND claim_amount IS NOT NULL

        GROUP BY county

        ORDER BY average_claim_amount DESC
    """

    return execute_analysis(
        "claim_amount_by_county",
        sql,
    )


def analyze_monthly_trends():
    """
    Detect changes, trends and spikes in claim activity.
    """

    sql = """
        SELECT
            DATE_TRUNC(
                'month',
                creation
            ) AS claim_month,

            COUNT(*) AS claim_count,

            SUM(claim_amount) AS total_claim_amount,

            AVG(claim_amount) AS average_claim_amount

        FROM claims

        WHERE creation IS NOT NULL

        GROUP BY
            DATE_TRUNC(
                'month',
                creation
            )

        ORDER BY claim_month ASC
    """

    return execute_analysis(
        "monthly_claim_trends",
        sql,
    )


def analyze_claim_types():
    """
    Identify dominant claim types.
    """

    sql = """
        SELECT
            claim_type,

            COUNT(*) AS claim_count,

            SUM(claim_amount) AS total_claim_amount,

            AVG(claim_amount) AS average_claim_amount

        FROM claims

        WHERE claim_type IS NOT NULL

        GROUP BY claim_type

        ORDER BY claim_count DESC
    """

    return execute_analysis(
        "claim_type_distribution",
        sql,
    )


def analyze_insurers():
    """
    Analyze insurer concentration.
    """

    sql = """
        SELECT
            insurer,

            COUNT(*) AS claim_count,

            SUM(claim_amount) AS total_claim_amount,

            AVG(claim_amount) AS average_claim_amount

        FROM claims

        WHERE insurer IS NOT NULL

        GROUP BY insurer

        ORDER BY claim_count DESC
    """

    return execute_analysis(
        "insurer_distribution",
        sql,
    )


def analyze_facilities():
    """
    Identify facilities with large claim volumes and
    financial exposure.
    """

    sql = f"""
        SELECT
            facility,

            COUNT(*) AS claim_count,

            SUM(claim_amount) AS total_claim_amount,

            AVG(claim_amount) AS average_claim_amount,

            MAX(claim_amount) AS maximum_claim_amount

        FROM claims

        WHERE facility IS NOT NULL

        GROUP BY facility

        ORDER BY total_claim_amount DESC

        LIMIT {FACILITY_LIMIT}
    """

    return execute_analysis(
        "facility_exposure",
        sql,
    )


def analyze_cancellations():
    """
    Analyze cancellation behavior by county.
    """

    sql = """
        SELECT
            county,

            COUNT(*) AS total_claims,

            COUNT(*) FILTER (
                WHERE is_cancelled = TRUE
            ) AS cancelled_claims,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE is_cancelled = TRUE
                )
                /
                NULLIF(
                    COUNT(*),
                    0
                ),
                2
            ) AS cancellation_percentage

        FROM claims

        WHERE county IS NOT NULL

        GROUP BY county

        ORDER BY cancellation_percentage DESC
    """

    return execute_analysis(
        "cancellation_by_county",
        sql,
    )


# ============================================================
# EXTREME CLAIM VALUES
# ============================================================

def analyze_extreme_claims():
    """
    Find the largest claims in the database.

    This is deliberately deterministic rather than allowing
    the LLM to define "extreme".
    """

    sql = f"""
        SELECT
            claim_id,
            county,
            facility,
            insurer,
            claim_type,
            claim_amount,
            creation

        FROM claims

        WHERE claim_amount IS NOT NULL

        ORDER BY claim_amount DESC

        LIMIT {EXTREME_CLAIM_LIMIT}
    """

    return execute_analysis(
        "extreme_claim_amounts",
        sql,
    )


# ============================================================
# DATA QUALITY
# ============================================================

def analyze_data_quality():
    """
    Measure important missing-value patterns.
    """

    sql = """
        SELECT

            COUNT(*) AS total_claims,

            COUNT(*) FILTER (
                WHERE county IS NULL
            ) AS missing_county,

            COUNT(*) FILTER (
                WHERE facility IS NULL
            ) AS missing_facility,

            COUNT(*) FILTER (
                WHERE insurer IS NULL
            ) AS missing_insurer,

            COUNT(*) FILTER (
                WHERE claim_amount IS NULL
            ) AS missing_claim_amount,

            COUNT(*) FILTER (
                WHERE claim_type IS NULL
            ) AS missing_claim_type,

            COUNT(*) FILTER (
                WHERE creation IS NULL
            ) AS missing_creation,

            COUNT(*) FILTER (
                WHERE diagnoses IS NULL
            ) AS missing_diagnoses,

            COUNT(*) FILTER (
                WHERE interventions IS NULL
            ) AS missing_interventions

        FROM claims
    """

    return execute_analysis(
        "data_quality",
        sql,
    )


# ============================================================
# CANCELLATION OUTLIERS
# ============================================================

def detect_cancellation_outliers(
    cancellation_analysis,
):
    """
    Detect counties with unusually high or low cancellation
    rates.

    Uses z-score analysis.
    """

    rows = cancellation_analysis.get(
        "rows",
        [],
    )

    values = []

    for row in rows:

        value = safe_float(
            row.get(
                "cancellation_percentage"
            )
        )

        if value is not None:
            values.append(value)

    if len(values) < MIN_STATISTICAL_SAMPLE:
        return []

    average = mean(values)

    try:
        deviation = stdev(values)

    except StatisticsError:
        return []

    if deviation == 0:
        return []

    findings = []

    for row in rows:

        value = safe_float(
            row.get(
                "cancellation_percentage"
            )
        )

        if value is None:
            continue

        z_score = (
            value - average
        ) / deviation

        if abs(z_score) >= OUTLIER_Z_THRESHOLD:

            findings.append(
                {
                    "county": row.get(
                        "county"
                    ),
                    "cancellation_percentage": round(
                        value,
                        2,
                    ),
                    "z_score": round(
                        z_score,
                        2,
                    ),
                    "direction": (
                        "high"
                        if z_score > 0
                        else "low"
                    ),
                }
            )

    findings.sort(
        key=lambda item: abs(
            item["z_score"]
        ),
        reverse=True,
    )

    return findings


# ============================================================
# CLAIM VOLUME OUTLIERS
# ============================================================

def detect_volume_outliers(
    volume_analysis,
):
    """
    Identify counties whose claim volume is substantially
    above or below the distribution mean.
    """

    rows = volume_analysis.get(
        "rows",
        [],
    )

    values = []

    for row in rows:

        value = safe_float(
            row.get(
                "claim_count"
            )
        )

        if value is not None:
            values.append(value)

    if len(values) < MIN_STATISTICAL_SAMPLE:
        return []

    average = mean(values)

    try:
        deviation = stdev(values)

    except StatisticsError:
        return []

    if deviation == 0:
        return []

    findings = []

    for row in rows:

        value = safe_float(
            row.get(
                "claim_count"
            )
        )

        if value is None:
            continue

        z_score = (
            value - average
        ) / deviation

        if abs(z_score) >= OUTLIER_Z_THRESHOLD:

            findings.append(
                {
                    "county": row.get(
                        "county"
                    ),
                    "claim_count": value,
                    "z_score": round(
                        z_score,
                        2,
                    ),
                    "direction": (
                        "high"
                        if z_score > 0
                        else "low"
                    ),
                }
            )

    findings.sort(
        key=lambda item: abs(
            item["z_score"]
        ),
        reverse=True,
    )

    return findings


# ============================================================
# CLAIM AMOUNT OUTLIERS
# ============================================================

def detect_amount_outliers(
    amount_analysis,
):
    """
    Identify counties with unusually high or low average
    claim amounts.
    """

    rows = amount_analysis.get(
        "rows",
        [],
    )

    values = []

    for row in rows:

        value = safe_float(
            row.get(
                "average_claim_amount"
            )
        )

        if value is not None:
            values.append(value)

    if len(values) < MIN_STATISTICAL_SAMPLE:
        return []

    average = mean(values)

    try:
        deviation = stdev(values)

    except StatisticsError:
        return []

    if deviation == 0:
        return []

    findings = []

    for row in rows:

        value = safe_float(
            row.get(
                "average_claim_amount"
            )
        )

        if value is None:
            continue

        z_score = (
            value - average
        ) / deviation

        if abs(z_score) >= OUTLIER_Z_THRESHOLD:

            findings.append(
                {
                    "county": row.get(
                        "county"
                    ),
                    "average_claim_amount": round(
                        value,
                        2,
                    ),
                    "z_score": round(
                        z_score,
                        2,
                    ),
                    "direction": (
                        "high"
                        if z_score > 0
                        else "low"
                    ),
                }
            )

    findings.sort(
        key=lambda item: abs(
            item["z_score"]
        ),
        reverse=True,
    )

    return findings


# ============================================================
# MONTHLY SPIKES
# ============================================================

def detect_monthly_spikes(
    monthly_analysis,
):
    """
    Detect months with unusually high claim volume.
    """

    rows = monthly_analysis.get(
        "rows",
        [],
    )

    values = []

    for row in rows:

        value = safe_float(
            row.get(
                "claim_count"
            )
        )

        if value is not None:
            values.append(value)

    if len(values) < MIN_STATISTICAL_SAMPLE:
        return []

    average = mean(values)

    try:
        deviation = stdev(values)

    except StatisticsError:
        return []

    if deviation == 0:
        return []

    findings = []

    for row in rows:

        value = safe_float(
            row.get(
                "claim_count"
            )
        )

        if value is None:
            continue

        z_score = (
            value - average
        ) / deviation

        # Only unusually high months are considered spikes.
        if z_score >= OUTLIER_Z_THRESHOLD:

            findings.append(
                {
                    "month": str(
                        row.get(
                            "claim_month"
                        )
                    ),
                    "claim_count": value,
                    "z_score": round(
                        z_score,
                        2,
                    ),
                }
            )

    findings.sort(
        key=lambda item: item["z_score"],
        reverse=True,
    )

    return findings


# ============================================================
# EXTREME CLAIM SUMMARY
# ============================================================

def summarize_extreme_claims(
    extreme_analysis,
):
    """
    Produce a compact summary of the extreme claims.
    """

    rows = extreme_analysis.get(
        "rows",
        [],
    )

    if not rows:
        return {
            "count": 0,
            "maximum_claim_amount": None,
            "minimum_claim_amount": None,
            "average_extreme_claim_amount": None,
        }

    amounts = []

    for row in rows:

        value = safe_float(
            row.get(
                "claim_amount"
            )
        )

        if value is not None:
            amounts.append(value)

    if not amounts:

        return {
            "count": 0,
            "maximum_claim_amount": None,
            "minimum_claim_amount": None,
            "average_extreme_claim_amount": None,
        }

    return {
        "count": len(amounts),

        "maximum_claim_amount": round(
            max(amounts),
            2,
        ),

        "minimum_claim_amount": round(
            min(amounts),
            2,
        ),

        "average_extreme_claim_amount": round(
            mean(amounts),
            2,
        ),
    }


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def summarize_data_quality(
    quality_analysis,
):
    """
    Convert raw data-quality counts into percentages.
    """

    rows = quality_analysis.get(
        "rows",
        [],
    )

    if not rows:
        return {}

    row = rows[0]

    total = safe_float(
        row.get(
            "total_claims"
        )
    )

    if not total or total <= 0:
        return {
            key: row.get(key)
            for key in row
        }

    result = {}

    for key, value in row.items():

        result[key] = value

        if (
            key.startswith(
                "missing_"
            )
        ):

            count = safe_float(
                value
            )

            if count is not None:

                result[
                    key + "_percentage"
                ] = round(
                    100.0 * count / total,
                    2,
                )

    return result


# ============================================================
# TOP ENTITY HELPERS
# ============================================================

def top_rows(
    analysis,
    metric,
    limit=10,
):
    """
    Return top rows from an analytical result by metric.
    """

    rows = analysis.get(
        "rows",
        [],
    )

    valid = []

    for row in rows:

        value = safe_float(
            row.get(metric)
        )

        if value is None:
            continue

        valid.append(
            (
                value,
                row,
            )
        )

    valid.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        row
        for _, row in valid[:limit]
    ]


# ============================================================
# COMPLEX ANALYSIS ORCHESTRATOR
# ============================================================

def run_complex_analysis(
    question,
):
    """
    Execute a controlled multi-dimensional investigation.

    The LLM does not decide what data exists.

    This engine collects evidence from the database first,
    then derives statistical findings from that evidence.
    """

    analyses = {}

    # ========================================================
    # 1. GEOGRAPHIC DISTRIBUTION
    # ========================================================

    analyses["volume"] = (
        analyze_claim_volume()
    )

    # ========================================================
    # 2. CLAIM AMOUNT DISTRIBUTION
    # ========================================================

    analyses["amounts"] = (
        analyze_claim_amounts()
    )

    # ========================================================
    # 3. TIME TRENDS
    # ========================================================

    analyses["monthly"] = (
        analyze_monthly_trends()
    )

    # ========================================================
    # 4. CLAIM TYPE CONCENTRATION
    # ========================================================

    analyses["types"] = (
        analyze_claim_types()
    )

    # ========================================================
    # 5. INSURER CONCENTRATION
    # ========================================================

    analyses["insurers"] = (
        analyze_insurers()
    )

    # ========================================================
    # 6. FACILITY EXPOSURE
    # ========================================================

    analyses["facilities"] = (
        analyze_facilities()
    )

    # ========================================================
    # 7. CANCELLATION BEHAVIOR
    # ========================================================

    analyses["cancellations"] = (
        analyze_cancellations()
    )

    # ========================================================
    # 8. EXTREME CLAIMS
    # ========================================================

    analyses["extreme_claims"] = (
        analyze_extreme_claims()
    )

    # ========================================================
    # 9. DATA QUALITY
    # ========================================================

    analyses["data_quality"] = (
        analyze_data_quality()
    )

    # ========================================================
    # STATISTICAL FINDINGS
    # ========================================================

    findings = {
        "volume_outliers": (
            detect_volume_outliers(
                analyses["volume"]
            )
        ),

        "amount_outliers": (
            detect_amount_outliers(
                analyses["amounts"]
            )
        ),

        "monthly_spikes": (
            detect_monthly_spikes(
                analyses["monthly"]
            )
        ),

        "cancellation_outliers": (
            detect_cancellation_outliers(
                analyses["cancellations"]
            )
        ),
    }

    # ========================================================
    # DERIVED SUMMARIES
    # ========================================================

    summaries = {
        "extreme_claims": (
            summarize_extreme_claims(
                analyses["extreme_claims"]
            )
        ),

        "data_quality": (
            summarize_data_quality(
                analyses["data_quality"]
            )
        ),

        "top_counties_by_volume": (
            top_rows(
                analyses["volume"],
                "claim_count",
                limit=10,
            )
        ),

        "top_counties_by_total_amount": (
            top_rows(
                analyses["amounts"],
                "total_claim_amount",
                limit=10,
            )
        ),

        "top_insurers_by_volume": (
            top_rows(
                analyses["insurers"],
                "claim_count",
                limit=10,
            )
        ),

        "top_claim_types_by_volume": (
            top_rows(
                analyses["types"],
                "claim_count",
                limit=10,
            )
        ),

        "top_facilities_by_exposure": (
            top_rows(
                analyses["facilities"],
                "total_claim_amount",
                limit=10,
            )
        ),
    }

    # ========================================================
    # PIPELINE STATUS
    # ========================================================

    successful_analyses = sum(
        1
        for analysis in analyses.values()
        if analysis.get("success")
    )

    failed_analyses = (
        len(analyses)
        - successful_analyses
    )

    # ========================================================
    # RETURN COMPLETE EVIDENCE PACKAGE
    # ========================================================

    return {
        "question": question,

        "analyses": analyses,

        "findings": findings,

        "summaries": summaries,

        "pipeline": {
            "total_analyses": len(
                analyses
            ),

            "successful_analyses": (
                successful_analyses
            ),

            "failed_analyses": (
                failed_analyses
            ),

            "complete": (
                failed_analyses == 0
            ),
        },
    }


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "Tell me something complicated "
        "about unusual claim patterns."
    )

    print("=" * 70)
    print(
        "CLAIMS INTELLIGENCE — COMPLEX ANALYSIS"
    )
    print("=" * 70)

    print()

    result = run_complex_analysis(
        question
    )

    pipeline = result.get(
        "pipeline",
        {},
    )

    print(
        "Analyses:",
        pipeline.get(
            "total_analyses",
            0,
        ),
    )

    print(
        "Successful:",
        pipeline.get(
            "successful_analyses",
            0,
        ),
    )

    print(
        "Failed:",
        pipeline.get(
            "failed_analyses",
            0,
        ),
    )

    print()

    print(
        "Volume outliers:",
        len(
            result[
                "findings"
            ][
                "volume_outliers"
            ]
        ),
    )

    print(
        "Amount outliers:",
        len(
            result[
                "findings"
            ][
                "amount_outliers"
            ]
        ),
    )

    print(
        "Cancellation outliers:",
        len(
            result[
                "findings"
            ][
                "cancellation_outliers"
            ]
        ),
    )

    print(
        "Monthly spikes:",
        len(
            result[
                "findings"
            ][
                "monthly_spikes"
            ]
        ),
    )

    print()

    print(
        "Top counties by volume:"
    )

    for row in result[
        "summaries"
    ][
        "top_counties_by_volume"
    ]:

        print(
            row
        )

    print()

    print(
        "Top counties by total amount:"
    )

    for row in result[
        "summaries"
    ][
        "top_counties_by_total_amount"
    ]:

        print(
            row
        )

    print()

    print(
        "Cancellation outliers:"
    )

    for finding in result[
        "findings"
    ][
        "cancellation_outliers"
    ]:

        print(
            finding
        )

    print()

    print(
        "Monthly spikes:"
    )

    for finding in result[
        "findings"
    ][
        "monthly_spikes"
    ]:

        print(
            finding
        )

    print()

    print(
        "Extreme claim summary:"
    )

    print(
        result[
            "summaries"
        ][
            "extreme_claims"
        ]
    )

    print()

    print(
        "Data quality:"
    )

    print(
        result[
            "summaries"
        ][
            "data_quality"
        ]
    )