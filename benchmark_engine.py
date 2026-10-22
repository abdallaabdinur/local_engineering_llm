from tools import run_select


def _rows(result):
    """
    Normalize database results into a list of dictionaries.
    """
    if not result:
        return []

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def _safe_float(value):
    """
    Safely convert a database value to float.
    """
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _median(values):
    """
    Calculate the median of a list of numeric values.
    """
    if not values:
        return 0.0

    values = sorted(values)

    n = len(values)
    middle = n // 2

    if n % 2 == 1:
        return values[middle]

    return (
        values[middle - 1]
        + values[middle]
    ) / 2.0


def benchmark_county(county):
    """
    Build national and claim-type benchmarks for a county.

    The benchmark population consists of ALL positive-value
    claims in the database.

    PostgreSQL calculates percentile ranks before the county
    filter is applied. Therefore the county claims are compared
    against the complete national population.

    This function provides statistical context only.

    It does NOT determine fraud probability or fraud status.
    """

    # =========================================================
    # 1. NORMALIZE COUNTY
    # =========================================================

    county = (
        county
        .upper()
        .strip()
    )

    if not county:
        return {
            "county": county,
            "error": "County cannot be empty.",
        }

    # Escape single quotes for SQL safety.
    safe_county = county.replace(
        "'",
        "''",
    )

    # =========================================================
    # 2. NATIONAL POSITIVE-CLAIM STATISTICS
    # =========================================================

    national_stats_sql = """
        SELECT
            COUNT(*) AS positive_claim_count,

            MIN(claim_amount) AS minimum,

            PERCENTILE_CONT(0.5)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS median,

            AVG(claim_amount) AS average,

            MAX(claim_amount) AS maximum

        FROM claims

        WHERE claim_amount > 0
    """

    national_stats_result = run_select(
        national_stats_sql,
        max_rows=1,
    )

    if (
        national_stats_result
        and national_stats_result.get("error")
    ):
        return {
            "county": county,
            "error": national_stats_result["error"],
        }

    national_stats_rows = _rows(
        national_stats_result
    )

    if not national_stats_rows:
        return {
            "county": county,
            "error": "No positive claims found.",
        }

    national_stats = (
        national_stats_rows[0]
    )

    national_positive_count = int(
        national_stats.get(
            "positive_claim_count"
        )
        or 0
    )

    national_median = _safe_float(
        national_stats.get("median")
    )

    national_average = _safe_float(
        national_stats.get("average")
    )

    national_maximum = _safe_float(
        national_stats.get("maximum")
    )

    national_minimum = _safe_float(
        national_stats.get("minimum")
    )

    # =========================================================
    # 3. COUNTY BENCHMARK QUERY
    # =========================================================
    #
    # IMPORTANT:
    #
    # The percentile calculations happen inside ranked_claims
    # BEFORE filtering to the requested county.
    #
    # This means:
    #
    # national_percentile
    #     = percentile against ALL positive claims
    #
    # claim_type_percentile
    #     = percentile against ALL positive claims of same type
    #
    # Only after these values are calculated do we select KITUI.
    #
    # =========================================================

    county_sql = f"""
        WITH ranked_claims AS (

            SELECT
                claim_id,
                claim_amount,
                claim_type,
                county,

                /*
                 * National empirical percentile.
                 *
                 * CUME_DIST returns a value between 0 and 1.
                 */
                100.0
                * CUME_DIST() OVER (
                    ORDER BY claim_amount
                ) AS national_percentile,

                /*
                 * Peer percentile.
                 *
                 * Claims are compared only against claims
                 * with the same claim_type.
                 */
                100.0
                * CUME_DIST() OVER (
                    PARTITION BY claim_type
                    ORDER BY claim_amount
                ) AS claim_type_percentile,

                /*
                 * Full national positive-claim population.
                 */
                COUNT(*) OVER ()
                    AS national_positive_claim_count,

                /*
                 * Full positive-claim population within
                 * the same claim type.
                 */
                COUNT(*) OVER (
                    PARTITION BY claim_type
                )
                    AS claim_type_positive_claim_count

            FROM claims

            WHERE claim_amount > 0
        )

        SELECT
            claim_id,
            claim_amount,
            claim_type,
            county,

            national_percentile,
            claim_type_percentile,

            national_positive_claim_count,
            claim_type_positive_claim_count

        FROM ranked_claims

        WHERE UPPER(county) = '{safe_county}'

        ORDER BY
            claim_amount DESC,
            claim_id
    """

    county_result = run_select(
        county_sql,
        max_rows=1000,
    )

    if (
        county_result
        and county_result.get("error")
    ):
        return {
            "county": county,
            "error": county_result["error"],
        }

    county_rows = _rows(
        county_result
    )

    # =========================================================
    # 4. BUILD BENCHMARK RECORDS
    # =========================================================

    benchmarks = []

    for row in county_rows:

        claim_id = row.get(
            "claim_id"
        )

        amount = _safe_float(
            row.get("claim_amount")
        )

        claim_type = (
            row.get("claim_type")
            or "UNKNOWN"
        )

        national_percentile = _safe_float(
            row.get(
                "national_percentile"
            )
        )

        peer_percentile = _safe_float(
            row.get(
                "claim_type_percentile"
            )
        )

        national_count = int(
            row.get(
                "national_positive_claim_count"
            )
            or national_positive_count
        )

        peer_count = int(
            row.get(
                "claim_type_positive_claim_count"
            )
            or 0
        )

        benchmarks.append(
            {
                "claim_id": claim_id,

                "claim_amount": amount,

                "claim_type": claim_type,

                "national_percentile": round(
                    national_percentile,
                    2,
                ),

                "claim_type_percentile": round(
                    peer_percentile,
                    2,
                ),

                "national_positive_claim_count": (
                    national_count
                ),

                "claim_type_positive_claim_count": (
                    peer_count
                ),
            }
        )

    # =========================================================
    # 5. COUNTY POSITIVE CLAIM DISTRIBUTION
    # =========================================================

    county_amounts = []

    for row in county_rows:

        amount = _safe_float(
            row.get("claim_amount")
        )

        if amount > 0:
            county_amounts.append(
                amount
            )

    county_amounts.sort()

    county_positive_count = len(
        county_amounts
    )

    county_median = _median(
        county_amounts
    )

    county_average = (
        sum(county_amounts)
        / county_positive_count
        if county_positive_count > 0
        else 0.0
    )

    county_maximum = (
        max(county_amounts)
        if county_amounts
        else 0.0
    )

    # =========================================================
    # 6. EXTREME CLAIMS
    # =========================================================
    #
    # A claim is considered statistically extreme if it is
    # >= 99th percentile nationally OR within its peer type.
    #
    # This is NOT a fraud determination.
    #
    # =========================================================

    extreme_claims = [
        item
        for item in benchmarks
        if (
            item["national_percentile"] >= 99.0
            or
            item["claim_type_percentile"] >= 99.0
        )
    ]

    extreme_claims.sort(
        key=lambda item: (
            max(
                item[
                    "national_percentile"
                ],
                item[
                    "claim_type_percentile"
                ],
            ),
            item["claim_amount"],
        ),
        reverse=True,
    )

    # =========================================================
    # 7. DATA INTEGRITY CHECK
    # =========================================================

    benchmark_claim_count = len(
        benchmarks
    )

    unique_claim_ids = {
        item["claim_id"]
        for item in benchmarks
        if item.get("claim_id")
    }

    unique_benchmark_claim_count = len(
        unique_claim_ids
    )

    data_integrity_ok = (
        benchmark_claim_count
        == unique_benchmark_claim_count
    )

    # =========================================================
    # 8. RATIO ANALYSIS
    # =========================================================

    county_to_national_median_ratio = (
        county_median
        / national_median
        if national_median > 0
        else 0.0
    )

    county_to_national_average_ratio = (
        county_average
        / national_average
        if national_average > 0
        else 0.0
    )

    # =========================================================
    # 9. PERCENTILE COUNTS
    # =========================================================

    claims_at_or_above_99th_national = sum(
        1
        for item in benchmarks
        if (
            item["national_percentile"]
            >= 99.0
        )
    )

    claims_at_or_above_99th_peer = sum(
        1
        for item in benchmarks
        if (
            item["claim_type_percentile"]
            >= 99.0
        )
    )

    # =========================================================
    # 10. RETURN RESULT
    # =========================================================

    return {
        "county": county,

        "national": {
            "positive_claim_count": (
                national_positive_count
            ),

            "minimum": round(
                national_minimum,
                2,
            ),

            "median": round(
                national_median,
                2,
            ),

            "average": round(
                national_average,
                2,
            ),

            "maximum": round(
                national_maximum,
                2,
            ),
        },

        "county": {
            "positive_claim_count": (
                county_positive_count
            ),

            "median": round(
                county_median,
                2,
            ),

            "average": round(
                county_average,
                2,
            ),

            "maximum": round(
                county_maximum,
                2,
            ),
        },

        "claims": benchmarks,

        "extreme_claims": extreme_claims,

        "summary": {
            "county_to_national_median_ratio": round(
                county_to_national_median_ratio,
                2,
            ),

            "county_to_national_average_ratio": round(
                county_to_national_average_ratio,
                2,
            ),

            "claims_at_or_above_99th_national_percentile": (
                claims_at_or_above_99th_national
            ),

            "claims_at_or_above_99th_peer_percentile": (
                claims_at_or_above_99th_peer
            ),

            "benchmark_claim_count": (
                benchmark_claim_count
            ),

            "unique_benchmark_claim_count": (
                unique_benchmark_claim_count
            ),

            "data_integrity_ok": (
                data_integrity_ok
            ),
        },
    }