from tools import run_select


# ============================================================
# ANOMALY INVESTIGATOR
# ============================================================

def execute(sql):
    """
    Execute a controlled analytical query.
    """
    try:
        return run_select(sql, max_rows=100)
    except Exception as error:
        return {
            "error": str(error),
            "columns": [],
            "rows": [],
            "row_count": 0,
        }


# ============================================================
# COUNTY PROFILE
# ============================================================

def investigate_county(county):
    """
    Build a detailed profile of a suspicious county.
    """

    county = county.upper().replace("'", "''")

    results = {}

    # --------------------------------------------------------
    # Overall county statistics
    # --------------------------------------------------------

    results["overview"] = execute(
        f"""
        SELECT
            COUNT(*) AS claim_count,
            SUM(claim_amount) AS total_claim_amount,
            AVG(claim_amount) AS average_claim_amount,
            MIN(claim_amount) AS minimum_claim_amount,
            MAX(claim_amount) AS maximum_claim_amount
        FROM claims
        WHERE county = '{county}'
        """
    )

    # --------------------------------------------------------
    # Claim type distribution
    # --------------------------------------------------------

    results["claim_types"] = execute(
        f"""
        SELECT
            claim_type,
            COUNT(*) AS claim_count,
            SUM(claim_amount) AS total_claim_amount,
            AVG(claim_amount) AS average_claim_amount
        FROM claims
        WHERE county = '{county}'
          AND claim_type IS NOT NULL
        GROUP BY claim_type
        ORDER BY claim_count DESC
        LIMIT 20
        """
    )

    # --------------------------------------------------------
    # Facility concentration
    # --------------------------------------------------------

    results["facilities"] = execute(
        f"""
        SELECT
            facility,
            COUNT(*) AS claim_count,
            SUM(claim_amount) AS total_claim_amount,
            AVG(claim_amount) AS average_claim_amount
        FROM claims
        WHERE county = '{county}'
          AND facility IS NOT NULL
        GROUP BY facility
        ORDER BY total_claim_amount DESC
        LIMIT 20
        """
    )

    # --------------------------------------------------------
    # Insurer distribution
    # --------------------------------------------------------

    results["insurers"] = execute(
        f"""
        SELECT
            insurer,
            COUNT(*) AS claim_count,
            SUM(claim_amount) AS total_claim_amount,
            AVG(claim_amount) AS average_claim_amount
        FROM claims
        WHERE county = '{county}'
          AND insurer IS NOT NULL
        GROUP BY insurer
        ORDER BY claim_count DESC
        LIMIT 20
        """
    )

    # --------------------------------------------------------
    # Monthly behaviour
    # --------------------------------------------------------

    results["monthly"] = execute(
        f"""
        SELECT
            DATE_TRUNC(
                'month',
                creation
            ) AS claim_month,

            COUNT(*) AS claim_count,

            SUM(claim_amount)
                AS total_claim_amount,

            AVG(claim_amount)
                AS average_claim_amount

        FROM claims

        WHERE county = '{county}'
          AND creation IS NOT NULL

        GROUP BY DATE_TRUNC(
            'month',
            creation
        )

        ORDER BY claim_month
        """
    )

    # --------------------------------------------------------
    # Largest claims
    # --------------------------------------------------------

    results["extreme_claims"] = execute(
        f"""
        SELECT
            claim_id,
            facility,
            insurer,
            claim_type,
            claim_amount,
            creation

        FROM claims

        WHERE county = '{county}'
          AND claim_amount IS NOT NULL

        ORDER BY claim_amount DESC

        LIMIT 20
        """
    )

    # --------------------------------------------------------
    # Cancellation behaviour
    # --------------------------------------------------------

    results["cancellations"] = execute(
        f"""
        SELECT

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
                NULLIF(COUNT(*), 0),
                2
            ) AS cancellation_percentage

        FROM claims

        WHERE county = '{county}'
        """
    )

    return results


# ============================================================
# COMPARISON AGAINST NATIONAL DATA
# ============================================================

def compare_county_to_national(county):
    """
    Compare a suspicious county against the overall database.
    """

    county = county.upper().replace("'", "''")

    sql = f"""
    WITH county_stats AS (

        SELECT

            AVG(claim_amount)
                AS county_average,

            COUNT(*)
                AS county_claims,

            SUM(claim_amount)
                AS county_total

        FROM claims

        WHERE county = '{county}'

    ),

    national_stats AS (

        SELECT

            AVG(claim_amount)
                AS national_average,

            COUNT(*)
                AS national_claims,

            SUM(claim_amount)
                AS national_total

        FROM claims

    )

    SELECT

        county_average,

        national_average,

        county_average
            - national_average
            AS average_difference,

        ROUND(
            100.0 *
            (
                county_average
                - national_average
            )
            /
            NULLIF(
                national_average,
                0
            ),
            2
        )
        AS percentage_difference,

        county_claims,

        national_claims,

        ROUND(
            100.0 *
            county_claims
            /
            NULLIF(
                national_claims,
                0
            ),
            2
        )
        AS share_of_claims_percentage

    FROM county_stats
    CROSS JOIN national_stats
    """

    return execute(sql)
# ============================================================
# CLAIM AMOUNT DISTRIBUTION
# ============================================================

def analyze_claim_distribution(county):
    """
    Determine whether a county's high average claim amount
    is caused by a few extreme claims or a broad distribution.
    """

    county = county.upper().replace("'", "''")

    results = {}

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    results["quantiles"] = execute(
        f"""
        SELECT

            COUNT(*) AS claim_count,

            MIN(claim_amount) AS minimum_amount,

            PERCENTILE_CONT(0.25)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS p25,

            PERCENTILE_CONT(0.50)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS median,

            PERCENTILE_CONT(0.75)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS p75,

            PERCENTILE_CONT(0.90)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS p90,

            PERCENTILE_CONT(0.95)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS p95,

            PERCENTILE_CONT(0.99)
                WITHIN GROUP (
                    ORDER BY claim_amount
                ) AS p99,

            MAX(claim_amount)
                AS maximum_amount,

            AVG(claim_amount)
                AS average_amount

        FROM claims

        WHERE county = '{county}'
          AND claim_amount IS NOT NULL
        """
    )

    # --------------------------------------------------------
    # Amount buckets
    # --------------------------------------------------------

    results["buckets"] = execute(
        f"""
        SELECT

            CASE

                WHEN claim_amount = 0
                    THEN '0'

                WHEN claim_amount > 0
                     AND claim_amount <= 100
                    THEN '1-100'

                WHEN claim_amount <= 500
                    THEN '101-500'

                WHEN claim_amount <= 1000
                    THEN '501-1,000'

                WHEN claim_amount <= 5000
                    THEN '1,001-5,000'

                WHEN claim_amount <= 10000
                    THEN '5,001-10,000'

                WHEN claim_amount <= 50000
                    THEN '10,001-50,000'

                WHEN claim_amount <= 100000
                    THEN '50,001-100,000'

                ELSE '100,000+'

            END AS amount_bucket,

            COUNT(*) AS claim_count,

            SUM(claim_amount)
                AS total_claim_amount,

            AVG(claim_amount)
                AS average_claim_amount

        FROM claims

        WHERE county = '{county}'
          AND claim_amount IS NOT NULL

        GROUP BY amount_bucket

        ORDER BY
            MIN(claim_amount)
        """
    )

    # --------------------------------------------------------
    # Top extreme claims
    # --------------------------------------------------------

    results["top_claims"] = execute(
        f"""
        SELECT

            claim_id,
            claim_amount,
            facility,
            facility_name,
            insurer,
            claim_type,
            creation

        FROM claims

        WHERE county = '{county}'
          AND claim_amount IS NOT NULL

        ORDER BY claim_amount DESC

        LIMIT 25
        """
    )

    return results

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    county = "KITUI"

    print("=" * 70)
    print("COUNTY ANOMALY INVESTIGATION")
    print("=" * 70)

    print()
    print("County:", county)

    print()
    print("OVERVIEW")
    print(
        investigate_county(
            county
        )["overview"]
    )

    print()
    print("CLAIM TYPES")
    print(
        investigate_county(
            county
        )["claim_types"]
    )

    print()
    print("INSURERS")
    print(
        investigate_county(
            county
        )["insurers"]
    )

    print()
    print("NATIONAL COMPARISON")

    print(
        compare_county_to_national(
            county
        )
    )
