from tools import run_select


def _rows(result):
    """
    Normalize the database result into a list of dictionaries.
    """
    if not result:
        return []

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def investigate_claims(county, limit=25):
    """
    Rank individual claims within a county by investigation priority.

    This is an investigation-priority score,
    NOT a fraud probability.
    """

    county = county.upper().strip()

    # ---------------------------------------------------------
    # 1. Claim-level evidence
    # ---------------------------------------------------------

    claims_sql = """
        WITH county_stats AS (
            SELECT
                COUNT(*) AS county_claim_count,
                COALESCE(
                    SUM(claim_amount),
                    0
                ) AS county_total_amount,
                COALESCE(
                    AVG(NULLIF(claim_amount, 0)),
                    0
                ) AS county_positive_average
            FROM claims
            WHERE UPPER(county) = %s
        ),

        facility_stats AS (
            SELECT
                facility,
                SUM(
                    COALESCE(claim_amount, 0)
                ) AS facility_total_amount
            FROM claims
            WHERE UPPER(county) = %s
            GROUP BY facility
        ),

        insurer_stats AS (
            SELECT
                insurer,
                SUM(
                    COALESCE(claim_amount, 0)
                ) AS insurer_total_amount
            FROM claims
            WHERE UPPER(county) = %s
            GROUP BY insurer
        ),

        amount_frequency AS (
            SELECT
                claim_amount,
                COUNT(*) AS amount_frequency
            FROM claims
            WHERE UPPER(county) = %s
              AND claim_amount > 0
            GROUP BY claim_amount
        ),

        daily_stats AS (
            SELECT
                DATE(creation) AS claim_date,
                COUNT(*) AS daily_claim_count,
                SUM(
                    COALESCE(claim_amount, 0)
                ) AS daily_total_amount
            FROM claims
            WHERE UPPER(county) = %s
            GROUP BY DATE(creation)
        ),

        daily_baseline AS (
            SELECT
                AVG(
                    daily_total_amount
                ) AS avg_daily_amount,

                STDDEV_POP(
                    daily_total_amount
                ) AS std_daily_amount

            FROM daily_stats
        )

        SELECT
            c.claim_id,
            c.claim_amount,
            c.county,
            c.facility,
            c.facility_name,
            c.insurer,
            c.claim_type,
            c.creation,

            COALESCE(
                af.amount_frequency,
                1
            ) AS amount_frequency,

            COALESCE(
                fs.facility_total_amount,
                0
            ) AS facility_total_amount,

            COALESCE(
                100.0
                * fs.facility_total_amount
                / NULLIF(
                    cs.county_total_amount,
                    0
                ),
                0
            ) AS facility_exposure_percentage,

            COALESCE(
                is2.insurer_total_amount,
                0
            ) AS insurer_total_amount,

            COALESCE(
                100.0
                * is2.insurer_total_amount
                / NULLIF(
                    cs.county_total_amount,
                    0
                ),
                0
            ) AS insurer_exposure_percentage,

            COALESCE(
                ds.daily_claim_count,
                0
            ) AS daily_claim_count,

            COALESCE(
                ds.daily_total_amount,
                0
            ) AS daily_total_amount,

            CASE
                WHEN db.std_daily_amount IS NULL
                  OR db.std_daily_amount = 0
                THEN 0

                ELSE
                    (
                        ds.daily_total_amount
                        - db.avg_daily_amount
                    )
                    / db.std_daily_amount
            END AS daily_amount_zscore

        FROM claims c

        CROSS JOIN county_stats cs

        LEFT JOIN facility_stats fs
            ON fs.facility = c.facility

        LEFT JOIN insurer_stats is2
            ON is2.insurer = c.insurer

        LEFT JOIN amount_frequency af
            ON af.claim_amount = c.claim_amount

        LEFT JOIN daily_stats ds
            ON ds.claim_date = DATE(c.creation)

        CROSS JOIN daily_baseline db

        WHERE UPPER(c.county) = %s
          AND c.claim_amount > 0

        ORDER BY c.claim_amount DESC

        LIMIT %s
    """

    # ---------------------------------------------------------
    # 2. Safely insert the county and limit.
    #
    # run_select() accepts a SELECT query, but does not
    # provide parameter binding. Therefore we escape the
    # county before inserting it into the SQL.
    # ---------------------------------------------------------

    safe_county = county.replace(
        "'",
        "''"
    )

    claims_sql = claims_sql.replace(
        "%s",
        f"'{safe_county}'",
        6
    )

    claims_sql = claims_sql.replace(
        "%s",
        str(int(limit)),
        1
    )

    # ---------------------------------------------------------
    # 3. Execute query
    # ---------------------------------------------------------

    result = run_select(
        claims_sql
    )

    # If the database layer returned an error,
    # return an empty investigation result rather than
    # crashing during scoring.
    if result and result.get("error"):
        return {
            "county": county,
            "claim_count": 0,
            "claims": [],
            "error": result["error"],
        }

    claims = _rows(result)

    # ---------------------------------------------------------
    # 4. Score each claim
    # ---------------------------------------------------------

    ranked_claims = []

    for claim in claims:

        # =====================================================
        # CLAIM-SPECIFIC ANOMALY SCORE
        # =====================================================

        anomaly_score = 0

        reasons = []

        amount = float(
            claim.get("claim_amount") or 0
        )

        frequency = int(
            claim.get("amount_frequency") or 1
        )

        daily_zscore = float(
            claim.get("daily_amount_zscore") or 0
        )

        # -----------------------------------------------------
        # Amount severity
        # -----------------------------------------------------

        if amount >= 100000:

            anomaly_score += 4

            reasons.append(
                "Extremely high claim amount"
            )

        elif amount >= 50000:

            anomaly_score += 3

            reasons.append(
                "Very high claim amount"
            )

        elif amount >= 20000:

            anomaly_score += 2

            reasons.append(
                "High claim amount"
            )

        elif amount >= 10000:

            anomaly_score += 1

            reasons.append(
                "Elevated claim amount"
            )

        # -----------------------------------------------------
        # Repeated exact amount
        # -----------------------------------------------------

        if frequency >= 20:

            anomaly_score += 3

            reasons.append(
                f"Exact amount repeated "
                f"{frequency} times"
            )

        elif frequency >= 5:

            anomaly_score += 2

            reasons.append(
                f"Exact amount repeated "
                f"{frequency} times"
            )

        # -----------------------------------------------------
        # Daily exposure anomaly
        # -----------------------------------------------------

        if daily_zscore >= 5:

            anomaly_score += 3

            reasons.append(
                "Submitted during an extreme "
                "daily exposure spike"
            )

        elif daily_zscore >= 3:

            anomaly_score += 2

            reasons.append(
                "Submitted during a major "
                "daily exposure spike"
            )

        elif daily_zscore >= 2:

            anomaly_score += 1

            reasons.append(
                "Submitted during an unusual "
                "daily exposure spike"
            )

        # =====================================================
        # CONTEXTUAL CONCENTRATION
        # =====================================================

        context_score = 0

        context_flags = []

        facility_share = float(
            claim.get(
                "facility_exposure_percentage"
            ) or 0
        )

        insurer_share = float(
            claim.get(
                "insurer_exposure_percentage"
            ) or 0
        )

        # -----------------------------------------------------
        # Facility concentration
        #
        # This is contextual evidence, not direct
        # claim-level anomaly evidence.
        # -----------------------------------------------------

        if facility_share >= 90:

            context_score += 1

            context_flags.append(
                "Dominant exposure facility"
            )

        elif facility_share >= 50:

            context_score += 1

            context_flags.append(
                "Highly concentrated facility"
            )

        # -----------------------------------------------------
        # Insurer concentration
        #
        # This is also contextual evidence.
        # -----------------------------------------------------

        if insurer_share >= 95:

            context_score += 1

            context_flags.append(
                "Dominant insurer"
            )

        elif insurer_share >= 75:

            context_score += 1

            context_flags.append(
                "Highly concentrated insurer"
            )

        # =====================================================
        # FINAL INVESTIGATION SCORE
        # =====================================================

        investigation_score = (
            anomaly_score
            + context_score
        )

        # -----------------------------------------------------
        # Risk classification
        # -----------------------------------------------------

        if investigation_score >= 8:

            risk_level = "CRITICAL"

        elif investigation_score >= 5:

            risk_level = "HIGH"

        elif investigation_score >= 3:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # -----------------------------------------------------
        # Store scoring components
        # -----------------------------------------------------

        claim["anomaly_score"] = anomaly_score

        claim["context_score"] = context_score

        claim["investigation_score"] = (
            investigation_score
        )

        claim["risk_level"] = risk_level

        claim["risk_reasons"] = reasons

        claim["context_flags"] = context_flags

        ranked_claims.append(
            claim
        )

    # ---------------------------------------------------------
    # 5. Rank claims
    #
    # Priority:
    #   1. Total investigation score
    #   2. Claim-specific anomaly score
    #   3. Claim amount
    # ---------------------------------------------------------

    ranked_claims.sort(
        key=lambda x: (
            x["investigation_score"],
            x["anomaly_score"],
            float(
                x.get("claim_amount") or 0
            ),
        ),
        reverse=True,
    )

    # ---------------------------------------------------------
    # 6. Return investigation result
    # ---------------------------------------------------------

    return {
        "county": county,
        "claim_count": len(
            ranked_claims
        ),
        "claims": ranked_claims,
    }