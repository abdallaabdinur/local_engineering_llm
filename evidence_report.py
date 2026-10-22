from evidence_engine import investigate_evidence


def _rows(result):
    """
    Convert a database-style result into a list of dictionaries.
    Expected structure:
        {
            "columns": [...],
            "rows": [...]
        }
    """
    if not result:
        return []

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    return [
        dict(zip(columns, row))
        for row in rows
    ]


def build_evidence_report(county):
    evidence = investigate_evidence(county)

    summary = evidence.get("summary", {})
    raw_evidence = evidence.get("evidence", {})

    facilities = raw_evidence.get("facilities", [])
    insurers = raw_evidence.get("insurers", [])
    high_value = raw_evidence.get("high_value_claims", {})
    repeated_amounts = raw_evidence.get("repeated_amounts", {})
    daily_activity = raw_evidence.get("daily_activity", {})

    facility_rows = (
        facilities
        if isinstance(facilities, list)
        else _rows(facilities)
    )

    insurer_rows = (
        insurers
        if isinstance(insurers, list)
        else _rows(insurers)
    )

    high_value_rows = _rows(high_value)
    repeated_amount_rows = _rows(repeated_amounts)
    daily_rows = _rows(daily_activity)

    # ---------------------------------------------------------
    # CORE METRICS
    # ---------------------------------------------------------

    zero_percentage = 0.0
    positive_percentage = 0.0

    for finding in summary.get("findings", []):
        if "zero monetary value" in finding:
            try:
                zero_percentage = float(
                    finding.split("%")[0]
                )
            except (ValueError, IndexError):
                pass

        if "positive monetary value" in finding:
            try:
                positive_percentage = float(
                    finding.split("%")[0]
                )
            except (ValueError, IndexError):
                pass

    # ---------------------------------------------------------
    # FACILITY CONCENTRATION
    # ---------------------------------------------------------

    dominant_facility = None

    if facility_rows:
        dominant_facility = max(
            facility_rows,
            key=lambda x: float(
                x.get("exposure_percentage", 0) or 0
            )
        )

    # ---------------------------------------------------------
    # INSURER CONCENTRATION
    # ---------------------------------------------------------

    dominant_insurer = None

    if insurer_rows:
        dominant_insurer = max(
            insurer_rows,
            key=lambda x: float(
                x.get("exposure_percentage", 0) or 0
            )
        )

    # ---------------------------------------------------------
    # REPEATED AMOUNTS
    # ---------------------------------------------------------

    repeated_positive_amounts = [
        row
        for row in repeated_amount_rows
        if float(row.get("claim_amount", 0) or 0) > 0
        and int(row.get("claim_count", 0) or 0) >= 2
    ]

    strongest_repeated_amount = None

    if repeated_positive_amounts:
        strongest_repeated_amount = max(
            repeated_positive_amounts,
            key=lambda x: int(
                x.get("claim_count", 0) or 0
            )
        )

    # ---------------------------------------------------------
    # HIGH VALUE CLAIMS
    # ---------------------------------------------------------

    highest_claim = None

    if high_value_rows:
        highest_claim = max(
            high_value_rows,
            key=lambda x: float(
                x.get("claim_amount", 0) or 0
            )
        )

    # ---------------------------------------------------------
    # DAILY CONCENTRATION
    # ---------------------------------------------------------

    highest_exposure_day = None

    if daily_rows:
        highest_exposure_day = max(
            daily_rows,
            key=lambda x: float(
                x.get("total_claim_amount", 0) or 0
            )
        )

    # ---------------------------------------------------------
    # FINDINGS
    # ---------------------------------------------------------

    findings = list(summary.get("findings", []))

    if strongest_repeated_amount:
        amount = strongest_repeated_amount.get("claim_amount")
        count = strongest_repeated_amount.get("claim_count")

        findings.append(
            f"Exact claim amount KES {amount:,.2f} "
            f"appears {count} times among positive-value claims."
        )

    if highest_claim:
        findings.append(
            f"The highest observed claim is "
            f"KES {float(highest_claim['claim_amount']):,.2f}."
        )

    if highest_exposure_day:
        findings.append(
            f"The highest daily monetary exposure occurred on "
            f"{highest_exposure_day['claim_date']} with "
            f"KES {float(highest_exposure_day['total_claim_amount']):,.2f}."
        )

    # ---------------------------------------------------------
    # CONCLUSIONS
    # ---------------------------------------------------------

    conclusions = []

    if zero_percentage >= 90:
        conclusions.append(
            "The county has an extremely zero-inflated claim-value "
            "distribution. The average claim amount therefore does "
            "not represent a typical claim."
        )

    if dominant_facility:
        exposure = float(
            dominant_facility.get("exposure_percentage", 0) or 0
        )

        if exposure >= 80:
            conclusions.append(
                f"Monetary exposure is highly concentrated in one "
                f"facility, which accounts for approximately "
                f"{exposure:.2f}% of county exposure."
            )

    if dominant_insurer:
        exposure = float(
            dominant_insurer.get("exposure_percentage", 0) or 0
        )

        if exposure >= 90:
            conclusions.append(
                f"The insurer distribution is highly concentrated, "
                f"with the dominant insurer accounting for "
                f"{exposure:.2f}% of monetary exposure."
            )

    if strongest_repeated_amount:
        count = int(
            strongest_repeated_amount.get("claim_count", 0) or 0
        )

        if count >= 5:
            conclusions.append(
                "Repeated exact claim amounts warrant validation "
                "against the underlying billing or tariff structure."
            )

    if highest_claim:
        conclusions.append(
            "The highest-value claims should be traced to their "
            "facility, provider, submission timing, and underlying "
            "service records."
        )

    # ---------------------------------------------------------
    # RECOMMENDATIONS
    # ---------------------------------------------------------

    recommendations = [
        "Review all high-value claims individually.",
        "Trace the dominant facility's positive-value claims "
        "to source records.",
        "Investigate repeated exact claim amounts.",
        "Determine why the majority of claims have zero monetary value.",
        "Compare positive-value claims with the same facilities "
        "and insurers in other counties.",
        "Check whether high-value claims cluster around specific "
        "submission dates or operational events.",
        "Validate the observed pattern against the underlying "
        "claims-generation and billing process.",
    ]

    # ---------------------------------------------------------
    # EXECUTIVE VERDICT
    # ---------------------------------------------------------

    verdict_parts = [
        f"{county} has a strong evidence-based anomaly requiring "
        f"further investigation."
    ]

    if zero_percentage:
        verdict_parts.append(
            f"{zero_percentage:.1f}% of claims have zero monetary value."
        )

    if dominant_facility:
        verdict_parts.append(
            f"The dominant facility accounts for "
            f"{float(dominant_facility['exposure_percentage']):.1f}% "
            f"of monetary exposure."
        )

    if dominant_insurer:
        verdict_parts.append(
            f"The dominant insurer accounts for "
            f"{float(dominant_insurer['exposure_percentage']):.1f}% "
            f"of monetary exposure."
        )

    if strongest_repeated_amount:
        verdict_parts.append(
            f"The most frequently repeated positive amount is "
            f"KES {float(strongest_repeated_amount['claim_amount']):,.2f}, "
            f"appearing {int(strongest_repeated_amount['claim_count'])} times."
        )

    verdict_parts.append(
        "These patterns constitute investigation signals, not proof "
        "of fraud or improper claims."
    )

    return {
        "county": county,
        "findings": findings,
        "conclusions": conclusions,
        "recommendations": recommendations,
        "zero_claim_percentage": zero_percentage,
        "positive_claim_percentage": positive_percentage,
        "dominant_facility": dominant_facility,
        "dominant_insurer": dominant_insurer,
        "strongest_repeated_amount": strongest_repeated_amount,
        "highest_claim": highest_claim,
        "highest_exposure_day": highest_exposure_day,
        "facilities": facility_rows,
        "insurers": insurer_rows,
        "high_value_claims": high_value_rows,
        "repeated_amounts": repeated_positive_amounts,
        "daily_activity": daily_rows,
        "executive_verdict": " ".join(verdict_parts),
    }
