"""
Claim Evidence Engine

Transforms scored claim cases into investigation-ready
evidence packages.

This layer DOES NOT determine fraud.

Pipeline:

claim_case_engine
        |
        v
claim_evidence_engine
        |
        v
investigation_workflow
"""

from claim_case_engine import build_claim_cases


# ============================================================
# SAFE CONVERSION HELPERS
# ============================================================

def _safe_float(value):
    """Safely convert a value to float."""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value):
    """Safely convert a value to int."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_list(value):
    """Safely convert a value into a list."""
    if not value:
        return []

    if isinstance(value, list):
        return list(value)

    return [value]


def _safe_dict(value):
    """Safely convert a value into a dictionary."""
    if isinstance(value, dict):
        return dict(value)

    return {}


# ============================================================
# STATISTICAL EVIDENCE
# ============================================================

def _build_statistical_evidence(case):
    """
    Build structured statistical evidence.

    This describes anomalies.
    It does NOT determine fraud.
    """

    signals = _safe_dict(
        case.get("statistical_signals")
    )

    evidence = []

    claim_amount = _safe_float(
        case.get("claim_amount")
    )

    national_percentile = _safe_float(
        signals.get("national_percentile")
    )

    peer_percentile = _safe_float(
        signals.get("peer_percentile")
    )

    daily_zscore = _safe_float(
        signals.get("daily_amount_zscore")
    )

    amount_frequency = _safe_int(
        signals.get("amount_frequency")
    )

    facility_exposure = _safe_float(
        signals.get(
            "facility_exposure_percentage"
        )
    )

    insurer_exposure = _safe_float(
        signals.get(
            "insurer_exposure_percentage"
        )
    )

    # --------------------------------------------------------
    # NATIONAL PERCENTILE
    # --------------------------------------------------------

    if national_percentile >= 99:
        evidence.append(
            {
                "type": "NATIONAL_PERCENTILE",
                "severity": "HIGH",
                "finding": (
                    "Claim amount is at or above "
                    "the 99th national percentile."
                ),
                "value": national_percentile,
            }
        )

    elif national_percentile >= 95:
        evidence.append(
            {
                "type": "NATIONAL_PERCENTILE",
                "severity": "MEDIUM",
                "finding": (
                    "Claim amount is within the "
                    "upper 5% of the national benchmark."
                ),
                "value": national_percentile,
            }
        )

    # --------------------------------------------------------
    # PEER PERCENTILE
    # --------------------------------------------------------

    if peer_percentile >= 99:
        evidence.append(
            {
                "type": "PEER_PERCENTILE",
                "severity": "HIGH",
                "finding": (
                    "Claim amount is at or above "
                    "the 99th claim-type percentile."
                ),
                "value": peer_percentile,
            }
        )

    elif peer_percentile >= 95:
        evidence.append(
            {
                "type": "PEER_PERCENTILE",
                "severity": "MEDIUM",
                "finding": (
                    "Claim amount is within the "
                    "upper 5% of its claim-type benchmark."
                ),
                "value": peer_percentile,
            }
        )

    # --------------------------------------------------------
    # DAILY Z-SCORE
    # --------------------------------------------------------

    if daily_zscore >= 5:
        evidence.append(
            {
                "type": "DAILY_AMOUNT_ZSCORE",
                "severity": "HIGH",
                "finding": (
                    "Daily claim exposure is an extreme "
                    "statistical deviation."
                ),
                "value": round(daily_zscore, 2),
            }
        )

    elif daily_zscore >= 3:
        evidence.append(
            {
                "type": "DAILY_AMOUNT_ZSCORE",
                "severity": "MEDIUM",
                "finding": (
                    "Daily claim exposure is materially "
                    "above the expected level."
                ),
                "value": round(daily_zscore, 2),
            }
        )

    # --------------------------------------------------------
    # AMOUNT FREQUENCY
    # --------------------------------------------------------

    if amount_frequency >= 5:
        evidence.append(
            {
                "type": "AMOUNT_FREQUENCY",
                "severity": "HIGH",
                "finding": (
                    "The exact claim amount appears "
                    "repeatedly in the benchmarked claims."
                ),
                "value": amount_frequency,
            }
        )

    elif amount_frequency >= 2:
        evidence.append(
            {
                "type": "AMOUNT_FREQUENCY",
                "severity": "MEDIUM",
                "finding": (
                    "The exact claim amount occurs "
                    "more than once."
                ),
                "value": amount_frequency,
            }
        )

    # --------------------------------------------------------
    # FACILITY CONCENTRATION
    # --------------------------------------------------------

    if facility_exposure >= 90:
        evidence.append(
            {
                "type": "FACILITY_CONCENTRATION",
                "severity": "HIGH",
                "finding": (
                    "The facility represents a dominant "
                    "share of the relevant claim exposure."
                ),
                "value": round(
                    facility_exposure,
                    2,
                ),
            }
        )

    elif facility_exposure >= 75:
        evidence.append(
            {
                "type": "FACILITY_CONCENTRATION",
                "severity": "MEDIUM",
                "finding": (
                    "The facility represents a high "
                    "share of the relevant claim exposure."
                ),
                "value": round(
                    facility_exposure,
                    2,
                ),
            }
        )

    # --------------------------------------------------------
    # INSURER CONCENTRATION
    # --------------------------------------------------------

    if insurer_exposure >= 90:
        evidence.append(
            {
                "type": "INSURER_CONCENTRATION",
                "severity": "HIGH",
                "finding": (
                    "The insurer represents a dominant "
                    "share of the relevant claim exposure."
                ),
                "value": round(
                    insurer_exposure,
                    2,
                ),
            }
        )

    elif insurer_exposure >= 75:
        evidence.append(
            {
                "type": "INSURER_CONCENTRATION",
                "severity": "MEDIUM",
                "finding": (
                    "The insurer represents a high "
                    "share of the relevant claim exposure."
                ),
                "value": round(
                    insurer_exposure,
                    2,
                ),
            }
        )

    # --------------------------------------------------------
    # CLAIM AMOUNT
    # --------------------------------------------------------

    if claim_amount >= 100000:
        evidence.append(
            {
                "type": "CLAIM_AMOUNT",
                "severity": "HIGH",
                "finding": (
                    "Claim amount is exceptionally large "
                    "within the observed claims."
                ),
                "value": claim_amount,
            }
        )

    elif claim_amount >= 30000:
        evidence.append(
            {
                "type": "CLAIM_AMOUNT",
                "severity": "MEDIUM",
                "finding": (
                    "Claim amount is materially elevated "
                    "within the observed claims."
                ),
                "value": claim_amount,
            }
        )

    return evidence


# ============================================================
# CONTEXT EVIDENCE
# ============================================================

def _build_context_evidence(case):
    """
    Convert contextual signals into structured evidence.
    """

    context_flags = _safe_list(
        case.get("context_flags")
    )

    evidence = []

    for flag in context_flags:
        evidence.append(
            {
                "type": "CONTEXT_SIGNAL",
                "severity": "MEDIUM",
                "finding": flag,
            }
        )

    return evidence


# ============================================================
# INVESTIGATION QUESTIONS
# ============================================================

def _build_investigation_questions(case):
    """
    Generate questions investigators should answer.

    These are investigative prompts, not allegations.
    """

    questions = [
        "What service or event generated the claim?",
        "What documentation supports the claimed amount?",
        "Does the submitted amount match the underlying records?",
        "Was the service actually delivered?",
        "Who authorized or submitted the claim?",
    ]

    signals = _safe_dict(
        case.get("statistical_signals")
    )

    if _safe_float(
        signals.get("daily_amount_zscore")
    ) >= 3:
        questions.append(
            "What explains the unusual daily exposure spike?"
        )

    if _safe_int(
        signals.get("amount_frequency")
    ) >= 2:
        questions.append(
            "Why is the exact claim amount repeated?"
        )

    if _safe_float(
        signals.get("facility_exposure_percentage")
    ) >= 75:
        questions.append(
            "What explains the concentration of exposure at this facility?"
        )

    if _safe_float(
        signals.get("insurer_exposure_percentage")
    ) >= 75:
        questions.append(
            "What explains the concentration of exposure under this insurer?"
        )

    if _safe_float(
        signals.get("national_percentile")
    ) >= 95:
        questions.append(
            "What explains the claim's unusually high national benchmark position?"
        )

    if _safe_float(
        signals.get("peer_percentile")
    ) >= 95:
        questions.append(
            "What explains the claim's unusually high position within its claim type?"
        )

    return questions


# ============================================================
# RECOMMENDED DOCUMENTS
# ============================================================

def _build_recommended_documents(case):
    """
    Identify evidence an investigator should obtain.
    """

    documents = [
        "Original claim submission",
        "Supporting invoices or receipts",
        "Service delivery records",
        "Authorization records",
        "Patient or beneficiary records where legally appropriate",
        "Relevant facility records",
    ]

    signals = _safe_dict(
        case.get("statistical_signals")
    )

    if _safe_int(
        signals.get("amount_frequency")
    ) >= 2:
        documents.append(
            "Related claims with the same claim amount"
        )

    if _safe_float(
        signals.get("daily_amount_zscore")
    ) >= 3:
        documents.append(
            "Claims submitted on the same date"
        )

    if _safe_float(
        signals.get("facility_exposure_percentage")
    ) >= 75:
        documents.append(
            "Facility-level claims and payment history"
        )

    if _safe_float(
        signals.get("insurer_exposure_percentage")
    ) >= 75:
        documents.append(
            "Insurer-level claims and payment history"
        )

    return documents


# ============================================================
# CASE ASSESSMENT
# ============================================================

def _build_case_assessment(case):
    """
    Produce a neutral investigation assessment.
    """

    priority = _safe_dict(
        case.get("priority")
    )

    score = _safe_float(
        priority.get("score")
    )

    level = (
        priority.get("level")
        or "LOW"
    )

    if level == "CRITICAL":
        assessment = (
            "The claim contains multiple high-priority "
            "statistical and contextual signals and "
            "should receive immediate investigation."
        )

    elif level == "HIGH":
        assessment = (
            "The claim contains material statistical "
            "or contextual anomalies and should receive "
            "priority investigation."
        )

    elif level == "MEDIUM":
        assessment = (
            "The claim contains moderate signals that "
            "warrant secondary review."
        )

    else:
        assessment = (
            "The available signals do not indicate an "
            "immediate investigation priority."
        )

    return {
        "priority_level": level,
        "priority_score": round(
            score,
            2,
        ),
        "assessment": assessment,
        "fraud_determination": None,
    }


# ============================================================
# EVIDENCE PACKAGE
# ============================================================

def _build_evidence_package(case):
    """
    Build a complete investigation evidence package.

    IMPORTANT:
    The original scored claim data is preserved.
    The evidence engine enriches it; it does not discard it.
    """

    statistical_evidence = _build_statistical_evidence(
        case
    )

    context_evidence = _build_context_evidence(
        case
    )

    all_structured_evidence = (
        statistical_evidence
        + context_evidence
    )

    assessment = _build_case_assessment(
        case
    )

    return {
        # ----------------------------------------------------
        # IDENTIFIERS
        # ----------------------------------------------------

        "case_id": case.get("case_id"),

        "claim_id": case.get("claim_id"),

        "county": case.get("county"),

        # ----------------------------------------------------
        # CLAIM METADATA
        # ----------------------------------------------------

        "facility": case.get("facility"),

        "facility_name": case.get(
            "facility_name"
        ),

        "insurer": case.get(
            "insurer"
        ),

        "claim_type": case.get(
            "claim_type"
        ),

        "creation": case.get(
            "creation"
        ),

        "claim_amount": _safe_float(
            case.get("claim_amount")
        ),

        # ----------------------------------------------------
        # ORIGINAL PRIORITY
        # ----------------------------------------------------

        "priority": _safe_dict(
            case.get("priority")
        ),

        # ----------------------------------------------------
        # ORIGINAL SCORES
        #
        # THIS IS THE CRITICAL FIX.
        # ----------------------------------------------------

        "scores": _safe_dict(
            case.get("scores")
        ),

        # ----------------------------------------------------
        # ORIGINAL HUMAN-READABLE EVIDENCE
        #
        # Preserve the six evidence strings produced by
        # claim_case_engine.
        # ----------------------------------------------------

        "evidence": _safe_list(
            case.get("evidence")
        ),

        # ----------------------------------------------------
        # CONTEXT FLAGS
        # ----------------------------------------------------

        "context_flags": _safe_list(
            case.get("context_flags")
        ),

        # ----------------------------------------------------
        # STATISTICAL SIGNALS
        # ----------------------------------------------------

        "statistical_signals": _safe_dict(
            case.get("statistical_signals")
        ),

        # ----------------------------------------------------
        # ENRICHED STRUCTURED EVIDENCE
        # ----------------------------------------------------

        "statistical_evidence": (
            statistical_evidence
        ),

        "context_evidence": (
            context_evidence
        ),

        "structured_evidence": (
            all_structured_evidence
        ),

        "evidence_count": len(
            all_structured_evidence
        ),

        # ----------------------------------------------------
        # INVESTIGATION SUPPORT
        # ----------------------------------------------------

        "investigation_questions": (
            _build_investigation_questions(
                case
            )
        ),

        "recommended_documents": (
            _build_recommended_documents(
                case
            )
        ),

        "assessment": assessment,

        # ----------------------------------------------------
        # WORKFLOW STATE
        # ----------------------------------------------------

        "investigation_status": (
            case.get(
                "investigation_status",
                "OPEN",
            )
        ),

        # No fraud determination is made here.
        "fraud_determination": None,
    }


# ============================================================
# PUBLIC API
# ============================================================

def build_evidence_packages(
    county,
    limit=25,
):
    """
    Build investigation evidence packages for a county.

    Returns:

        {
            "county": ...,
            "package_count": ...,
            "packages": [...]
        }
    """

    result = build_claim_cases(
        county,
        limit=limit,
    )

    if result.get("error"):
        return {
            "county": county,
            "package_count": 0,
            "packages": [],
            "error": result["error"],
        }

    claim_cases = (
        result.get("cases")
        or []
    )

    packages = []

    for case in claim_cases:
        packages.append(
            _build_evidence_package(
                case
            )
        )

    return {
        "county": county,
        "package_count": len(packages),
        "packages": packages,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    result = build_evidence_packages(
        "KITUI",
        limit=1,
    )

    print("=" * 80)
    print("CLAIM EVIDENCE ENGINE TEST")
    print("=" * 80)

    print(
        "PACKAGE COUNT:",
        result["package_count"],
    )

    if result["packages"]:
        package = result["packages"][0]

        print(
            "CLAIM:",
            package["claim_id"],
        )

        print(
            "AMOUNT:",
            package["claim_amount"],
        )

        print(
            "PRIORITY:",
            package["priority"],
        )

        print(
            "SCORES:",
            package["scores"],
        )

        print(
            "EVIDENCE:",
            package["evidence"],
        )

        print(
            "CONTEXT FLAGS:",
            package["context_flags"],
        )

        print(
            "STATISTICAL SIGNALS:",
            package["statistical_signals"],
        )

        print(
            "FACILITY:",
            package["facility"],
        )

        print(
            "FACILITY NAME:",
            package["facility_name"],
        )

        print(
            "INSURER:",
            package["insurer"],
        )

        print(
            "CLAIM TYPE:",
            package["claim_type"],
        )

    print("=" * 80)