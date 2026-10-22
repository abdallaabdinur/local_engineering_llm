from claim_investigation_engine import investigate_claims
from benchmark_engine import benchmark_county


def _safe_float(value):
    """
    Safely convert a value to float.
    """
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _benchmark_score(national_percentile, peer_percentile):
    """
    Convert benchmark percentiles into a capped
    0-30 benchmark contribution.

    We use the stronger of the national and
    peer benchmarks rather than adding them,
    preventing double counting.
    """

    strongest_percentile = max(
        national_percentile,
        peer_percentile,
    )

    if strongest_percentile >= 99:
        return 30

    if strongest_percentile >= 97:
        return 27

    if strongest_percentile >= 95:
        return 24

    if strongest_percentile >= 90:
        return 20

    if strongest_percentile >= 80:
        return 15

    if strongest_percentile >= 70:
        return 10

    if strongest_percentile >= 50:
        return 5

    return 0


def _anomaly_score_component(score):
    """
    Convert the existing investigation anomaly
    score into a 0-50 contribution.

    Existing anomaly score currently has a maximum
    of approximately 7 under the current rules.
    """

    score = _safe_float(score)

    return min(
        50,
        round(
            (score / 7.0) * 50,
            2,
        ),
    )


def _context_score_component(score):
    """
    Convert contextual concentration evidence
    into a 0-20 contribution.

    Current context score is expected to be
    approximately 0-2.
    """

    score = _safe_float(score)

    return min(
        20,
        round(
            (score / 2.0) * 20,
            2,
        ),
    )


def _priority_level(priority_score):
    """
    Convert priority score into an operational tier.
    """

    if priority_score >= 85:
        return "CRITICAL"

    if priority_score >= 70:
        return "HIGH"

    if priority_score >= 50:
        return "MEDIUM"

    return "LOW"


def _recommended_action(priority_level):
    """
    Map priority level to an investigation action.
    """

    if priority_level == "CRITICAL":
        return (
            "PRIORITY INVESTIGATION"
        )

    if priority_level == "HIGH":
        return (
            "INVESTIGATE BEFORE ROUTINE PROCESSING"
        )

    if priority_level == "MEDIUM":
        return (
            "SECONDARY REVIEW"
        )

    return (
        "ROUTINE PROCESSING"
    )


def _benchmark_reasons(
    national_percentile,
    peer_percentile,
):
    """
    Generate human-readable benchmark evidence.
    """

    reasons = []

    if national_percentile >= 99:
        reasons.append(
            "At or above the 99th national percentile"
        )

    elif national_percentile >= 95:
        reasons.append(
            "At or above the 95th national percentile"
        )

    elif national_percentile >= 90:
        reasons.append(
            "At or above the 90th national percentile"
        )

    if peer_percentile >= 99:
        reasons.append(
            "At or above the 99th peer percentile"
        )

    elif peer_percentile >= 95:
        reasons.append(
            "At or above the 95th peer percentile"
        )

    elif peer_percentile >= 90:
        reasons.append(
            "At or above the 90th peer percentile"
        )

    return reasons


def _deduplicate(items):
    """
    Preserve order while removing duplicate evidence.
    """

    result = []
    seen = set()

    for item in items:

        if item in seen:
            continue

        seen.add(item)
        result.append(item)

    return result


def decide_claims(county, limit=25):
    """
    Combine claim-level investigation evidence
    with statistical benchmark evidence.

    This is an investigation-priority engine.

    It does NOT estimate fraud probability.
    It does NOT determine whether a claim is fraudulent.

    Scores:

        anomaly contribution  : 0-50
        context contribution  : 0-20
        benchmark contribution: 0-30

        total priority score : 0-100
    """

    county = county.upper().strip()

    # =====================================================
    # 1. RUN EXISTING ENGINES
    # =====================================================

    investigation = investigate_claims(
        county,
        limit=limit,
    )

    if investigation.get("error"):
        return investigation

    benchmark = benchmark_county(
        county,
    )

    if benchmark.get("error"):
        return {
            "county": county,
            "error": benchmark["error"],
        }

    # =====================================================
    # 2. INDEX BENCHMARK RESULTS BY CLAIM ID
    # =====================================================

    benchmark_by_id = {
        item["claim_id"]: item
        for item in benchmark.get(
            "claims",
            [],
        )
    }

    # =====================================================
    # 3. COMBINE CLAIMS
    # =====================================================

    decisions = []

    for claim in investigation.get(
        "claims",
        [],
    ):

        claim_id = claim.get(
            "claim_id"
        )

        benchmark_item = (
            benchmark_by_id.get(
                claim_id
            )
        )

        if benchmark_item is None:
            # This should never happen if the
            # two engines are operating on the
            # same positive-claim population.
            continue

        # -------------------------------------------------
        # Existing investigation scores
        # -------------------------------------------------

        anomaly_score = _safe_float(
            claim.get(
                "anomaly_score",
                0,
            )
        )

        context_score = _safe_float(
            claim.get(
                "context_score",
                0,
            )
        )

        # -------------------------------------------------
        # Benchmark evidence
        # -------------------------------------------------

        national_percentile = _safe_float(
            benchmark_item.get(
                "national_percentile",
                0,
            )
        )

        peer_percentile = _safe_float(
            benchmark_item.get(
                "claim_type_percentile",
                0,
            )
        )

        # -------------------------------------------------
        # Convert components
        # -------------------------------------------------

        anomaly_component = (
            _anomaly_score_component(
                anomaly_score
            )
        )

        context_component = (
            _context_score_component(
                context_score
            )
        )

        benchmark_component = (
            _benchmark_score(
                national_percentile,
                peer_percentile,
            )
        )

        # -------------------------------------------------
        # Final priority score
        # -------------------------------------------------

        priority_score = round(
            anomaly_component
            + context_component
            + benchmark_component,
            2,
        )

        priority_score = min(
            100.0,
            priority_score,
        )

        priority_level = (
            _priority_level(
                priority_score
            )
        )

        recommended_action = (
            _recommended_action(
                priority_level
            )
        )

        # -------------------------------------------------
        # Evidence
        # -------------------------------------------------

        investigation_reasons = (
            claim.get(
                "risk_reasons",
                [],
            )
        )

        context_flags = (
            claim.get(
                "context_flags",
                [],
            )
        )

        benchmark_reasons = (
            _benchmark_reasons(
                national_percentile,
                peer_percentile,
            )
        )

        evidence = _deduplicate(
            investigation_reasons
            + context_flags
            + benchmark_reasons
        )

        # -------------------------------------------------
        # Build decision record
        # -------------------------------------------------

        decision = dict(claim)

        decision.update(
            {
                "national_percentile": round(
                    national_percentile,
                    2,
                ),

                "peer_percentile": round(
                    peer_percentile,
                    2,
                ),

                "benchmark_score": (
                    benchmark_component
                ),

                "anomaly_component": (
                    anomaly_component
                ),

                "context_component": (
                    context_component
                ),

                "priority_score": (
                    priority_score
                ),

                "priority_level": (
                    priority_level
                ),

                "recommended_action": (
                    recommended_action
                ),

                "benchmark_reasons": (
                    benchmark_reasons
                ),

                "evidence": evidence,
            }
        )

        decisions.append(
            decision
        )

    # =====================================================
    # 4. SORT BY PRIORITY
    # =====================================================

    decisions.sort(
        key=lambda item: (
            item["priority_score"],
            _safe_float(
                item.get(
                    "claim_amount",
                    0,
                )
            ),
        ),
        reverse=True,
    )

    # =====================================================
    # 5. SUMMARY
    # =====================================================

    priority_counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for item in decisions:

        level = item[
            "priority_level"
        ]

        if level in priority_counts:
            priority_counts[level] += 1

    return {
        "county": county,

        "claim_count": len(
            decisions
        ),

        "claims": decisions,

        "summary": {
            "critical": priority_counts[
                "CRITICAL"
            ],

            "high": priority_counts[
                "HIGH"
            ],

            "medium": priority_counts[
                "MEDIUM"
            ],

            "low": priority_counts[
                "LOW"
            ],

            "benchmark_claim_count": len(
                benchmark.get(
                    "claims",
                    [],
                )
            ),

            "benchmark_integrity": (
                benchmark.get(
                    "summary",
                    {},
                ).get(
                    "data_integrity_ok",
                    False,
                )
            ),
        },
    }
