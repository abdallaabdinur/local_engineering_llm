import os
import re
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("claims_agent.query")


# ============================================================
# QUERY PLAN
# ============================================================

@dataclass
class QueryPlan:
    route: str
    intent: str
    county: Optional[str] = None
    counties: Optional[list[str]] = None
    limit: int = 10
    sql: Optional[str] = None
    complexity_score: Optional[int] = None
    complexity_band: Optional[str] = None
    analysis_seconds: float = 0.0


# ============================================================
# COMPLEXITY SCORING (OpenAI)
# ============================================================
#
# When no deterministic SQL pattern matches a question, we ask
# a real OpenAI model to rate how complex the question is on a
# scale of 1-10 BEFORE handing it to the local SQL-generation
# model. The score decides which SQL-generation strategy
# agent.py should use:
#
#   1-3   SIMPLE      -> single-shot SQL generation
#   4-7   STANDARD    -> single-shot SQL generation + repair
#   8-10  MULTI_STEP  -> multi-step / agentic SQL generation
#
# This call is intentionally decoupled from the local Ollama
# client used for actual SQL generation elsewhere in this
# project (see agent.py) -- it uses a separate, faster/cheaper
# hosted model because a small local model is not reliable at
# self-rating difficulty. Defaults to Groq's free tier; see
# _get_complexity_client() below for how to point it elsewhere.
#
# The call is only made when a deterministic pattern in
# build_sql() did NOT match, so it never adds latency or cost to
# the fast-path questions that already work today.
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

# Groq's free tier (https://console.groq.com/keys) is the default
# provider for complexity scoring since it's a genuinely free,
# OpenAI-compatible API -- no card required, generous rate limits.
# Set GROQ_API_KEY in .env to use it. If you'd rather use real
# OpenAI (or any other OpenAI-compatible provider), set
# COMPLEXITY_API_KEY and COMPLEXITY_BASE_URL explicitly and they
# take priority over the Groq defaults below.
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

COMPLEXITY_API_KEY = (
    os.getenv("COMPLEXITY_API_KEY")
    or GROQ_API_KEY
    or OPENAI_API_KEY
)

COMPLEXITY_BASE_URL = os.getenv(
    "COMPLEXITY_BASE_URL",
    (
        "https://api.groq.com/openai/v1"
        if (
            os.getenv("COMPLEXITY_API_KEY") is None
            and GROQ_API_KEY is not None
        )
        else None
    ),
)

COMPLEXITY_MODEL = os.getenv(
    "COMPLEXITY_MODEL",
    "qwen/qwen3.8-27b",
)

COMPLEXITY_TIMEOUT = float(
    os.getenv(
        "COMPLEXITY_TIMEOUT",
        "10",
    )
)

# Used whenever the scoring call itself is unavailable (missing
# API key, network error, bad response, etc). STANDARD is a
# deliberately safe middle-ground default: it keeps using the
# existing, already-working single-shot generation path rather
# than either under- or over-escalating a question we couldn't
# actually score.
DEFAULT_COMPLEXITY_SCORE = 5

_complexity_client = None


def _get_complexity_client():
    """
    Lazily construct the OpenAI-compatible client used only for
    complexity scoring. Lazy so importing this module never fails
    just because no API key is set (e.g. in tests, or when a
    deployment never needs the LLM/COMPLEX path).

    Defaults to Groq's free, OpenAI-compatible endpoint. Set
    COMPLEXITY_API_KEY (+ optionally COMPLEXITY_BASE_URL) to point
    this at a different provider, such as real OpenAI, OpenRouter,
    or anything else that speaks the OpenAI chat completions API.
    """

    global _complexity_client

    if _complexity_client is not None:

        return _complexity_client

    if not COMPLEXITY_API_KEY:

        return None

    from openai import OpenAI

    client_kwargs = {
        "api_key": COMPLEXITY_API_KEY,
    }

    if COMPLEXITY_BASE_URL:

        client_kwargs["base_url"] = COMPLEXITY_BASE_URL

    _complexity_client = OpenAI(
        **client_kwargs
    )

    return _complexity_client


COMPLEXITY_SYSTEM_PROMPT = """
You rate the complexity of natural-language questions that will
be turned into a PostgreSQL query against a healthcare claims
database.

Respond with a JSON object in this exact format:
{"score": <integer from 1 to 10>}

Guidance:

1-3  = a single aggregate over one table, one or two simple
       filters (e.g. a count, sum, or average with an optional
       county/date filter).

4-7  = requires grouping, ranking, multiple filters, joins
       across concepts, or comparing more than one segment.

8-10 = requires multiple dependent steps, statistical reasoning,
       ambiguous or open-ended analysis, or combining several
       unrelated metrics into one judgement.
"""


def score_complexity(
    question: str,
) -> int:
    """
    Ask OpenAI/Groq to rate a question's complexity from 1-10.

    Always returns an int in [1, 10]. Results are cached in
    _COMPLEXITY_CACHE keyed on normalize_question() to avoid
    redundant LLM round-trips for repeated or similar questions.
    Uses response_format={"type": "json_object"} for structured output.
    Falls back to DEFAULT_COMPLEXITY_SCORE on any failure (missing key,
    network error, malformed response) so a scoring outage never
    breaks the planner -- it just degrades to today's behavior.
    """

    if not question:
        return DEFAULT_COMPLEXITY_SCORE

    cached_score = get_cached_complexity(question)
    if cached_score is not None:
        logger.info(
            "Complexity score from cache for %r: %d",
            question,
            cached_score,
        )
        return cached_score

    client = _get_complexity_client()

    if client is None:

        logger.warning(
            "No complexity-scoring API key set "
            "(GROQ_API_KEY / COMPLEXITY_API_KEY / "
            "OPENAI_API_KEY); using default complexity "
            "score %d",
            DEFAULT_COMPLEXITY_SCORE,
        )

        return DEFAULT_COMPLEXITY_SCORE

    try:

        response = client.chat.completions.create(
            model=COMPLEXITY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": COMPLEXITY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=30,
            temperature=0,
            timeout=COMPLEXITY_TIMEOUT,
        )

        content = (
            response.choices[0].message.content
            or ""
        ).strip()

        score = None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                val = parsed.get("score")
                if val is None:
                    val = (
                        parsed.get("complexity")
                        or parsed.get("complexity_score")
                    )
                if val is not None:
                    score = int(val)
            elif isinstance(parsed, (int, float)):
                score = int(parsed)
        except Exception:
            pass

        if score is None:
            match = re.search(
                r"\d+",
                content,
            )
            if match:
                score = int(match.group(0))

        if score is None:

            logger.warning(
                "Complexity scoring returned no score "
                "(%r); using default score %d",
                content,
                DEFAULT_COMPLEXITY_SCORE,
            )

            return DEFAULT_COMPLEXITY_SCORE

        final_score = min(
            max(score, 1),
            10,
        )

        cache_complexity(
            question,
            final_score,
        )

        return final_score

    except Exception:

        logger.exception(
            "Complexity scoring call failed; "
            "using default score %d",
            DEFAULT_COMPLEXITY_SCORE,
        )

        return DEFAULT_COMPLEXITY_SCORE


def band_for_score(
    score: int,
) -> str:
    """
    Map a 1-10 complexity score to a routing band.
    """

    if score <= 3:

        return "SIMPLE"

    if score <= 7:

        return "STANDARD"

    return "MULTI_STEP"


# ============================================================
# COUNTY NORMALIZATION
# ============================================================

COUNTIES = {
    "nairobi": "NAIROBI",
    "mombasa": "MOMBASA",
    "nandi": "NANDI",
    "embu": "EMBU",
    "kirinyaga": "KIRINYAGA",
    "elgeyo marakwet": "ELGEYO MARAKWET",
    "bomet": "BOMET",
    "makueni": "MAKUENI",
    "uasin gishu": "UASIN GISHU",
    "kakamega": "KAKAMEGA",
    "kitui": "KITUI",
    "trans nzoia": "TRANS NZOIA",
    "narok": "NAROK",
    "marsabit": "MARSABIT",
    "kwale": "KWALE",
    "machakos": "MACHAKOS",
    "west pokot": "WEST POKOT",
    "kiambu": "KIAMBU",
    "nyamira": "NYAMIRA",
    "tana river": "TANA RIVER",
    "migori": "MIGORI",
    "baringo": "BARINGO",
    "lamu": "LAMU",
    "kisii": "KISII",
    "nakuru": "NAKURU",
    "bungoma": "BUNGOMA",
    "garissa": "GARISSA",
    "mandera": "MANDERA",
    "samburu": "SAMBURU",
    "nyeri": "NYERI",
    "vihiga": "VIHIGA",
    "nyandarua": "NYANDARUA",
    "kericho": "KERICHO",
    "busia": "BUSIA",
    "taita taveta": "TAITA TAVETA",
    "waijir": "WAJIR",
    "wajir": "WAJIR",
    "turkana": "TURKANA",
    "laikipia": "LAIKIPIA",
    "muranga": "MURANGA",
    "murang'a": "MURANGA",
    "isiolo": "ISIOLO",
    "homa bay": "HOMA BAY",
    "tharaka nithi": "THARAKA NITHI",
    "meru": "MERU",
}


# ============================================================
# QUESTION NORMALIZATION
# ============================================================

def normalize_question(question: str) -> str:
    """
    Normalize whitespace and casing.
    """

    if not question:
        return ""

    question = str(question).lower().strip()

    question = re.sub(
        r"\s+",
        " ",
        question,
    )

    return question


def extract_time_window(question: str) -> Optional[str]:
    """Return a normalized relative time interval from a question."""

    q = normalize_question(question)
    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    match = re.search(
        r"\b(?:last|past|previous)\s+"
        r"(?:(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+)?"
        r"(minutes?|hours?|days?|weeks?|months?|years?)\b",
        q,
    )

    if not match:
        return None

    raw_amount = match.group(1) or "1"
    amount = number_words.get(raw_amount, None)
    if amount is None:
        amount = int(raw_amount)

    if not 1 <= amount <= 1000:
        return None

    unit = match.group(2).rstrip("s")
    return f"{amount} {unit}"


# ============================================================
# COUNTY EXTRACTION
# ============================================================

def extract_county(
    question: str,
) -> Optional[str]:

    normalized = normalize_question(
        question
    )

    candidates = sorted(
        COUNTIES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    for name, database_value in candidates:

        pattern = (
            r"\b"
            + re.escape(name)
            + r"\b"
        )

        if re.search(
            pattern,
            normalized,
        ):

            return database_value

    return None


# ============================================================
# MULTI-COUNTY EXTRACTION
# ============================================================

def extract_counties(
    question: str,
) -> list[str]:

    normalized = normalize_question(
        question
    )

    candidates = sorted(
        COUNTIES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )

    found_with_positions = []

    for name, database_value in candidates:

        pattern = (
            r"\b"
            + re.escape(name)
            + r"\b"
        )

        match = re.search(
            pattern,
            normalized,
        )

        if match:

            found_with_positions.append(
                (
                    match.start(),
                    database_value,
                )
            )

    found_with_positions.sort(
        key=lambda x: x[0]
    )

    found = []

    for _, database_value in found_with_positions:

        if database_value not in found:

            found.append(
                database_value
            )

    return found


# ============================================================
# LIMIT EXTRACTION
# ============================================================

def extract_limit(
    question: str,
    default: int = 10,
) -> int:

    match = re.search(
        r"\b(?:top|bottom)\s+(\d+)\b",
        normalize_question(question),
    )

    if match:

        try:

            value = int(
                match.group(1)
            )

            return min(
                max(value, 1),
                100,
            )

        except ValueError:
            pass

    return default


# ============================================================
# COMPARISON DETECTION
# ============================================================

def is_comparison_question(
    question: str,
) -> bool:

    q = normalize_question(
        question
    )

    comparison_terms = [
        "compare",
        "comparison",
        "versus",
        " vs ",
        "difference between",
        "between",
        "compared with",
        "compared to",
        "against",
        "higher than",
        "lower than",
        "more than",
        "less than",
    ]

    padded = f" {q} "

    return any(
        term in padded
        for term in comparison_terms
    )


# ============================================================
# INTENT DETECTION
# ============================================================

def detect_intent(
    question: str,
) -> str:

    q = normalize_question(
        question
    )

    counties = extract_counties(
        q
    )

    comparison = is_comparison_question(
        q
    )

    # ========================================================
    # ANOMALY / UNUSUAL PATTERN ANALYSIS
    # ========================================================

    anomaly_terms = [
        "unusual",
        "unusually",
        "anomal",
        "anomaly",
        "anomalies",
        "outlier",
        "outliers",
        "strange pattern",
        "strange patterns",
        "unusual pattern",
        "unusual patterns",
        "unusual claim",
        "unusual claims",
        "suspicious pattern",
        "suspicious patterns",
        "suspicious claim pattern",
        "suspicious claim patterns",
        "suspicious claims pattern",
        "suspicious claims patterns",
        "irregular pattern",
        "irregular patterns",
        "unexpected pattern",
        "unexpected patterns",
        "something complicated",
        "complicated analysis",
        "complex analysis",
        "interesting pattern",
        "interesting patterns",
        "fraud pattern",
        "fraud patterns",
        "risk pattern",
        "risk patterns",
        "abnormal claim",
        "abnormal claims",
    ]

    if any(
        term in q
        for term in anomaly_terms
    ):

        return "ANOMALY_ANALYSIS"

    # ========================================================
    # CLIENTS WITH MULTIPLE CLAIMS ON THE SAME DAY
    # ========================================================
    #
    # Must be checked BEFORE the generic "SIMPLE CLAIM COUNT"
    # branch below. A phrase like "most claim counts" contains
    # the literal substring "claim count", which would otherwise
    # be caught by the generic COUNT branch's
    # `"claim count" in q` check and silently collapse the
    # question down to `SELECT COUNT(*) FROM claims`, discarding
    # "clients" and "same day" entirely. Checking this more
    # specific intent first prevents that misclassification.
    # ========================================================

    same_day_terms = [
        "same day",
        "same-day",
        "per day",
        "each day",
        "in a day",
        "single day",
        "one day",
    ]

    client_terms = [
        "client",
        "clients",
        "member",
        "members",
        "patient",
        "patients",
    ]

    repeat_terms = [
        "most",
        "highest",
        "top",
        "repeat",
        "repeated",
        "duplicate",
        "duplicates",
        "multiple",
    ]

    if (
        any(term in q for term in same_day_terms)
        and any(term in q for term in client_terms)
        and any(term in q for term in repeat_terms)
    ):

        return "CLIENTS_MULTIPLE_CLAIMS_SAME_DAY"

    multiple_claim_phrases = [
        "more than one claim",
        "more than 1 claim",
        "multiple claims",
        "at least two claims",
        "at least 2 claims",
        "two or more claims",
        "2 or more claims",
        "over one claim",
        "over 1 claim",
    ]

    if (
        any(term in q for term in client_terms)
        and (
            any(phrase in q for phrase in multiple_claim_phrases)
            or re.search(r"claims?\s*(?:count\s*)?>\s*1\b", q)
        )
    ):

        return "CLIENTS_MULTIPLE_CLAIMS"

    # ========================================================
    # COUNTY AVERAGE COMPARISON
    # ========================================================

    if (
        len(counties) >= 2
        and (
            "average" in q
            or "mean" in q
        )
        and (
            "amount" in q
            or "value" in q
            or "claim amount" in q
        )
        and comparison
    ):

        return "COMPARE_AVERAGE_AMOUNT"

    # ========================================================
    # COUNTY CLAIM COUNT COMPARISON
    # ========================================================

    if (
        len(counties) >= 2
        and (
            "how many claims" in q
            or "number of claims" in q
            or "claim count" in q
            or "count of claims" in q
            or (
                "claims" in q
                and comparison
            )
        )
        and comparison
    ):

        return "COMPARE_CLAIM_COUNT"

    # ========================================================
    # COUNTY CANCELLATION RATE COMPARISON
    # ========================================================

    if (
        len(counties) >= 2
        and (
            "cancel" in q
            or "cancellation" in q
        )
        and (
            "rate" in q
            or "percentage" in q
            or "percent" in q
            or "%" in q
        )
        and comparison
    ):

        return "COMPARE_CANCELLATION_RATE"

    # ========================================================
    # COUNTY TOTAL AMOUNT COMPARISON
    # ========================================================

    if (
        len(counties) >= 2
        and (
            "total" in q
            or "sum" in q
        )
        and (
            "amount" in q
            or "value" in q
            or "spending" in q
        )
        and comparison
    ):

        return "COMPARE_TOTAL_AMOUNT"

    # ========================================================
    # CANCELLATION PERCENTAGE
    # ========================================================

    if (
        (
            "percentage" in q
            or "percent" in q
            or "%" in q
            or "rate" in q
        )
        and (
            "cancelled" in q
            or "canceled" in q
            or "cancellation" in q
        )
    ):

        return "CANCELLED_PERCENTAGE"

    # ========================================================
    # CANCELLATION COUNT
    # ========================================================

    if (
        (
            "how many" in q
            or "number of" in q
            or "count" in q
        )
        and (
            "cancelled" in q
            or "canceled" in q
            or "cancellation" in q
        )
    ):

        return "CANCELLED_COUNT"

    # ========================================================
    # LAST MONTH / YEAR simple counts
    # ========================================================

    time_window = extract_time_window(q)

    if ("how many" in q or "number of" in q or "count" in q) and time_window is not None:
        if time_window == "LAST_MONTH":
            return "COUNT_LAST_MONTH"
        if time_window == "LAST_YEAR":
            return "COUNT_LAST_YEAR"

    # ========================================================
    # TOP COUNTIES
    # ========================================================

    if (
        (
            "top" in q
            or "highest" in q
            or "most" in q
        )
        and (
            "counties" in q
            or "county" in q
        )
        and (
            "claims" in q
            or "claim" in q
        )
    ):

        return "TOP_COUNTIES"

    # ========================================================
    # BOTTOM COUNTIES
    # ========================================================

    if (
        (
            "bottom" in q
            or "lowest" in q
            or "least" in q
        )
        and (
            "counties" in q
            or "county" in q
        )
        and (
            "claims" in q
            or "claim" in q
        )
    ):

        return "BOTTOM_COUNTIES"

    # ========================================================
    # AVERAGE BY COUNTY
    # ========================================================

    if (
        (
            "average" in q
            or "mean" in q
        )
        and (
            "by county" in q
            or "each county" in q
            or "per county" in q
        )
    ):

        return "AVERAGE_BY_COUNTY"

    # ========================================================
    # TOTAL AMOUNT BY COUNTY
    # ========================================================

    if (
        (
            "total" in q
            or "sum" in q
        )
        and (
            "amount" in q
            or "value" in q
        )
        and (
            "by county" in q
            or "each county" in q
            or "per county" in q
        )
    ):

        return "TOTAL_AMOUNT_BY_COUNTY"

    # ========================================================
    # TOP FACILITIES BY AMOUNT
    # ========================================================

    if (
        (
            "top" in q
            or "highest" in q
        )
        and "facilit" in q
        and (
            "amount" in q
            or "value" in q
            or "spending" in q
        )
    ):

        return "TOP_FACILITIES_AMOUNT"

    # ========================================================
    # TOP FACILITIES BY CLAIM COUNT
    # ========================================================

    if (
        (
            "top" in q
            or "highest" in q
            or "most" in q
        )
        and "facilit" in q
        and (
            "claims" in q
            or "claim count" in q
        )
    ):

        return "TOP_FACILITIES_COUNT"

    # ========================================================
    # TOP INSURER
    # ========================================================

    if (
        (
            "which insurer" in q
            or "what insurer" in q
            or "top insurer" in q
            or "most claims" in q
        )
        and "insur" in q
    ):

        return "TOP_INSURER"

    # ========================================================
    # TOTAL BY INSURER
    # ========================================================

    # Top insurers by amount or count (explicit 'top' phrasing)
    if (
        (
            "top" in q
            or "highest" in q
            or "most" in q
        )
        and "insur" in q
    ):
        if "amount" in q or "total" in q or "spending" in q:
            return "TOP_INSURERS_AMOUNT"
        return "TOP_INSURERS_COUNT"

    if (
        "insurer" in q
        and (
            "total amount" in q
            or "total claim amount" in q
            or "sum" in q
        )
    ):

        return "TOTAL_AMOUNT_BY_INSURER"

    # ========================================================
    # CLAIM TYPES
    # ========================================================

    if (
        "claim type" in q
        or "claim types" in q
        or "most common type" in q
        or "common claim type" in q
    ):

        return "CLAIM_TYPES"

    # ========================================================
    # MONTHLY CLAIMS
    # ========================================================

    if (
        (
            "each month" in q
            or "per month" in q
            or "monthly" in q
            or "by month" in q
        )
        and "claim" in q
    ):

        return "MONTHLY_CLAIMS"

    # ========================================================
    # YEARLY CLAIMS
    # ========================================================

    if (
        (
            "each year" in q
            or "per year" in q
            or "yearly" in q
            or "annual" in q
            or "by year" in q
        )
        and "claim" in q
    ):

        return "YEARLY_CLAIMS"

    # ========================================================
    # CLAIM STATUS
    # ========================================================

    if (
        "claim status" in q
        or "claims by status" in q
        or "status distribution" in q
    ):

        return "CLAIMS_BY_STATUS"

    # ========================================================
    # FACILITY LEVEL
    # ========================================================

    if (
        "facility level" in q
        or "facilities by level" in q
        or "claims by facility level" in q
    ):

        return "CLAIMS_BY_FACILITY_LEVEL"

    # ========================================================
    # SUBTYPE
    # ========================================================

    if (
        (
            "claim subtype" in q
            or "subtype" in q
        )
        and "claim" in q
    ):

        return "CLAIMS_BY_SUBTYPE"

    # ========================================================
    # AVERAGE CLAIM AMOUNT
    # ========================================================

    if (
        (
            "average" in q
            or "mean" in q
        )
        and (
            "claim amount" in q
            or "claim value" in q
            or "amount per claim" in q
        )
    ):

        return "AVERAGE_AMOUNT"

    # ========================================================
    # TOTAL CLAIM AMOUNT
    # ========================================================

    if (
        (
            "total" in q
            or "sum" in q
        )
        and (
            "claim amount" in q
            or "claim value" in q
            or "amount" in q
        )
    ):

        return "TOTAL_AMOUNT"

    # ========================================================
    # SIMPLE CLAIM COUNT
    # ========================================================

    if (
        "how many claims" in q
        or "number of claims" in q
        or "claim count" in q
        or "count of claims" in q
    ):

        return "COUNT"

    # ========================================================
    # COMPLEX FALLBACK
    # ========================================================

    return "COMPLEX"


# ============================================================
# QUESTION ANALYSIS
# ============================================================

def analyze_question(
    question: str,
) -> dict:

    started = time.perf_counter()

    if (
        not question
        or not str(question).strip()
    ):

        return {
            "question": "",
            "normalized_question": "",
            "intent": "UNKNOWN",
            "county": None,
            "counties": [],
            "time_window": None,
            "limit": 10,
            "analysis_seconds": 0.0,
        }

    question = str(
        question
    ).strip()

    normalized_question = normalize_question(
        question
    )

    intent = detect_intent(
        normalized_question
    )

    county = extract_county(
        normalized_question
    )

    counties = extract_counties(
        normalized_question
    )

    time_window = extract_time_window(normalized_question)

    limit = extract_limit(
        normalized_question
    )

    return {
        "question": question,
        "normalized_question": normalized_question,
        "intent": intent,
        "county": county,
        "counties": counties,
        "time_window": time_window,
        "limit": limit,
        "analysis_seconds": round(
            time.perf_counter() - started,
            4,
        ),
    }


# ============================================================
# SQL STRING HELPERS
# ============================================================

def quote_sql_string(
    value: str,
) -> str:

    return (
        "'"
        + value.replace(
            "'",
            "''",
        )
        + "'"
    )


def build_county_in_clause(
    counties: list[str],
) -> str:

    return ", ".join(
        quote_sql_string(county)
        for county in counties
    )


# ============================================================
# SQL BUILDERS
# ============================================================

def build_sql(
    intent: str,
    county: Optional[str] = None,
    counties: Optional[list[str]] = None,
    limit: int = 10,
    time_window: Optional[str] = None,
) -> Optional[str]:

    if counties is None:
        counties = []

    if county and not counties:
        counties = [county]

    # ========================================================
    # SINGLE COUNTY FILTER
    # ========================================================

    filters = []

    if county:
        filters.append(
            "county = "
            + quote_sql_string(county)
        )

    time_filter = ""
    if time_window:
        interval_match = re.fullmatch(
            r"([1-9]\d{0,3}) (minute|hour|day|week|month|year)",
            time_window,
        )
        if interval_match:
            amount, unit = interval_match.groups()
            time_filter = (
                "creation >= NOW() - "
                f"INTERVAL '{amount} {unit}' "
                "AND creation <= NOW()"
            )
            filters.append(time_filter)

    county_filter = (
        "WHERE " + " AND ".join(filters)
        if filters
        else ""
    )

    # ========================================================
    # CLIENTS WITH MULTIPLE CLAIMS ON THE SAME DAY
    # ========================================================

    if intent == "CLIENTS_MULTIPLE_CLAIMS_SAME_DAY":

        return f"""
SELECT
    client,
    creation::date AS claim_date,
    COUNT(*) AS claim_count
FROM claims
WHERE creation IS NOT NULL
  AND client IS NOT NULL
GROUP BY client, creation::date
HAVING COUNT(*) > 1
ORDER BY claim_count DESC
LIMIT {limit}
""".strip()

    if intent == "CLIENTS_MULTIPLE_CLAIMS":

        return """
WITH client_claim_counts AS (
    SELECT
        client,
        COUNT(*) AS claim_count
    FROM claims
    WHERE client IS NOT NULL
    GROUP BY client
    HAVING COUNT(*) > 1
)
SELECT
    COUNT(*) AS clients_with_multiple_claims
FROM client_claim_counts
""".strip()

    # ========================================================
    # COUNT
    # ========================================================

    if intent == "COUNT":

        return f"""
SELECT
    COUNT(*) AS claim_count
FROM claims
{county_filter}
""".strip()

    # ========================================================
    # AVERAGE
    # ========================================================

    if intent == "AVERAGE_AMOUNT":

        return f"""
SELECT
    ROUND(
        AVG(claim_amount),
        2
    ) AS average_claim_amount,
    COUNT(*) AS total_claims
FROM claims
{county_filter}
""".strip()

    # ========================================================
    # TOTAL
    # ========================================================

    if intent == "TOTAL_AMOUNT":

        return f"""
SELECT
    ROUND(
        COALESCE(
            SUM(claim_amount),
            0
        ),
        2
    ) AS total_claim_amount,
    COUNT(*) AS claim_count
FROM claims
{county_filter}
""".strip()

    # ========================================================
    # CANCELLED COUNT
    # ========================================================

    if intent == "CANCELLED_COUNT":

        filters = ["is_cancelled = TRUE"]
        if county:
            filters.append(
                "county = " + quote_sql_string(county)
            )
        if time_filter:
            filters.append(time_filter)

        return f"""
SELECT
    COUNT(*) AS cancelled_claims
FROM claims
WHERE {" AND ".join(filters)}
""".strip()

    # ========================================================
    # CANCELLED PERCENTAGE
    # ========================================================

    if intent == "CANCELLED_PERCENTAGE":

        filters = []
        if county:
            filters.append(
                "county = " + quote_sql_string(county)
            )
        if time_filter:
            filters.append(time_filter)

        where_clause = (
            "WHERE " + " AND ".join(filters)
            if filters
            else ""
        )

        return f"""
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
        / NULLIF(
            COUNT(*),
            0
        ),
        2
    ) AS cancellation_rate

FROM claims

{where_clause}
""".strip()

    # ========================================================
    # COMPARE CLAIM COUNT
    # ========================================================

    if intent == "COMPARE_CLAIM_COUNT":

        if len(counties) < 2:
            return None

        county_values = build_county_in_clause(
            counties
        )

        return f"""
SELECT
    county,
    COUNT(*) AS claim_count
FROM claims
WHERE county IN ({county_values})
GROUP BY county
ORDER BY claim_count DESC
""".strip()

    # ========================================================
    # COMPARE AVERAGE AMOUNT
    # ========================================================

    if intent == "COMPARE_AVERAGE_AMOUNT":

        if len(counties) < 2:
            return None

        county_values = build_county_in_clause(
            counties
        )

        return f"""
SELECT
    county,

    ROUND(
        AVG(claim_amount),
        2
    ) AS average_claim_amount,

    COUNT(*) AS total_claims

FROM claims

WHERE county IN ({county_values})

GROUP BY county

ORDER BY average_claim_amount DESC
""".strip()

    # ========================================================
    # COMPARE TOTAL AMOUNT
    # ========================================================

    if intent == "COMPARE_TOTAL_AMOUNT":

        if len(counties) < 2:
            return None

        county_values = build_county_in_clause(
            counties
        )

        return f"""
SELECT
    county,

    ROUND(
        COALESCE(
            SUM(claim_amount),
            0
        ),
        2
    ) AS total_claim_amount,

    COUNT(*) AS claim_count

FROM claims

WHERE county IN ({county_values})

GROUP BY county

ORDER BY total_claim_amount DESC
""".strip()

    # ========================================================
    # COMPARE CANCELLATION RATE
    # ========================================================

    if intent == "COMPARE_CANCELLATION_RATE":

        if len(counties) < 2:
            return None

        county_values = build_county_in_clause(
            counties
        )

        return f"""
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
        / NULLIF(
            COUNT(*),
            0
        ),
        2
    ) AS cancellation_rate

FROM claims

WHERE county IN ({county_values})

GROUP BY county

ORDER BY cancellation_rate DESC
""".strip()

    # ========================================================
    # TOP COUNTIES
    # ========================================================

    if intent == "TOP_COUNTIES":

        return f"""
SELECT
    county,
    COUNT(*) AS claim_count
FROM claims
WHERE county IS NOT NULL
GROUP BY county
ORDER BY claim_count DESC
LIMIT {limit}
""".strip()

    # ========================================================
    # BOTTOM COUNTIES
    # ========================================================

    if intent == "BOTTOM_COUNTIES":

        return f"""
SELECT
    county,
    COUNT(*) AS claim_count
FROM claims
WHERE county IS NOT NULL
GROUP BY county
ORDER BY claim_count ASC
LIMIT {limit}
""".strip()

    # ========================================================
    # AVERAGE BY COUNTY
    # ========================================================

    if intent == "AVERAGE_BY_COUNTY":

        return """
SELECT
    county,

    ROUND(
        AVG(claim_amount),
        2
    ) AS average_claim_amount

FROM claims

WHERE county IS NOT NULL

GROUP BY county

ORDER BY average_claim_amount DESC
""".strip()

    # ========================================================
    # TOTAL AMOUNT BY COUNTY
    # ========================================================

    if intent == "TOTAL_AMOUNT_BY_COUNTY":

        return """
SELECT
    county,

    ROUND(
        COALESCE(
            SUM(claim_amount),
            0
        ),
        2
    ) AS total_claim_amount,

    COUNT(*) AS claim_count

FROM claims

WHERE county IS NOT NULL
AND claim_amount IS NOT NULL

GROUP BY county

ORDER BY total_claim_amount DESC
""".strip()

    # ========================================================
    # TOP FACILITIES BY AMOUNT
    # ========================================================

    if intent == "TOP_FACILITIES_AMOUNT":

        return f"""
SELECT
    facility AS facility_name,

    ROUND(
        COALESCE(
            SUM(claim_amount),
            0
        ),
        2
    ) AS total_claim_amount

FROM claims

WHERE facility IS NOT NULL
AND claim_amount IS NOT NULL
AND claim_amount > 0

GROUP BY facility

ORDER BY total_claim_amount DESC

LIMIT {limit}
""".strip()

    # ========================================================
    # TOP FACILITIES BY COUNT
    # ========================================================

    if intent == "TOP_FACILITIES_COUNT":

        return f"""
SELECT
    facility AS facility_name,
    COUNT(*) AS claim_count

FROM claims

WHERE facility IS NOT NULL

GROUP BY facility

ORDER BY claim_count DESC

LIMIT {limit}
""".strip()

    # ========================================================
    # TOP INSURER
    # ========================================================

    if intent == "TOP_INSURER":

        return """
SELECT
    insurer,
    COUNT(*) AS claim_count

FROM claims

WHERE insurer IS NOT NULL

GROUP BY insurer

ORDER BY claim_count DESC

LIMIT 1
""".strip()

    # ========================================================
    # TOTAL BY INSURER
    # ========================================================

    if intent == "TOTAL_AMOUNT_BY_INSURER":

        return """
SELECT
    insurer,

    ROUND(
        COALESCE(
            SUM(claim_amount),
            0
        ),
        2
    ) AS total_claim_amount,

    COUNT(*) AS claim_count

FROM claims

WHERE insurer IS NOT NULL
AND claim_amount IS NOT NULL

GROUP BY insurer

ORDER BY total_claim_amount DESC
""".strip()

    # ========================================================
    # TOP INSURERS (by amount / by count)
    # ========================================================

    if intent == "TOP_INSURERS_AMOUNT":

        return f"""
SELECT
    insurer,
    ROUND(COALESCE(SUM(claim_amount),0),2) AS total_claim_amount,
    COUNT(*) AS claim_count
FROM claims
WHERE insurer IS NOT NULL
  AND claim_amount IS NOT NULL
GROUP BY insurer
ORDER BY total_claim_amount DESC
LIMIT {limit}
""".strip()

    if intent == "TOP_INSURERS_COUNT":

        return f"""
SELECT
    insurer,
    COUNT(*) AS claim_count
FROM claims
WHERE insurer IS NOT NULL
GROUP BY insurer
ORDER BY claim_count DESC
LIMIT {limit}
""".strip()

    # ========================================================
    # CLAIM TYPES
    # ========================================================

    if intent == "CLAIM_TYPES":

        return """
SELECT
    claim_type,
    COUNT(*) AS claim_count

FROM claims

WHERE claim_type IS NOT NULL

GROUP BY claim_type

ORDER BY claim_count DESC
""".strip()

    # ========================================================
    # MONTHLY CLAIMS
    # ========================================================

    if intent == "MONTHLY_CLAIMS":

        return """
SELECT
    TO_CHAR(
        creation,
        'YYYY-MM'
    ) AS claim_month,

    COUNT(*) AS claims_count

FROM claims

WHERE creation IS NOT NULL

GROUP BY TO_CHAR(
    creation,
    'YYYY-MM'
)

ORDER BY claim_month ASC
""".strip()

    # ========================================================
    # YEARLY CLAIMS
    # ========================================================

    if intent == "YEARLY_CLAIMS":

        return """
SELECT
    EXTRACT(
        YEAR FROM creation
    )::INTEGER AS claim_year,

    COUNT(*) AS claims_count

FROM claims

WHERE creation IS NOT NULL

GROUP BY EXTRACT(
    YEAR FROM creation
)

ORDER BY claim_year ASC
""".strip()
    # ========================================================
    # COUNT LAST MONTH / YEAR
    # ========================================================

    if intent == "COUNT_LAST_MONTH":

        return """
SELECT
    COUNT(*) AS claim_count
FROM claims
WHERE creation >= (date_trunc('month', CURRENT_DATE) - INTERVAL '1 month')
  AND creation < date_trunc('month', CURRENT_DATE)
""".strip()

    if intent == "COUNT_LAST_YEAR":

        return """
SELECT
    COUNT(*) AS claim_count
FROM claims
WHERE creation >= (date_trunc('year', CURRENT_DATE) - INTERVAL '1 year')
  AND creation < date_trunc('year', CURRENT_DATE)
""".strip()

    # ========================================================
    # STATUS
    # ========================================================

    if intent == "CLAIMS_BY_STATUS":

        return """
SELECT
    claim_status,
    COUNT(*) AS claim_count

FROM claims

WHERE claim_status IS NOT NULL

GROUP BY claim_status

ORDER BY claim_count DESC
""".strip()

    # ========================================================
    # FACILITY LEVEL
    # ========================================================

    if intent == "CLAIMS_BY_FACILITY_LEVEL":

        return """
SELECT
    facility_level,
    COUNT(*) AS claim_count

FROM claims

WHERE facility_level IS NOT NULL

GROUP BY facility_level

ORDER BY claim_count DESC
""".strip()

    # ========================================================
    # SUBTYPE
    # ========================================================

    if intent == "CLAIMS_BY_SUBTYPE":

        return """
SELECT
    claim_subtype,
    COUNT(*) AS claim_count

FROM claims

WHERE claim_subtype IS NOT NULL

GROUP BY claim_subtype

ORDER BY claim_count DESC
""".strip()

    # ========================================================
    # AVERAGE CLAIM AMOUNT
    # ========================================================

    if intent == "AVERAGE_AMOUNT":

        return f"""
SELECT
    ROUND(
        AVG(claim_amount),
        2
    ) AS average_claim_amount,
    COUNT(*) AS total_claims
FROM claims
{county_filter}
""".strip()

    # ========================================================
    # ROBUST ANOMALY ANALYSIS
    # ========================================================
    #
    # IMPORTANT DESIGN:
    #
    # We DO NOT use raw claim volume directly as a dominant
    # anomaly signal.
    #
    # Instead:
    #
    #   1. Average claim amount
    #      -> robust deviation from county median.
    #
    #   2. Log claim volume
    #      -> reduces the effect of huge counties.
    #
    #   3. Cancellation rate
    #      -> included only if it actually varies.
    #
    #   4. Median absolute deviation (MAD)
    #      -> much more resistant to extreme counties than
    #         ordinary mean/stddev.
    #
    #   5. The final score is the average of ACTIVE metrics.
    #
    # This prevents a county such as Nairobi from being labelled
    # anomalous merely because it has a large population/volume.
    #
    # ========================================================

    if intent == "ANOMALY_ANALYSIS":

        return f"""
WITH county_stats AS (

    SELECT

        county,

        COUNT(*) AS claim_count,

        AVG(claim_amount) AS average_claim_amount,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY claim_amount
            ) AS median_claim_amount,

        ROUND(
            CAST(
                COALESCE(
                    SUM(claim_amount),
                    0
                ) AS numeric
            ),
            2
        ) AS total_claim_amount,

        COUNT(*) FILTER (
            WHERE is_cancelled = TRUE
        ) AS cancelled_claims,

        100.0 *
        COUNT(*) FILTER (
            WHERE is_cancelled = TRUE
        )
        / NULLIF(
            COUNT(*),
            0
        ) AS cancellation_rate

    FROM claims

    WHERE county IS NOT NULL

    GROUP BY county
),

eligible AS (

    SELECT *

    FROM county_stats

    WHERE claim_count >= 100
),

medians AS (

    SELECT

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY average_claim_amount
            ) AS median_avg_amount,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY LN(1 + claim_count)
            ) AS median_log_volume,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY cancellation_rate
            ) AS median_cancellation_rate

    FROM eligible
),

distribution AS (

    SELECT

        m.median_avg_amount,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY ABS(
                    e.average_claim_amount - m.median_avg_amount
                )
            ) AS mad_avg_amount,

        m.median_log_volume,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY ABS(
                    LN(1 + e.claim_count) - m.median_log_volume
                )
            ) AS mad_log_volume,

        m.median_cancellation_rate,

        PERCENTILE_CONT(0.50)
            WITHIN GROUP (
                ORDER BY ABS(
                    e.cancellation_rate - m.median_cancellation_rate
                )
            ) AS mad_cancellation_rate

    FROM eligible e
    CROSS JOIN medians m

    GROUP BY
        m.median_avg_amount,
        m.median_log_volume,
        m.median_cancellation_rate
),

raw_scores AS (

    SELECT

        e.county,

        e.claim_count,

        ROUND(
            CAST(e.average_claim_amount AS numeric),
            2
        ) AS average_claim_amount,

        e.total_claim_amount,

        e.cancelled_claims,

        ROUND(
            CAST(e.cancellation_rate AS numeric),
            2
        ) AS cancellation_rate,

        /* =================================================
           ROBUST AMOUNT SCORE

           0.6745 converts MAD to approximately a standard
           normal scale.

           If MAD is zero, the metric is inactive.
           ================================================= */

        CASE

            WHEN
                d.mad_avg_amount > 0
                AND e.average_claim_amount IS NOT NULL

            THEN

                ABS(
                    e.average_claim_amount
                    -
                    d.median_avg_amount
                )
                /
                (
                    1.4826
                    * d.mad_avg_amount
                )

            ELSE 0

        END AS amount_anomaly,

        /* =================================================
           ROBUST VOLUME SCORE

           Uses log1p(volume), not raw volume.
           ================================================= */

        CASE

            WHEN d.mad_log_volume > 0

            THEN

                ABS(
                    LN(
                        1 + e.claim_count
                    )
                    -
                    d.median_log_volume
                )
                /
                (
                    1.4826
                    * d.mad_log_volume
                )

            ELSE 0

        END AS volume_anomaly,

        /* =================================================
           ROBUST CANCELLATION SCORE
           ================================================= */

        CASE

            WHEN d.mad_cancellation_rate > 0

            THEN

                ABS(
                    e.cancellation_rate
                    -
                    d.median_cancellation_rate
                )
                /
                (
                    1.4826
                    * d.mad_cancellation_rate
                )

            ELSE 0

        END AS cancellation_anomaly,

        /* =================================================
           SIGNED AMOUNT DEVIATION

           Useful to explain WHY a county is unusual.
           ================================================= */

        CASE

            WHEN
                d.mad_avg_amount > 0
                AND e.average_claim_amount IS NOT NULL

            THEN

                (
                    e.average_claim_amount
                    -
                    d.median_avg_amount
                )
                /
                (
                    1.4826
                    * d.mad_avg_amount
                )

            ELSE 0

        END AS amount_zscore,

        /* =================================================
           SIGNED VOLUME DEVIATION
           ================================================= */

        CASE

            WHEN d.mad_log_volume > 0

            THEN

                (
                    LN(
                        1 + e.claim_count
                    )
                    -
                    d.median_log_volume
                )
                /
                (
                    1.4826
                    * d.mad_log_volume
                )

            ELSE 0

        END AS volume_zscore,

        /* =================================================
           SIGNED CANCELLATION DEVIATION
           ================================================= */

        CASE

            WHEN d.mad_cancellation_rate > 0

            THEN

                (
                    e.cancellation_rate
                    -
                    d.median_cancellation_rate
                )
                /
                (
                    1.4826
                    * d.mad_cancellation_rate
                )

            ELSE 0

        END AS cancellation_zscore,

        /* =================================================
           ACTIVE METRICS

           A metric with zero MAD carries no information and
           therefore must NOT affect the final score.
           ================================================= */

        (
            CASE

                WHEN
                    d.mad_avg_amount > 0
                    AND e.average_claim_amount IS NOT NULL

                THEN 1

                ELSE 0

            END

            +

            CASE

                WHEN d.mad_log_volume > 0

                THEN 1

                ELSE 0

            END

            +

            CASE

                WHEN d.mad_cancellation_rate > 0

                THEN 1

                ELSE 0

            END

        ) AS active_metric_count

    FROM eligible e

    CROSS JOIN distribution d
),

scored AS (

    SELECT

        *,

        /* =================================================
           FINAL ROBUST ANOMALY SCORE

           Amount receives the highest weight because unusual
           claim value is generally more informative than raw
           volume.

           Volume receives a lower weight because high-volume
           counties are not automatically suspicious.

           Cancellation receives its own weight when variation
           exists.
           ================================================= */

        (

            amount_anomaly * 0.60

            +

            volume_anomaly * 0.20

            +

            cancellation_anomaly * 0.20

        ) AS raw_anomaly_score

    FROM raw_scores
)

SELECT

    county,

    claim_count,

    average_claim_amount,

    total_claim_amount,

    cancelled_claims,

    cancellation_rate,

    ROUND(
        CAST(raw_anomaly_score AS numeric),
        2
    ) AS anomaly_score,

    ROUND(
        CAST(amount_zscore AS numeric),
        2
    ) AS amount_anomaly_zscore,

    ROUND(
        CAST(volume_zscore AS numeric),
        2
    ) AS volume_anomaly_zscore,

    ROUND(
        CAST(cancellation_zscore AS numeric),
        2
    ) AS cancellation_anomaly_zscore,

    active_metric_count,

    CASE

        WHEN
            ABS(amount_zscore)
            >=
            GREATEST(
                ABS(volume_zscore),
                ABS(cancellation_zscore)
            )

        THEN 'CLAIM_AMOUNT'

        WHEN
            ABS(volume_zscore)
            >=
            ABS(cancellation_zscore)

        THEN 'CLAIM_VOLUME'

        ELSE 'CANCELLATION_RATE'

    END AS primary_anomaly_driver

FROM scored

WHERE active_metric_count > 0

ORDER BY
    anomaly_score DESC

LIMIT {limit}
""".strip()

    # ========================================================
    # FALLBACK
    # ========================================================

    return None


# ============================================================
# MAIN PLANNER
# ============================================================

def needs_multi_entity_reasoning(question: str) -> bool:
    q = normalize_question(question)
    entity_patterns = (
        r"\bclients?\b",
        r"\bcounties?\b",
        r"\bfacilit(?:y|ies)\b",
        r"\binsurers?\b",
        r"\bclaim types?\b",
        r"\bclaim statuses\b",
    )
    entity_count = sum(
        bool(re.search(pattern, q))
        for pattern in entity_patterns
    )
    has_threshold = bool(
        re.search(
            r"\b(?:more than|greater than|at least|over)\s+"
            r"(?:\d+|one|two|three|four|five)\b|"
            r"(?:>|>=)\s*\d+",
            q,
        )
    )
    has_ranking_or_grouping = bool(
        re.search(
            r"\b(?:which|highest|lowest|most|least|top|bottom|rank|by)\b",
            q,
        )
    )
    has_grouped_extreme = bool(
        entity_count >= 1
        and has_ranking_or_grouping
        and re.search(
            r"\b(?:by|per|for each|for every)\s+"
            r"(?:(?:a|the)\s+)?(?:(?:single|individual)\s+)?"
            r"(?:clients?|counties?|facilit(?:y|ies)|insurers?|claim types?|claim statuses)\b",
            q,
        )
        and re.search(r"\b(?:claims?|count|number)\b", q)
    )

    return (
        (
            entity_count >= 2
            and has_threshold
            and has_ranking_or_grouping
        )
        or has_grouped_extreme
    )


def plan_query(
    question: str,
) -> QueryPlan:

    started = time.perf_counter()

    intent = detect_intent(question)

    county = extract_county(question)

    counties = extract_counties(question)

    limit = extract_limit(question)

    # If a comparison intent didn't find two counties, attempt simple
    # "X vs Y" / "X v Y" parsing to recover county names from the
    # user's phrasing before falling back to LLM path.
    if intent.startswith("COMPARE_") and len(counties) < 2:
        m = re.search(r"([A-Za-z ]+?)\s+v(?:s|ersus)?\.?\s+([A-Za-z ]+)", question, flags=re.IGNORECASE)
        if m:
            a = m.group(1).strip()
            b = m.group(2).strip()
            # normalize uppercase county tokens
            counties = [a.upper(), b.upper()]
            logger.debug("Recovered counties for comparison: %s", counties)

    time_window = extract_time_window(question)
    time_aware_intents = {
        "COUNT",
        "AVERAGE_AMOUNT",
        "TOTAL_AMOUNT",
        "CANCELLED_COUNT",
        "CANCELLED_PERCENTAGE",
    }
    needs_model_reasoning = needs_multi_entity_reasoning(question)

    sql = None
    if (
        not needs_model_reasoning
        and (not time_window or intent in time_aware_intents)
    ):
        sql = build_sql(
            intent=intent,
            county=county,
            counties=counties,
            limit=limit,
            time_window=time_window,
        )

    if sql:

        if intent.startswith(
            "COMPARE_"
        ):

            route = "FAST / COMPARISON"

        elif intent == "ANOMALY_ANALYSIS":

            route = "FAST / ANALYTICS"

        elif intent == "CLIENTS_MULTIPLE_CLAIMS_SAME_DAY":

            route = "FAST / SAME_DAY"

        else:

            route = "FAST"

        return QueryPlan(
            route=route,
            intent=intent,
            county=county,
            counties=counties,
            limit=limit,
            sql=sql,
            analysis_seconds=round(
                time.perf_counter() - started,
                4,
            ),
        )

    # ========================================================
    # No deterministic pattern matched. Score complexity so
    # agent.py knows which SQL-generation strategy to use.
    # ========================================================

    complexity_score = score_complexity(
        question
    )

    if needs_model_reasoning:
        complexity_score = max(complexity_score, 8)

    complexity_band = band_for_score(
        complexity_score
    )

    logger.info(
        "Complexity score for %r: %d (%s)",
        question,
        complexity_score,
        complexity_band,
    )

    return QueryPlan(
        route=f"LLM / {complexity_band}",
        intent=intent,
        county=county,
        counties=counties,
        limit=limit,
        sql=None,
        complexity_score=complexity_score,
        complexity_band=complexity_band,
        analysis_seconds=round(
            time.perf_counter() - started,
            4,
        ),
    )


# ============================================================
# RESULT AND COMPLEXITY CACHE
# ============================================================

_QUERY_CACHE = {}
_COMPLEXITY_CACHE = {}


def get_cached_result(
    question: str,
):

    if not question:

        return None

    key = normalize_question(
        str(question)
    )

    return _QUERY_CACHE.get(
        key
    )


def cache_result(
    question: str,
    result: dict,
):

    if not question:
        return

    if result is None:
        return

    key = normalize_question(
        str(question)
    )

    _QUERY_CACHE[key] = result


def get_cached_complexity(
    question: str,
) -> Optional[int]:

    if not question:
        return None

    key = normalize_question(
        str(question)
    )

    return _COMPLEXITY_CACHE.get(
        key
    )


def cache_complexity(
    question: str,
    score: int,
):

    if not question or score is None:
        return

    key = normalize_question(
        str(question)
    )

    _COMPLEXITY_CACHE[key] = int(score)


def clear_complexity_cache():

    _COMPLEXITY_CACHE.clear()


def clear_result_cache():

    _QUERY_CACHE.clear()
    _COMPLEXITY_CACHE.clear()


# ============================================================
# TEST QUESTIONS
# ============================================================

def test_questions():

    questions = [

        # Basic
        "how many claims are in Nairobi",

        "what is the average claim amount in Nairobi",

        "what is the total claim amount in Nairobi",

        # Cancellation
        "what percentage of claims are cancelled",

        "what percentage of Nairobi claims are cancelled",

        "how many cancelled claims are there",

        # Rankings
        "show me the top 10 counties by claims",

        "show me the top 5 counties by claims",

        "show me the bottom 10 counties by claims",

        # County analytics
        "what is the average claim amount by county",

        "what is the total claim amount by county",

        # Facilities
        "what are the top 10 facilities by total claim amount",

        "what are the top 10 facilities by claims",

        # Insurers
        "which insurer has the most claims",

        "what is the total claim amount for each insurer",

        # Classification
        "what are the most common claim types",

        # Time
        "how many claims were created each month",

        "how many claims were created each year",

        # Distribution
        "show claims by status",

        "show claims by facility level",

        "show claim subtypes",

        # Comparisons
        "compare the number of claims between Nairobi and Mombasa",

        "compare the average claim amount between Nairobi and Mombasa",

        "compare the total claim amount between Nairobi and Mombasa",

        "compare the cancellation rate between Nairobi and Mombasa",

        # Anomaly analysis
        "tell me something complicated about unusual claim patterns",

        "show me unusual claim patterns",

        "find anomalous counties",

        "find claim outliers",

        "identify suspicious claim patterns",

        # Same-day repeat claims
        "how many clients have more than one claim",
        "how many clients had multiple claims",
        "give me the clients having most claim counts on the same day",

        "which patients had multiple claims in a day",
    ]

    for question in questions:

        plan = plan_query(
            question
        )

        print("=" * 80)

        print(
            f"QUESTION: {question}"
        )

        print(
            f"ROUTE: {plan.route}"
        )

        print(
            f"INTENT: {plan.intent}"
        )

        print(
            f"COUNTY: {plan.county}"
        )

        print(
            f"COUNTIES: {plan.counties}"
        )

        print(
            f"LIMIT: {plan.limit}"
        )

        if plan.complexity_score is not None:

            print(
                f"COMPLEXITY: {plan.complexity_score} "
                f"({plan.complexity_band})"
            )

        if plan.sql:

            print(
                "SQL:"
            )

            print(
                plan.sql
            )

        print()


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    test_questions()