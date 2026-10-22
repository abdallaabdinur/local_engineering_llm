import os
import threading

import streamlit as st
import pandas as pd
import plotly.express as px

from agent import ask_agent
from tools import list_tables, run_select
from visualization import visualize_result
from investigation_workflow import build_investigation_cases
from anomaly_investigator import (
    compare_county_to_national,
    investigate_county as investigate_legacy_county,
)
from complex_analytics import run_complex_analysis
from evidence_report import build_evidence_report
from investigation_engine import investigate_county as investigate_anomaly_county
from investigation_report import build_investigation_report
from robust_anomaly import (
    analyze_county_anomalies,
    analyze_peer_claims,
    analyze_temporal_anomalies,
)
from advanced_anomaly import (
    analyze_claim_ml_anomalies,
    analyze_relationship_anomalies,
    analyze_count_dispersion,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Claims Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "investigation_result" not in st.session_state:
    st.session_state.investigation_result = None

if "advanced_analysis_result" not in st.session_state:
    st.session_state.advanced_analysis_result = None


@st.cache_resource
def start_claim_generator():
    """Start one shared ingestion thread for this Streamlit process."""

    enabled = os.getenv(
        "AUTO_START_CLAIM_GENERATOR",
        "true",
    ).strip().lower() not in {"0", "false", "no", "off"}

    if not enabled:
        return {"status": "disabled"}

    if not os.getenv("PG_WRITER_USER") or not os.getenv("PG_WRITER_PASSWORD"):
        return {"status": "missing_credentials"}

    try:
        interval_seconds = float(
            os.getenv("CLAIM_GENERATION_INTERVAL_SECONDS", "10")
        )
    except ValueError:
        interval_seconds = 10.0

    from agentfor import run_live_ingestion

    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_live_ingestion,
        kwargs={
            "interval_seconds": interval_seconds,
            "stop_event": stop_event,
        },
        name="live-claim-ingestion",
        daemon=True,
    )
    thread.start()

    return {
        "status": "running",
        "thread": thread,
        "stop_event": stop_event,
    }


generator_worker = start_claim_generator()


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>

    /* Layout padding only - let Streamlit theme control colors */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Metrics: rounded cards, minimal border and padding. Avoid hard-coded colors so
       the Streamlit light/dark theme styles apply consistently. */
    div[data-testid="stMetric"] {
        border-radius: 14px;
        padding: 18px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 1px 0 rgba(0,0,0,0.02);
        background-clip: padding-box;
    }

    div[data-testid="stMetricLabel"] {
        opacity: 0.85;
        font-size: 0.95rem;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 600;
        font-size: 1.25rem;
    }

    /* Chat messages: rounded with subtle border */
    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        border: 1px solid rgba(0,0,0,0.06);
        margin-bottom: 10px;
        padding: 8px 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE HELPERS
# ============================================================

@st.cache_data(ttl=60)
def get_dashboard_data():
    """
    Load the main dashboard analytics from PostgreSQL.

    Cached for 60 seconds to avoid repeatedly executing the
    same aggregation queries on every Streamlit rerun.
    """

    data = {}

    # ========================================================
    # TOTAL CLAIMS
    # ========================================================

    result = run_select(
        """
        SELECT COUNT(*) AS total_claims
        FROM claims
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["total_claims"] = result["rows"][0][0]

    # ========================================================
    # TOTAL CLAIM AMOUNT
    # ========================================================

    result = run_select(
        """
        SELECT
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount
        FROM claims
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["total_amount"] = result["rows"][0][0]

    # ========================================================
    # CANCELLED CLAIMS
    # ========================================================

    result = run_select(
        """
        SELECT COUNT(*) AS cancelled_claims
        FROM claims
        WHERE is_cancelled = TRUE
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["cancelled_claims"] = result["rows"][0][0]

    # ========================================================
    # CANCELLATION RATE
    # ========================================================

    result = run_select(
        """
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
                / NULLIF(COUNT(*), 0),
                2
            ) AS cancelled_percentage

        FROM claims
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    cancellation_row = result["rows"][0]

    data["cancellation_rate"] = cancellation_row[2]

    # ========================================================
    # COUNTY CLAIMS
    # ========================================================

    result = run_select(
        """
        SELECT
            county,
            COUNT(*) AS claim_count
        FROM claims
        WHERE county IS NOT NULL
        GROUP BY county
        ORDER BY claim_count DESC
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["county_claims"] = result["rows"]

    # ========================================================
    # COUNTY CLAIM AMOUNTS
    # ========================================================

    result = run_select(
        """
        SELECT
            county,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount
        FROM claims
        WHERE county IS NOT NULL
        GROUP BY county
        ORDER BY total_claim_amount DESC
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["county_amounts"] = result["rows"]

    # ========================================================
    # MONTHLY CLAIMS
    # ========================================================

    result = run_select(
        """
        SELECT
            TO_CHAR(creation, 'YYYY-MM')
                AS claim_month,
            COUNT(*) AS claim_count
        FROM claims
        WHERE creation IS NOT NULL
        GROUP BY TO_CHAR(creation, 'YYYY-MM')
        ORDER BY claim_month
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["monthly_claims"] = result["rows"]

    # ========================================================
    # MONTHLY CLAIM AMOUNTS
    # ========================================================

    result = run_select(
        """
        SELECT
            TO_CHAR(creation, 'YYYY-MM')
                AS claim_month,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount
        FROM claims
        WHERE creation IS NOT NULL
        GROUP BY TO_CHAR(creation, 'YYYY-MM')
        ORDER BY claim_month
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["monthly_amounts"] = result["rows"]

    # ========================================================
    # INSURERS
    # ========================================================

    result = run_select(
        """
        SELECT
            insurer,
            COUNT(*) AS claim_count
        FROM claims
        WHERE insurer IS NOT NULL
        GROUP BY insurer
        ORDER BY claim_count DESC
        LIMIT 15
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["insurer_claims"] = result["rows"]

    # ========================================================
    # INSURER AMOUNTS
    # ========================================================

    result = run_select(
        """
        SELECT
            insurer,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount
        FROM claims
        WHERE insurer IS NOT NULL
        GROUP BY insurer
        ORDER BY total_claim_amount DESC
        LIMIT 15
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["insurer_amounts"] = result["rows"]

    # ========================================================
    # CLAIM TYPES
    # ========================================================

    result = run_select(
        """
        SELECT
            claim_type,
            COUNT(*) AS claim_count
        FROM claims
        WHERE claim_type IS NOT NULL
        GROUP BY claim_type
        ORDER BY claim_count DESC
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["claim_types"] = result["rows"]

    # ========================================================
    # CLAIM STATUS
    # ========================================================

    result = run_select(
        """
        SELECT
            claim_status,
            COUNT(*) AS claim_count
        FROM claims
        WHERE claim_status IS NOT NULL
        GROUP BY claim_status
        ORDER BY claim_count DESC
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["claim_status"] = result["rows"]

    # ========================================================
    # FACILITIES
    # ========================================================

    result = run_select(
        """
        SELECT
            facility,
            COUNT(*) AS claim_count,
            COALESCE(SUM(claim_amount), 0)
                AS total_claim_amount
        FROM claims
        WHERE facility IS NOT NULL
        GROUP BY facility
        ORDER BY total_claim_amount DESC
        LIMIT 15
        """
    )

    if "error" in result:
        raise RuntimeError(result["error"])

    data["facilities"] = result["rows"]

    return data


def get_live_feed(limit=25):
    """Read the newest generated claims directly from PostgreSQL."""

    recent = run_select(
        f"""
        SELECT
            claim_id,
            creation,
            county,
            facility_name,
            insurer,
            claim_type,
            claim_amount,
            claim_status,
            claim_verdict,
            is_cancelled
        FROM claims
        ORDER BY creation DESC
        LIMIT {int(limit)}
        """,
        max_rows=limit,
    )

    if recent.get("error"):
        raise RuntimeError(recent["error"])

    activity = run_select(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE creation >= NOW() - INTERVAL '1 minute'
            ) AS claims_last_minute,
            MAX(creation) AS latest_claim_at,
            COUNT(*) AS total_claims
        FROM claims
        """,
        max_rows=1,
    )

    if activity.get("error"):
        raise RuntimeError(activity["error"])

    return {
        "columns": recent.get("columns") or [],
        "rows": recent.get("rows") or [],
        "activity": (activity.get("rows") or [[0, None, 0]])[0],
    }


# ============================================================
# RUN AI QUESTION
# ============================================================

def process_question(question):
    """
    Execute one natural-language database question.

    Returns the complete agent response plus a visualization
    specification derived from the database result.
    """

    history = st.session_state.messages.copy()

    try:

        result = ask_agent(
            question,
            history,
        )

        database_result = result.get("result")
        visualization = None

        if database_result:
            visualization = visualize_result(
                None,
                question,
                database_result,
                render=False,
            )

        return {
            "answer": result.get(
                "answer",
                "No answer generated.",
            ),
            "sql": result.get("sql"),
            "result": database_result,
            "visualization": visualization,
        }

    except Exception as error:

        return {
            "answer": (
                f"Error while processing query: "
                f"{error}"
            ),
            "sql": None,
            "result": None,
            "visualization": None,
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("◈ Claims Intelligence")

    st.caption(
        "Healthcare Claims Analytics"
    )

    st.divider()

    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.subheader("Quick Questions")

    quick_questions = [
        "How many claims are in Nairobi?",
        "Which county has the most claims?",
        "What is the total claim amount in Nairobi?",
        "What is the average claim amount in Nairobi?",
        "Show me the top 10 counties by claims",
        "Which insurer has the most claims?",
        "What are the most common claim types?",
        "What percentage of claims are cancelled?",
        "How many claims were created each month?",
        "What are the top 10 facilities by total claim amount?",
    ]

    for index, quick_question in enumerate(
        quick_questions
    ):

        if st.button(
            quick_question,
            key=f"quick_question_{index}",
            width="stretch",
        ):

            st.session_state.pending_question = (
                quick_question
            )

            st.rerun()

    st.divider()

    # ========================================================
    # DATABASE STATUS
    # ========================================================

    st.subheader("Database")

    try:

        tables = list_tables()

        st.success(
            "PostgreSQL connected"
        )

        st.write(
            f"**{len(tables)} tables available**"
        )

        for table_index, table in enumerate(tables):

            st.caption(
                f"• {table}"
            )

    except Exception as error:

        st.error(
            f"Database error: {error}"
        )

    st.subheader("Synthetic claim generator")
    if generator_worker["status"] == "running":
        if generator_worker["thread"].is_alive():
            st.success("Generating claims in the background")
        else:
            st.error("Claim generator stopped; check the app logs")
    elif generator_worker["status"] == "missing_credentials":
        st.info("Set PG_WRITER_USER and PG_WRITER_PASSWORD to enable generation")
    else:
        st.caption("Auto-start disabled by AUTO_START_CLAIM_GENERATOR")

    st.divider()

    # ========================================================
    # REFRESH ANALYTICS
    # ========================================================

    if st.button(
        "Refresh analytics",
        key="refresh_analytics",
        width="stretch",
    ):

        get_dashboard_data.clear()

        st.rerun()

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "Clear conversation",
        key="clear_conversation",
        width="stretch",
    ):

        st.session_state.messages = []

        st.session_state.pending_question = None

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title(
    "Claims Intelligence"
)

st.caption(
    "Natural-language analytics for your healthcare claims database."
)


# ============================================================
# LOAD DASHBOARD
# ============================================================

try:

    dashboard = get_dashboard_data()

    dashboard_error = None

except Exception as error:

    dashboard = None

    dashboard_error = str(error)


# ============================================================
# LIVE INCOMING CLAIMS FEED
# ============================================================

@st.fragment(run_every="5s")
def render_live_feed():
    st.header("Incoming claims feed")
    st.caption(
        "Fresh rows read from PostgreSQL. Updates every five seconds while this page is open."
    )

    try:
        feed = get_live_feed(limit=25)
        claims_last_minute, latest_claim_at, total_claims = feed["activity"]
        metric_columns = st.columns(3)

        with metric_columns[0]:
            st.metric("Claims in last minute", int(claims_last_minute or 0))

        with metric_columns[1]:
            st.metric("Claims in database", int(total_claims or 0))

        with metric_columns[2]:
            st.metric(
                "Latest claim",
                str(latest_claim_at or "No claims"),
            )

        if feed["rows"]:
            feed_df = pd.DataFrame(
                feed["rows"],
                columns=feed["columns"],
            )
            st.dataframe(
                feed_df,
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No incoming claims have been inserted yet.")
    except Exception as error:
        st.error(f"Live feed unavailable: {error}")


render_live_feed()


# ============================================================
# KPI CARDS
# ============================================================

if dashboard:

    total_claims = dashboard[
        "total_claims"
    ]

    total_amount = dashboard[
        "total_amount"
    ]

    cancelled_claims = dashboard[
        "cancelled_claims"
    ]

    cancellation_rate = dashboard[
        "cancellation_rate"
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Claims",
            f"{total_claims:,}",
        )

    with col2:

        st.metric(
            "Total Claim Amount",
            f"KES {float(total_amount):,.2f}",
        )

    with col3:

        st.metric(
            "Cancelled Claims",
            f"{cancelled_claims:,}",
        )

    with col4:

        st.metric(
            "Cancellation Rate",
            f"{float(cancellation_rate):.2f}%",
        )

elif dashboard_error:

    st.warning(
        f"Analytics unavailable: {dashboard_error}"
    )


st.divider()


# ============================================================
# INVESTIGATION WORKSPACE
# ============================================================

st.header(
    "Investigation workspace"
)

st.caption(
    "Build and review investigation cases from the same claims database."
)

county_options = []

if dashboard:
    county_options = [
        str(row[0])
        for row in dashboard.get("county_claims", [])
        if row and row[0]
    ]

if county_options:
    selected_county = st.selectbox(
        "County",
        county_options,
        key="investigation_county",
    )
else:
    selected_county = st.text_input(
        "County",
        placeholder="For example, NAIROBI",
        key="investigation_county_text",
    )

with st.form("investigation_form"):
    case_limit = st.number_input(
        "Maximum cases",
        min_value=1,
        max_value=500,
        value=25,
        step=1,
    )
    build_submitted = st.form_submit_button(
        "Build investigation cases",
        type="primary",
    )

if build_submitted:
    if not selected_county.strip():
        st.warning("Enter a county before building cases.")
    else:
        with st.spinner("Building investigation cases..."):
            investigation_result = build_investigation_cases(
                selected_county,
                limit=case_limit,
            )

        st.session_state.investigation_result = investigation_result

investigation_result = st.session_state.investigation_result

if investigation_result:
    if investigation_result.get("error"):
        st.error(investigation_result["error"])
    else:
        summary = investigation_result.get("summary") or {}
        summary_columns = st.columns(4)

        with summary_columns[0]:
            st.metric("Cases", investigation_result.get("case_count", 0))

        with summary_columns[1]:
            st.metric("Open", summary.get("open", 0))

        with summary_columns[2]:
            st.metric("Under review", summary.get("under_review", 0))

        with summary_columns[3]:
            st.metric("Escalated", summary.get("escalated", 0))

        cases = investigation_result.get("cases") or []

        if cases:
            case_columns = [
                "case_id",
                "claim_id",
                "county",
                "facility_name",
                "claim_amount",
                "priority",
                "status",
            ]
            case_rows = [
                {
                    column: case.get(column)
                    for column in case_columns
                }
                for case in cases
            ]
            for row, case in zip(case_rows, cases):
                priority = case.get("priority") or {}
                row["priority"] = (
                    priority.get("priority_level")
                    or priority.get("level")
                    or priority.get("priority")
                )
                row["status"] = case.get("investigation_status")
            st.dataframe(
                pd.DataFrame(case_rows),
                width="stretch",
                hide_index=True,
            )


st.divider()


# ============================================================
# ADVANCED ANOMALY ANALYSIS
# ============================================================

st.header(
    "Advanced anomaly analysis"
)

st.caption(
    "Run deterministic evidence and anomaly algorithms directly against claims."
)

analysis_mode = st.selectbox(
    "Analysis mode",
    [
        "Robust county anomaly ranking",
        "Peer-normalized claim scoring",
        "Temporal anomaly detection",
        "Isolation Forest + LOF claim scoring",
        "Client-facility relationship anomalies",
        "Poisson vs Negative Binomial check",
        "Robust county investigation",
        "Evidence report",
        "Legacy county profile",
        "Database-wide complex analytics",
    ],
    key="advanced_analysis_mode",
)

with st.form("advanced_analysis_form"):
    analysis_county = st.text_input(
        "County for county-level analysis",
        value=selected_county or "",
        key="advanced_analysis_county",
    )
    analysis_submitted = st.form_submit_button(
        "Run analysis",
        type="primary",
    )

if analysis_submitted:
    normalized_analysis_county = analysis_county.strip().upper()
    county_modes = {
        "Robust county investigation",
        "Evidence report",
        "Legacy county profile",
    }

    if analysis_mode in county_modes and not normalized_analysis_county:
        st.warning("Enter a county for this analysis mode.")
    else:
        with st.spinner("Running deterministic anomaly analysis..."):
            if analysis_mode == "Robust county anomaly ranking":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_county_anomalies(),
                }
            elif analysis_mode == "Peer-normalized claim scoring":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_peer_claims(),
                }
            elif analysis_mode == "Temporal anomaly detection":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_temporal_anomalies(),
                }
            elif analysis_mode == "Isolation Forest + LOF claim scoring":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_claim_ml_anomalies(),
                }
            elif analysis_mode == "Client-facility relationship anomalies":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_relationship_anomalies(),
                }
            elif analysis_mode == "Poisson vs Negative Binomial check":
                advanced_result = {
                    "mode": analysis_mode,
                    "data": analyze_count_dispersion(),
                }
            elif analysis_mode == "Robust county investigation":
                investigation = investigate_anomaly_county(
                    normalized_analysis_county
                )
                advanced_result = {
                    "mode": analysis_mode,
                    "data": investigation,
                    "report": build_investigation_report(
                        investigation
                    ),
                }
            elif analysis_mode == "Evidence report":
                advanced_result = {
                    "mode": analysis_mode,
                    "report": build_evidence_report(
                        normalized_analysis_county
                    ),
                }
            elif analysis_mode == "Legacy county profile":
                advanced_result = {
                    "mode": analysis_mode,
                    "profile": investigate_legacy_county(
                        normalized_analysis_county
                    ),
                    "national_comparison": compare_county_to_national(
                        normalized_analysis_county
                    ),
                }
            else:
                advanced_result = {
                    "mode": analysis_mode,
                    "data": run_complex_analysis(
                        "Database-wide claims anomaly analysis"
                    ),
                }

        st.session_state.advanced_analysis_result = advanced_result

advanced_result = st.session_state.advanced_analysis_result

if advanced_result:
    st.subheader(advanced_result["mode"])

    if advanced_result["mode"] == "Robust county anomaly ranking":
        data = advanced_result["data"]
        if data.get("error"):
            st.error(data["error"])
        else:
            st.write(data.get("summary"))
            anomaly_rows = data.get("rows") or []
            if anomaly_rows:
                anomaly_df = pd.DataFrame(anomaly_rows)
                component_columns = pd.json_normalize(
                    anomaly_df.pop("components")
                ).add_prefix("component_")
                anomaly_df = pd.concat(
                    [anomaly_df.reset_index(drop=True), component_columns],
                    axis=1,
                )
                st.dataframe(
                    anomaly_df,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(data.get("summary"))

    elif advanced_result["mode"] in {
        "Peer-normalized claim scoring",
        "Temporal anomaly detection",
        "Isolation Forest + LOF claim scoring",
        "Client-facility relationship anomalies",
    }:
        data = advanced_result["data"]
        if data.get("error"):
            st.error(data["error"])
        else:
            st.write(data.get("summary"))
            result_rows = data.get("rows") or []
            if result_rows:
                result_df = pd.DataFrame(result_rows)
                if "components" in result_df:
                    component_columns = pd.json_normalize(
                        result_df.pop("components")
                    ).add_prefix("component_")
                    result_df = pd.concat(
                        [result_df.reset_index(drop=True), component_columns],
                        axis=1,
                    )
                st.dataframe(
                    result_df,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No anomalies met the configured thresholds.")

    elif advanced_result["mode"] == "Poisson vs Negative Binomial check":
        data = advanced_result["data"]
        if data.get("error"):
            st.error(data["error"])
        else:
            st.json(data)

    elif advanced_result["mode"] == "Robust county investigation":
        investigation = advanced_result["data"]
        metrics = investigation.get("metrics") or {}
        metric_columns = st.columns(4)

        with metric_columns[0]:
            st.metric("Risk", investigation.get("risk_level", "UNKNOWN"))
        with metric_columns[1]:
            st.metric("Score", investigation.get("investigation_score", 0))
        with metric_columns[2]:
            st.metric("Claims", int(metrics.get("claim_count", 0)))
        with metric_columns[3]:
            st.metric(
                "Average multiplier",
                f"{float(metrics.get('average_multiplier', 0)):.2f}x",
            )

        if investigation.get("signals"):
            st.write("**Signals**")
            st.json(investigation["signals"])
        if investigation.get("conclusions"):
            st.write("**Conclusions**")
            for conclusion in investigation["conclusions"]:
                st.write(f"- {conclusion}")

    elif advanced_result["mode"] == "Evidence report":
        report = advanced_result["report"]
        st.write(report.get("conclusion") or report.get("summary"))
        st.json(report.get("findings") or report)

    elif advanced_result["mode"] == "Legacy county profile":
        profile = advanced_result["profile"]
        st.json({
            "overview": profile.get("overview"),
            "cancellations": profile.get("cancellations"),
            "national_comparison": advanced_result["national_comparison"],
        })

    else:
        data = advanced_result["data"]
        pipeline = data.get("pipeline") or {}
        st.write(
            f"Completed {pipeline.get('successful_analyses', 0)} of "
            f"{pipeline.get('total_analyses', 0)} analyses."
        )
        st.json(data.get("findings") or {})
        st.json(data.get("summaries") or {})

st.expander("Recommended anomaly algorithms").markdown(
    """
**Recommended order for this claims data:**

- **Robust MAD/IQR scores:** use median, MAD, P90/P95, and log-transformed amounts instead of raw mean and standard deviation.
- **Two-part monetary model:** score zero-claim rate separately from the positive-claim amount distribution.
- **Peer-normalized residuals:** compare claims within claim type, facility, insurer, and facility category before flagging them.
- **Temporal monitoring:** use rolling weekly/monthly baselines; use Poisson or negative-binomial models for claim counts.
- **Concentration signals:** measure facility/insurer exposure share, repeated exact amounts, same-client same-day activity, and bundle/link relationships.
- **Isolation Forest:** use only after building a claim-level feature table; it is useful for multivariate prioritization, not as a fraud probability.

These methods produce investigation-priority signals, not fraud determinations.
"""
)


st.divider()


# ============================================================
# ANALYTICS DASHBOARD
# ============================================================

st.header(
    "Analytics Dashboard"
)


if dashboard:

    (
        county_tab,
        trend_tab,
        insurer_tab,
        type_tab,
        facility_tab,
    ) = st.tabs(
        [
            "County",
            "Claims Trend",
            "Insurers",
            "Claim Types",
            "Facilities",
        ]
    )

    # ========================================================
    # COUNTY TAB
    # ========================================================

    with county_tab:

        county_rows = dashboard[
            "county_claims"
        ]

        county_df = pd.DataFrame(
            county_rows,
            columns=[
                "County",
                "Claims",
            ],
        )

        if not county_df.empty:

            left, right = st.columns(
                [2, 1]
            )

            with left:

                chart_df = county_df.head(15)

                fig = px.bar(
                    chart_df,
                    x="County",
                    y="Claims",
                    title="Top 15 Counties by Claims",
                )

                fig.update_layout(
                    height=500,
                    template="plotly_dark",
                    xaxis_tickangle=-45,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="dashboard_county_claims_bar",
                )

            with right:

                st.subheader(
                    "County Ranking"
                )

                st.dataframe(
                    county_df.head(15),
                    width="stretch",
                    hide_index=True,
                    key="dashboard_county_ranking_table",
                )

        # ----------------------------------------------------
        # COUNTY CLAIM AMOUNT
        # ----------------------------------------------------

        st.subheader(
            "Claim Amount by County"
        )

        amount_rows = dashboard[
            "county_amounts"
        ]

        county_amount_df = pd.DataFrame(
            amount_rows,
            columns=[
                "County",
                "Total Claim Amount",
            ],
        )

        if not county_amount_df.empty:

            chart_df = (
                county_amount_df
                .head(15)
                .sort_values(
                    "Total Claim Amount"
                )
            )

            fig = px.bar(
                chart_df,
                x="Total Claim Amount",
                y="County",
                orientation="h",
                title="Top 15 Counties by Claim Amount",
            )

            fig.update_layout(
                height=550,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_county_amount_bar",
            )

    # ========================================================
    # CLAIMS TREND TAB
    # ========================================================

    with trend_tab:

        monthly_rows = dashboard[
            "monthly_claims"
        ]

        monthly_df = pd.DataFrame(
            monthly_rows,
            columns=[
                "Month",
                "Claims",
            ],
        )

        if not monthly_df.empty:

            fig = px.line(
                monthly_df,
                x="Month",
                y="Claims",
                markers=True,
                title="Claims Created by Month",
            )

            fig.update_layout(
                height=500,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_monthly_claims_line",
            )

            st.dataframe(
                monthly_df,
                width="stretch",
                hide_index=True,
                key="dashboard_monthly_claims_table",
            )

        # ----------------------------------------------------
        # MONTHLY CLAIM AMOUNT
        # ----------------------------------------------------

        st.subheader(
            "Monthly Claim Amount"
        )

        monthly_amount_rows = dashboard[
            "monthly_amounts"
        ]

        monthly_amount_df = pd.DataFrame(
            monthly_amount_rows,
            columns=[
                "Month",
                "Total Claim Amount",
            ],
        )

        if not monthly_amount_df.empty:

            fig = px.area(
                monthly_amount_df,
                x="Month",
                y="Total Claim Amount",
                title="Claim Amount by Month",
            )

            fig.update_layout(
                height=500,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_monthly_amount_area",
            )

    # ========================================================
    # INSURER TAB
    # ========================================================

    with insurer_tab:

        insurer_rows = dashboard[
            "insurer_claims"
        ]

        insurer_df = pd.DataFrame(
            insurer_rows,
            columns=[
                "Insurer",
                "Claims",
            ],
        )

        if not insurer_df.empty:

            fig = px.bar(
                insurer_df.head(10),
                x="Claims",
                y="Insurer",
                orientation="h",
                title="Top 10 Insurers by Claims",
            )

            fig.update_layout(
                height=550,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_insurer_claims_bar",
            )

            st.dataframe(
                insurer_df,
                width="stretch",
                hide_index=True,
                key="dashboard_insurer_claims_table",
            )

        # ----------------------------------------------------
        # INSURER CLAIM AMOUNT
        # ----------------------------------------------------

        st.subheader(
            "Claim Amount by Insurer"
        )

        insurer_amount_rows = dashboard[
            "insurer_amounts"
        ]

        insurer_amount_df = pd.DataFrame(
            insurer_amount_rows,
            columns=[
                "Insurer",
                "Total Claim Amount",
            ],
        )

        if not insurer_amount_df.empty:

            chart_df = (
                insurer_amount_df
                .head(10)
                .sort_values(
                    "Total Claim Amount"
                )
            )

            fig = px.bar(
                chart_df,
                x="Total Claim Amount",
                y="Insurer",
                orientation="h",
                title="Top 10 Insurers by Claim Amount",
            )

            fig.update_layout(
                height=550,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_insurer_amount_bar",
            )

    # ========================================================
    # CLAIM TYPE TAB
    # ========================================================

    with type_tab:

        type_rows = dashboard[
            "claim_types"
        ]

        type_df = pd.DataFrame(
            type_rows,
            columns=[
                "Claim Type",
                "Claims",
            ],
        )

        if not type_df.empty:

            col1, col2 = st.columns(
                [1, 1]
            )

            with col1:

                fig = px.pie(
                    type_df,
                    names="Claim Type",
                    values="Claims",
                    title="Claims by Claim Type",
                )

                fig.update_layout(
                    height=500,
                    template="plotly_dark",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    key="dashboard_claim_type_pie",
                )

            with col2:

                st.dataframe(
                    type_df,
                    width="stretch",
                    hide_index=True,
                    key="dashboard_claim_type_table",
                )

        # ----------------------------------------------------
        # CLAIM STATUS
        # ----------------------------------------------------

        st.subheader(
            "Claim Status"
        )

        status_rows = dashboard[
            "claim_status"
        ]

        status_df = pd.DataFrame(
            status_rows,
            columns=[
                "Claim Status",
                "Claims",
            ],
        )

        if not status_df.empty:

            fig = px.bar(
                status_df,
                x="Claim Status",
                y="Claims",
                title="Claims by Status",
            )

            fig.update_layout(
                height=450,
                template="plotly_dark",
                xaxis_tickangle=-45,
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_claim_status_bar",
            )

    # ========================================================
    # FACILITY TAB
    # ========================================================

    with facility_tab:

        facility_rows = dashboard[
            "facilities"
        ]

        facility_df = pd.DataFrame(
            facility_rows,
            columns=[
                "Facility",
                "Claims",
                "Total Claim Amount",
            ],
        )

        if not facility_df.empty:

            st.subheader(
                "Top Facilities by Claim Amount"
            )

            chart_df = (
                facility_df
                .head(10)
                .sort_values(
                    "Total Claim Amount"
                )
            )

            fig = px.bar(
                chart_df,
                x="Total Claim Amount",
                y="Facility",
                orientation="h",
                title="Top 10 Facilities by Claim Amount",
            )

            fig.update_layout(
                height=550,
                template="plotly_dark",
            )

            st.plotly_chart(
                fig,
                width="stretch",
                key="dashboard_facility_amount_bar",
            )

            st.dataframe(
                facility_df,
                width="stretch",
                hide_index=True,
                key="dashboard_facility_table",
            )


st.divider()


# ============================================================
# AI CHAT
# ============================================================

st.header(
    "Ask the Database"
)

st.caption(
    "Ask questions in natural language. "
    "The AI generates read-only SQL, executes it, "
    "and explains the result."
)


# ============================================================
# PROCESS PENDING QUICK QUESTION
# ============================================================

if st.session_state.pending_question:

    question = st.session_state.pending_question

    st.session_state.pending_question = None

    # --------------------------------------------------------
    # Add user message.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # Execute AI query.
    # --------------------------------------------------------

    with st.spinner(
        "Analyzing the claims database..."
    ):

        result = process_question(
            question
        )

    # --------------------------------------------------------
    # Save assistant response.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sql": result["sql"],
            "result": result["result"],
            "visualization": result.get("visualization"),
        }
    )

    st.rerun()


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message_index, message in enumerate(
    st.session_state.messages
):

    role = message.get(
        "role"
    )

    # ========================================================
    # USER
    # ========================================================

    if role == "user":

        with st.chat_message("user"):

            st.write(
                message.get(
                    "content",
                    "",
                )
            )

    # ========================================================
    # ASSISTANT
    # ========================================================

    elif role == "assistant":

        with st.chat_message("assistant"):

            answer = message.get(
                "content",
                "No answer generated.",
            )

            st.markdown(
                answer
            )

            # ------------------------------------------------
            # GENERATED SQL
            # ------------------------------------------------

            sql = message.get(
                "sql"
            )

            if sql:

                with st.expander(
                    "View generated SQL",
                ):

                    st.code(
                        sql,
                        language="sql",
                    )

            # ------------------------------------------------
            # DATABASE RESULT
            # ------------------------------------------------

            result = message.get(
                "result"
            )

            if result:

                rows = result.get(
                    "rows",
                    [],
                )

                columns = result.get(
                    "columns",
                    [],
                )

                if rows and columns:

                    result_df = pd.DataFrame(
                        rows,
                        columns=columns,
                    )

                    visualization = message.get("visualization")

                    if visualization:
                        st.subheader("Visualization")
                        visualize_result(
                            st,
                            message.get("content", ""),
                            result,
                            specification=visualization,
                            key=f"chart_history_{message_index}",
                        )

                    st.subheader("Data")

                    st.dataframe(
                        result_df,
                        width="stretch",
                        hide_index=True,
                        key=f"table_history_{message_index}",
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about the claims database..."
)


if question:

    # --------------------------------------------------------
    # Add user message.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # A unique index for the assistant message that is about to
    # be appended below (before it's actually appended), so the
    # keys used here can never collide with the history-loop
    # keys on the rerun that follows.
    next_message_index = len(st.session_state.messages)

    # --------------------------------------------------------
    # Execute query.
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing claims..."
        ):

            result = process_question(
                question
            )

        answer = result["answer"]

        sql = result["sql"]

        database_result = result["result"]

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.markdown(
            answer
        )

        # ----------------------------------------------------
        # SQL
        # ----------------------------------------------------

        if sql:

            with st.expander(
                "View generated SQL",
            ):

                st.code(
                    sql,
                    language="sql",
                )

        # ----------------------------------------------------
        # DATABASE RESULT
        # ----------------------------------------------------

        if database_result:

            rows = database_result.get(
                "rows",
                [],
            )

            columns = database_result.get(
                "columns",
                [],
            )

            if rows and columns:

                result_df = pd.DataFrame(
                    rows,
                    columns=columns,
                )

                visualization = result.get("visualization")

                if visualization:
                    st.subheader("Visualization")
                    visualize_result(
                        st,
                        question,
                        database_result,
                        specification=visualization,
                        key=f"chart_live_{next_message_index}",
                    )

                st.subheader("Data")

                st.dataframe(
                    result_df,
                    width="stretch",
                    hide_index=True,
                    key=f"table_live_{next_message_index}",
                )

    # --------------------------------------------------------
    # Save assistant message.
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "result": database_result,
            "visualization": result.get("visualization"),
        }
    )

    st.rerun()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.messages:

    st.info(
        "Start by asking a question such as: "
        "\"How many claims are in Nairobi?\""
    )