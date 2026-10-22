import pandas as pd
import plotly.express as px


# ============================================================
# VALUE FORMATTING
# ============================================================

def format_value(value, fmt="number"):
    """
    Format a KPI value for display.
    """

    if value is None:
        return "N/A"

    try:
        number = float(value)

        if fmt == "currency":
            return f"KES {number:,.2f}"

        if fmt == "percentage":
            return f"{number:,.2f}%"

        if fmt == "decimal":
            return f"{number:,.2f}"

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.2f}"

    except (TypeError, ValueError):
        return str(value)


# ============================================================
# COLUMN HELPERS
# ============================================================

def is_numeric_column(rows, column_index):
    """
    Determine whether a result column is numeric.
    """

    if not rows:
        return False

    valid_values = 0
    numeric_values = 0

    for row in rows:

        if column_index >= len(row):
            continue

        value = row[column_index]

        if value is None:
            continue

        valid_values += 1

        try:
            float(value)
            numeric_values += 1

        except (TypeError, ValueError):
            pass

    if valid_values == 0:
        return False

    return numeric_values == valid_values


def find_numeric_columns(columns, rows):
    """
    Return all numeric columns.
    """

    numeric_columns = []

    for index, column in enumerate(columns):

        if is_numeric_column(
            rows,
            index,
        ):
            numeric_columns.append(column)

    return numeric_columns


def find_category_column(
    columns,
    numeric_columns,
):
    """
    Find the first non-numeric column.
    """

    for column in columns:

        if column not in numeric_columns:
            return column

    return None


# ============================================================
# QUESTION HELPERS
# ============================================================

def question_title(question):
    """
    Convert a question into a clean chart title.
    """

    if not question:
        return "Results"

    title = (
        str(question)
        .strip()
        .capitalize()
    )

    return title


# ============================================================
# COLUMN SEMANTIC DETECTION
# ============================================================

def looks_like_percentage(column):
    """
    Determine whether a column represents a percentage/rate.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    percentage_words = [
        "percentage",
        "percentage_rate",
        "rate",
        "percent",
        "ratio",
        "share",
        "proportion",
    ]

    return any(
        word in column_lower
        for word in percentage_words
    )


def looks_like_currency(column):
    """
    Determine whether a column represents monetary value.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    currency_words = [
        "amount",
        "cost",
        "value",
        "price",
        "total_claim_amount",
        "average_claim_amount",
        "minimum_claim_amount",
        "maximum_claim_amount",
        "claim_amount",
        "revenue",
        "payment",
        "paid",
        "exposure",
        "financial",
    ]

    return any(
        word in column_lower
        for word in currency_words
    )


def looks_like_decimal(column):
    """
    Determine whether a column is likely a decimal metric.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    decimal_words = [
        "average",
        "avg",
        "mean",
        "median",
        "stddev",
        "standard_deviation",
        "score",
        "zscore",
        "z_score",
    ]

    return any(
        word in column_lower
        for word in decimal_words
    )


def looks_like_count(column):
    """
    Determine whether a column represents a count.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    count_words = [
        "count",
        "claims",
        "claim_count",
        "total_claims",
        "cancelled_claims",
        "number",
        "volume",
    ]

    return any(
        word in column_lower
        for word in count_words
    )


def looks_like_anomaly_score(column):
    """
    Determine whether a column represents an anomaly score.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    anomaly_words = [
        "anomaly_score",
        "anomaly",
        "outlier_score",
        "risk_score",
        "fraud_score",
        "deviation_score",
    ]

    return any(
        word in column_lower
        for word in anomaly_words
    )


def looks_like_zscore(column):
    """
    Detect statistical z-score columns.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    zscore_words = [
        "zscore",
        "z_score",
        "_z",
    ]

    return any(
        word in column_lower
        for word in zscore_words
    )


def looks_like_time_column(column):
    """
    Determine whether a column likely represents time.
    """

    column_lower = (
        str(column)
        .lower()
        .strip()
    )

    time_words = [
        "month",
        "date",
        "day",
        "week",
        "year",
        "time",
        "period",
        "created",
        "creation",
        "modified",
        "timestamp",
    ]

    return any(
        word in column_lower
        for word in time_words
    )


def looks_like_time_values(rows, column_index):
    """
    Detect common date/time value patterns.
    """

    values = []

    for row in rows:

        if column_index >= len(row):
            continue

        value = row[column_index]

        if value is None:
            continue

        values.append(
            str(value)
        )

    if not values:
        return False

    sample_values = values[:10]

    for value in sample_values:

        # YYYY-MM
        if (
            len(value) == 7
            and value[4] == "-"
            and value[:4].isdigit()
            and value[5:].isdigit()
        ):
            return True

        # YYYY-MM-DD
        if (
            len(value) >= 10
            and value[4] == "-"
            and value[7] == "-"
        ):
            return True

        # ISO timestamp
        if "T" in value and "-" in value:
            return True

    return False


# ============================================================
# RESULT CLASSIFICATION
# ============================================================

def _normalized_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def _column_lookup(columns):
    return {
        _normalized_name(column): column
        for column in columns
    }


def _looks_like_identifier(column):
    normalized = _normalized_name(column)
    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized in {"uuid", "claim_id", "member_id"}
    )


def choose_best_metric_column(columns, numeric_columns):
    """
    Select the most meaningful numeric metric.
    """

    priorities = (
        (
            "anomaly_score",
            "outlier_score",
            "risk_score",
            "fraud_score",
        ),
        (
            "cancellation_rate",
            "cancellation_percentage",
            "percentage",
            "percent",
            "rate",
            "ratio",
            "share",
            "proportion",
        ),
        (
            "total_claim_amount",
            "total_amount",
            "claim_amount",
            "amount",
            "cost",
            "value",
            "price",
            "exposure",
            "revenue",
            "payment",
            "paid",
        ),
        (
            "average_claim_amount",
            "average_amount",
            "avg_claim_amount",
            "mean_claim_amount",
            "average",
            "avg",
            "mean",
            "median",
            "stddev",
            "standard_deviation",
        ),
        (
            "claim_count",
            "total_claims",
            "cancelled_claims",
            "count",
            "number",
            "volume",
        ),
    )

    normalized = [
        (_normalized_name(column), column)
        for column in numeric_columns
    ]

    for group in priorities:
        for normalized_name, original in normalized:
            if any(token in normalized_name for token in group):
                return original

    return numeric_columns[0] if numeric_columns else None


def _choose_category_column(columns, numeric_columns):
    non_numeric = [
        column
        for column in columns
        if column not in numeric_columns
    ]

    if not non_numeric:
        return None

    preferred = (
        "county",
        "insurer",
        "facility",
        "claim_type",
        "claim_status",
        "status",
        "category",
        "type",
        "month",
        "date",
        "year",
        "week",
        "day",
        "period",
    )

    for wanted in preferred:
        for column in non_numeric:
            if _normalized_name(column) == wanted:
                return column

    non_ids = [
        column for column in non_numeric
        if not _looks_like_identifier(column)
    ]

    return non_ids[0] if non_ids else non_numeric[0]


def _is_time_dimension(column, rows, columns):
    index = columns.index(column)

    return (
        looks_like_time_column(column)
        or looks_like_time_values(rows, index)
    )


def classify_result(question, result):
    """
    Determine the strongest defensible visualization for a database result.
    """

    if not result or "error" in result:
        return {"type": "none"}

    columns = result.get("columns", [])
    rows = result.get("rows", [])

    if not columns or not rows:
        return {"type": "none"}

    normalized = _column_lookup(columns)

    # Explicit anomaly/risk score result.
    anomaly_column = (
        normalized.get("anomaly_score")
        or normalized.get("outlier_score")
        or normalized.get("risk_score")
        or normalized.get("fraud_score")
    )

    if anomaly_column:
        numeric_columns = find_numeric_columns(
            columns,
            rows,
        )
        category = _choose_category_column(
            columns,
            numeric_columns,
        )

        if category:
            return {
                "type": "anomaly",
                "title": question_title(question),
                "x": category,
                "y": anomaly_column,
            }

    # Single value = KPI.
    if len(columns) == 1 and len(rows) == 1:
        column = columns[0]

        if looks_like_percentage(column):
            fmt = "percentage"
        elif looks_like_currency(column):
            fmt = "currency"
        elif looks_like_decimal(column):
            fmt = "decimal"
        else:
            fmt = "number"

        title = (
            str(column)
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        return {
            "type": "kpi",
            "title": title,
            "value": rows[0][0],
            "format": fmt,
        }

    numeric_columns = find_numeric_columns(
        columns,
        rows,
    )

    # Two columns.
    if len(columns) == 2:
        first, second = columns

        # Time/value pair regardless of order.
        for category, metric in (
            (first, second),
            (second, first),
        ):
            if (
                metric in numeric_columns
                and _is_time_dimension(category, rows, columns)
            ):
                return {
                    "type": "line",
                    "title": question_title(question),
                    "x": category,
                    "y": metric,
                }

        # Category/numeric pair regardless of order.
        if first in numeric_columns and second not in numeric_columns:
            category, metric = second, first
        elif second in numeric_columns and first not in numeric_columns:
            category, metric = first, second
        elif numeric_columns:
            category = _choose_category_column(
                columns,
                numeric_columns,
            )
            metric = choose_best_metric_column(
                columns,
                numeric_columns,
            )
        else:
            return {"type": "table"}

        if category and metric:
            distinct_values = {
                row[columns.index(category)]
                for row in rows
            }

            # Donut for small non-temporal category distributions.
            if (
                len(distinct_values) <= 8
                and not _is_time_dimension(
                    category,
                    rows,
                    columns,
                )
            ):
                return {
                    "type": "pie",
                    "title": question_title(question),
                    "names": category,
                    "values": metric,
                }

            return {
                "type": "bar",
                "title": question_title(question),
                "x": category,
                "y": metric,
            }

    # Multi-column results.
    if len(columns) >= 3 and numeric_columns:
        category = _choose_category_column(
            columns,
            numeric_columns,
        )

        if category:
            if _is_time_dimension(
                category,
                rows,
                columns,
            ):
                metric = choose_best_metric_column(
                    columns,
                    numeric_columns,
                )

                if metric:
                    return {
                        "type": "line",
                        "title": question_title(question),
                        "x": category,
                        "y": metric,
                    }

            anomaly_column = next(
                (
                    column
                    for column in numeric_columns
                    if any(
                        token in _normalized_name(column)
                        for token in (
                            "anomaly_score",
                            "outlier_score",
                            "risk_score",
                            "fraud_score",
                        )
                    )
                ),
                None,
            )

            if anomaly_column:
                return {
                    "type": "anomaly",
                    "title": question_title(question),
                    "x": category,
                    "y": anomaly_column,
                }

            metric = choose_best_metric_column(
                columns,
                numeric_columns,
            )

            if metric:
                return {
                    "type": "bar",
                    "title": question_title(question),
                    "x": category,
                    "y": metric,
                }

    return {"type": "table"}


# ============================================================
# KPI RENDERER
# ============================================================

def render_kpi(
    st,
    result,
    specification,
):
    """
    Render one KPI.
    """

    rows = result.get(
        "rows",
        [],
    )

    if not rows:
        return

    value = rows[0][0]

    formatted = format_value(
        value,
        specification.get(
            "format",
            "number",
        ),
    )

    st.metric(
        specification.get(
            "title",
            "Metric",
        ),
        formatted,
    )


# ============================================================
# BAR CHART
# ============================================================

def render_bar_chart(
    st,
    result,
    specification,
    key=None,
):
    """
    Render a categorical bar chart.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    x_column = specification.get(
        "x"
    )

    y_column = specification.get(
        "y"
    )

    if (
        x_column not in columns
        or y_column not in columns
    ):
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return

    df[y_column] = pd.to_numeric(
        df[y_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[y_column]
    )

    if df.empty:
        return

    # Keep charts readable.
    if len(df) > 20:
        df = df.head(20)

    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        title=specification.get(
            "title",
            "Results",
        ),
        text_auto=True,
    )

    # Format axes based on metric type.
    if looks_like_currency(y_column):

        fig.update_yaxes(
            tickprefix="KES ",
            separatethousands=True,
        )

    elif looks_like_percentage(y_column):

        fig.update_yaxes(
            ticksuffix="%",
        )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=100,
        ),
        xaxis_tickangle=-45,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


# ============================================================
# ANOMALY CHART
# ============================================================

def render_anomaly_chart(
    st,
    result,
    specification,
    key=None,
):
    """
    Render an anomaly-score chart.

    Highest anomaly scores appear first.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    x_column = specification.get(
        "x",
        "county",
    )

    y_column = specification.get(
        "y",
        "anomaly_score",
    )

    if (
        x_column not in columns
        or y_column not in columns
    ):
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return

    df[y_column] = pd.to_numeric(
        df[y_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[y_column]
    )

    if df.empty:
        return

    df = df.sort_values(
        y_column,
        ascending=False,
    )

    if len(df) > 20:
        df = df.head(20)

    fig = px.bar(
        df,
        x=y_column,
        y=x_column,
        orientation="h",
        title=specification.get(
            "title",
            "County Anomaly Score",
        ),
        text_auto=".2f",
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=40,
        ),
        yaxis={
            "categoryorder": "total ascending"
        },
        xaxis_title="Anomaly Score",
        yaxis_title="County",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


# ============================================================
# HORIZONTAL BAR CHART
# ============================================================

def render_horizontal_bar(
    st,
    result,
    specification,
    key=None,
):
    """
    Render a horizontal bar chart.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    x_column = specification.get(
        "x"
    )

    y_column = specification.get(
        "y"
    )

    if (
        x_column not in columns
        or y_column not in columns
    ):
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return

    df[x_column] = pd.to_numeric(
        df[x_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[x_column]
    )

    if df.empty:
        return

    if len(df) > 20:
        df = df.head(20)

    df = df.sort_values(
        x_column
    )

    fig = px.bar(
        df,
        x=x_column,
        y=y_column,
        orientation="h",
        title=specification.get(
            "title",
            "Results",
        ),
        text_auto=True,
    )

    if looks_like_currency(x_column):

        fig.update_xaxes(
            tickprefix="KES ",
            separatethousands=True,
        )

    elif looks_like_percentage(x_column):

        fig.update_xaxes(
            ticksuffix="%",
        )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=30,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


# ============================================================
# LINE CHART
# ============================================================

def render_line_chart(
    st,
    result,
    specification,
    key=None,
):
    """
    Render a time-series line chart.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    x_column = specification.get(
        "x"
    )

    y_column = specification.get(
        "y"
    )

    if (
        x_column not in columns
        or y_column not in columns
    ):
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return

    df[y_column] = pd.to_numeric(
        df[y_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[y_column]
    )

    if df.empty:
        return

    # Try to convert dates.
    try:

        converted = pd.to_datetime(
            df[x_column],
            errors="coerce",
        )

        if converted.notna().any():

            df[x_column] = converted

    except Exception:
        pass

    # Sort time-series chronologically.
    try:

        df = df.sort_values(
            x_column
        )

    except Exception:
        pass

    fig = px.line(
        df,
        x=x_column,
        y=y_column,
        markers=True,
        title=specification.get(
            "title",
            "Trend",
        ),
    )

    if looks_like_currency(y_column):

        fig.update_yaxes(
            tickprefix="KES ",
            separatethousands=True,
        )

    elif looks_like_percentage(y_column):

        fig.update_yaxes(
            ticksuffix="%",
        )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        margin=dict(
            l=20,
            r=20,
            t=70,
            b=50,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


# ============================================================
# PIE CHART
# ============================================================

def render_pie_chart(
    st,
    result,
    specification,
    key=None,
):
    """
    Render a pie/donut chart.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    names_column = specification.get(
        "names"
    )

    values_column = specification.get(
        "values"
    )

    if (
        names_column not in columns
        or values_column not in columns
    ):
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    if df.empty:
        return

    df[values_column] = pd.to_numeric(
        df[values_column],
        errors="coerce",
    )

    df = df.dropna(
        subset=[values_column]
    )

    if df.empty:
        return

    # Avoid unreadable pie charts.
    if len(df) > 10:
        df = df.head(10)

    fig = px.pie(
        df,
        names=names_column,
        values=values_column,
        title=specification.get(
            "title",
            "Distribution",
        ),
        hole=0.35,
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
    )


# ============================================================
# TABLE RENDERER
# ============================================================

def render_table(
    st,
    result,
    key=None,
):
    """
    Render the raw database result as a table.
    """

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    df = pd.DataFrame(
        rows,
        columns=columns,
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        key=key,
    )


# ============================================================
# MAIN VISUALIZATION RENDERER
# ============================================================

def render_visualization(
    st,
    result,
    specification,
    key=None,
):
    """
    Render KPI/chart/table based on a specification.

    `key` should be a string that is unique across every chart
    rendered in the same Streamlit run (e.g. derived from the
    message index in chat history), so Streamlit doesn't collide
    auto-generated element IDs when two charts end up with
    identical parameters.
    """

    if not result:
        return

    if "error" in result:
        return

    rows = result.get(
        "rows",
        [],
    )

    columns = result.get(
        "columns",
        [],
    )

    if not rows or not columns:
        return

    if not specification:
        return

    visualization_type = (
        specification.get(
            "type"
        )
    )

    # ========================================================
    # KPI
    # ========================================================

    if visualization_type == "kpi":

        render_kpi(
            st,
            result,
            specification,
        )

        return

    # ========================================================
    # ANOMALY
    # ========================================================

    if visualization_type == "anomaly":

        render_anomaly_chart(
            st,
            result,
            specification,
            key=key,
        )

        return

    # ========================================================
    # BAR
    # ========================================================

    if visualization_type == "bar":

        render_bar_chart(
            st,
            result,
            specification,
            key=key,
        )

        return

    # ========================================================
    # HORIZONTAL BAR
    # ========================================================

    if visualization_type in (
        "horizontal_bar",
        "bar_horizontal",
    ):

        render_horizontal_bar(
            st,
            result,
            specification,
            key=key,
        )

        return

    # ========================================================
    # LINE
    # ========================================================

    if visualization_type == "line":

        render_line_chart(
            st,
            result,
            specification,
            key=key,
        )

        return

    # ========================================================
    # PIE
    # ========================================================

    if visualization_type == "pie":

        render_pie_chart(
            st,
            result,
            specification,
            key=key,
        )

        return

    # ========================================================
    # TABLE
    # ========================================================

    if visualization_type == "table":

        render_table(
            st,
            result,
            key=key,
        )

        return

    # ========================================================
    # NONE
    # ========================================================

    if visualization_type == "none":
        return

    # ========================================================
    # UNKNOWN TYPE
    # ========================================================

    render_table(
        st,
        result,
        key=key,
    )


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def visualize_result(
    st,
    question,
    result,
    specification=None,
    render=True,
    key=None,
):
    """
    Classify and optionally render a database result.

    Set render=False when the caller only needs a specification.

    `key`, when rendering, is forwarded to `render_visualization`
    so the underlying chart/table widget gets a unique Streamlit
    element ID. Pass something unique per call site (e.g. an index
    into chat history) to avoid StreamlitDuplicateElementId errors
    when this function is invoked more than once per run.
    """

    if specification is None:
        specification = classify_result(
            question,
            result,
        )

    if render:
        if st is None:
            raise ValueError(
                "st is required when render=True"
            )

        render_visualization(
            st,
            result,
            specification,
            key=key,
        )

    return specification