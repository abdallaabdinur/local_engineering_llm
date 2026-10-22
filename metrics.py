import time
import threading
from prometheus_client import Counter, Histogram, start_http_server, REGISTRY


# Start Prometheus metrics server on port 8000 in background (only once)
_metrics_server_started = False


def _start_metrics_server(port=8000):
    global _metrics_server_started
    if _metrics_server_started:
        return
    try:
        start_http_server(port)
        _metrics_server_started = True
    except Exception:
        # best-effort; don't fail imports
        pass


_thread = threading.Thread(target=_start_metrics_server, args=(8000,), daemon=True)
_thread.start()


# Helper to reuse existing collectors if already registered (safe for reloads)
def _existing_collector(name: str):
    names = getattr(REGISTRY, "_names_to_collectors", None)
    if names and name in names:
        return names[name]
    return None


def _safe_counter(name: str, doc: str):
    existing = _existing_collector(name)
    if existing is not None:
        return existing
    return Counter(name, doc)


def _safe_histogram(name: str, doc: str):
    existing = _existing_collector(name)
    if existing is not None:
        return existing
    return Histogram(name, doc)


# Metrics (use safe constructors to avoid DuplicateTimeseries on reload)
LLM_CALLS = _safe_counter("claims_agent_llm_calls_total", "Total LLM calls")
LLM_ERRORS = _safe_counter("claims_agent_llm_errors_total", "Total LLM errors")
SQL_EXECUTIONS = _safe_counter("claims_agent_sql_executions_total", "Total SQL executions")
SQL_ERRORS = _safe_counter("claims_agent_sql_errors_total", "Total SQL errors")
LLM_LATENCY = _safe_histogram("claims_agent_llm_seconds", "LLM call latency seconds")
SQL_LATENCY = _safe_histogram("claims_agent_sql_seconds", "SQL execution latency seconds")
SQL_CACHE_HITS = _safe_counter("claims_agent_sql_cache_hits_total", "SQL result cache hits")
SQL_CACHE_MISSES = _safe_counter("claims_agent_sql_cache_misses_total", "SQL result cache misses")

# Routing metrics
DIRECT_ROUTED = _safe_counter("claims_agent_direct_routed_total", "Queries routed to deterministic planner / direct SQL")
LLM_ROUTED = _safe_counter("claims_agent_llm_routed_total", "Queries routed to LLM for SQL generation")
DIRECT_FAILURES = _safe_counter("claims_agent_direct_failures_total", "Direct routing failures that fell back to LLM")
ROUTE_LATENCY = _safe_histogram("claims_agent_route_seconds", "Time spent deciding and executing route")
