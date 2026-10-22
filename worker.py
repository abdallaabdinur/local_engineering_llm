import threading
from concurrent.futures import ThreadPoolExecutor, Future
import time
import uuid
import logging
from typing import Any, Callable

logger = logging.getLogger("claims_agent.worker")

_WORKER_LOCK = threading.Lock()

POOL_SIZE = int(__import__("os").environ.get("WORKER_POOL_SIZE", "4"))

_executor: ThreadPoolExecutor | None = None
_jobs: dict[str, Future] = {}


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    with _WORKER_LOCK:
        if _executor is None:
            logger.info("Starting worker ThreadPoolExecutor (size=%d)", POOL_SIZE)
            _executor = ThreadPoolExecutor(max_workers=POOL_SIZE)
    return _executor


def submit_task(fn: Callable, *args, **kwargs) -> str:
    """Submit a callable to the worker pool. Returns a job id."""
    executor = _get_executor()
    job_id = str(uuid.uuid4())
    future = executor.submit(fn, *args, **kwargs)
    _jobs[job_id] = future
    logger.debug("Submitted job %s", job_id)
    return job_id


def get_task_status(job_id: str) -> str:
    f = _jobs.get(job_id)
    if f is None:
        return "NOT_FOUND"
    if f.running():
        return "RUNNING"
    if f.done():
        return "DONE"
    return "PENDING"


def get_task_result(job_id: str, timeout: float | None = 0.0) -> dict:
    """Return dict with status and result or exception."""
    f = _jobs.get(job_id)
    if f is None:
        return {"status": "NOT_FOUND"}
    if not f.done():
        return {"status": "RUNNING"}
    try:
        res = f.result(timeout=timeout) if timeout else f.result()
        return {"status": "DONE", "result": res}
    except Exception as e:
        logger.exception("Job %s raised", job_id)
        return {"status": "ERROR", "error": str(e)}


def cancel_task(job_id: str) -> bool:
    f = _jobs.get(job_id)
    if f is None:
        return False
    return f.cancel()
