from cachetools import TTLCache
from threading import Lock

# Simple in-memory TTL cache for LLM results
_CACHE = TTLCache(maxsize=1024, ttl=300)
_LOCK = Lock()


def get_cache(key):
    with _LOCK:
        return _CACHE.get(key)


def set_cache(key, value):
    with _LOCK:
        _CACHE[key] = value


def make_key(question, history):
    # history can be None or list of dicts; use last items
    h = tuple((item.get("role"), item.get("content")) for item in (history or [])[-6:])
    return (question, h)
