"""Simple in-memory TTL cache for scrape-backed API endpoints."""

import threading
import time
from functools import wraps


class TTLCache:
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            value, expires = item
            if time.time() >= expires:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key, value, ttl_seconds):
        with self._lock:
            self._store[key] = (value, time.time() + ttl_seconds)

    def clear(self):
        with self._lock:
            self._store.clear()


cache = TTLCache()


def cached(ttl_seconds, key_fn=None):
    """Decorator for functions with hashable / stringifiable args."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if key_fn:
                key = key_fn(*args, **kwargs)
            else:
                key = f"{fn.__module__}.{fn.__name__}:{args}:{sorted(kwargs.items())}"
            hit = cache.get(key)
            if hit is not None:
                return hit
            value = fn(*args, **kwargs)
            cache.set(key, value, ttl_seconds)
            return value
        return wrapper
    return decorator
