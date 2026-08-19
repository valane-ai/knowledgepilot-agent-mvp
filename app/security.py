"""Small, dependency-free API protection primitives for single-process deployments."""

import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: int = 60) -> None:
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= window_seconds:
                events.popleft()
            if len(events) >= limit:
                raise HTTPException(429, "请求过于频繁，请稍后再试。")
            events.append(now)


rate_limiter = InMemoryRateLimiter()


def require_api_key(request: Request) -> None:
    """Enable API-key protection only when KNOWLEDGEPILOT_API_KEYS is configured."""
    configured = [key.strip() for key in os.getenv("KNOWLEDGEPILOT_API_KEYS", "").split(",") if key.strip()]
    if not configured:
        return
    supplied = request.headers.get("X-API-Key", "")
    if not any(secrets.compare_digest(supplied, expected) for expected in configured):
        raise HTTPException(401, "缺少或无效的 API Key。")


def client_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")


def configured_limit(name: str, default: int) -> int:
    try:
        return max(1, min(int(os.getenv(name, str(default))), 10_000))
    except ValueError:
        return default
