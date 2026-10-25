"""Autentikasi API key dan rate limiting sederhana (in-memory, per-instance).

Rate limiter ini cukup untuk satu instance. Untuk multi-instance, pindahkan ke Redis
atau terapkan di API gateway/reverse proxy.
"""

from __future__ import annotations

import hmac
import threading
import time
from collections import deque

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(provided: str | None, valid_keys: list[str], env: str) -> str:
    if not valid_keys:
        if env == "production":  # pragma: no cover - dicegah juga oleh Settings
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API key belum dikonfigurasi")
        return "anonymous-dev"
    if provided and any(hmac.compare_digest(provided, k) for k in valid_keys):
        return f"key:{provided[:6]}"
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED, "API key tidak valid", headers={"WWW-Authenticate": "ApiKey"}
    )


class SlidingWindowRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            window = self._hits.setdefault(identity, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limit:
                retry = max(1, int(60 - (now - window[0])))
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    "Terlalu banyak permintaan, coba lagi nanti.",
                    headers={"Retry-After": str(retry)},
                )
            window.append(now)
            if len(self._hits) > 10_000:  # cegah pertumbuhan memori tak terbatas
                self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < 60}


def client_identity(request: Request, principal: str) -> str:
    if principal != "anonymous-dev":
        return principal
    return request.client.host if request.client else "unknown"
