"""FastAPI application factory.

Jalankan: ``uvicorn cekdulu.api.app:create_app --factory --host 0.0.0.0 --port 8000``
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cekdulu import __version__
from cekdulu.api.container import Container, build_container
from cekdulu.api.routes import ops, v1
from cekdulu.api.security import SlidingWindowRateLimiter
from cekdulu.config import Settings, get_settings
from cekdulu.logging_setup import configure_logging, request_id_var
from cekdulu.observability import HTTP_LATENCY

logger = logging.getLogger("cekdulu.api")


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container or build_container(settings)
        logger.info("startup", extra={"env": settings.env, "version": __version__})
        yield
        app.state.container.repo.close()
        logger.info("shutdown")

    app = FastAPI(
        title="CekDulu API",
        version=__version__,
        description="Triase penipuan keuangan: analisis teks, gambar/QRIS, dan panduan golden hour.",
        lifespan=lifespan,
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url=None,
    )
    app.state.rate_limiter = SlidingWindowRateLimiter(settings.rate_limit_per_minute)
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(rid[:64])
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            logger.exception("unhandled_error", extra={"path": request.url.path})
            response = JSONResponse({"detail": "Terjadi kesalahan internal."}, status_code=500)
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            HTTP_LATENCY.labels(
                request.method, getattr(route, "path", "unmatched"), str(status_code)
            ).observe(elapsed)
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid[:64]
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    app.include_router(ops)
    app.include_router(v1)
    return app
