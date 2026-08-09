"""
The FastAPI application.

Assembly only: middleware, error translation, routers. The app knows nothing
about how a key is estimated, which is the point -- ``POST /v1/tracks/analyze``
calls the same :func:`dj_intelligence.engine.analyze` the CLI does.

Errors are translated once, here, so that every failure a client can provoke
comes back with a stable machine-readable code and the request id, rather
than as an anonymous 500.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..config import Settings, get_settings
from ..errors import (
    AudioDecodeError,
    AudioIngestError,
    AudioTooShortError,
    BackendUnavailableError,
    EmptyAudioError,
    FileTooLargeError,
    ToolNotFoundError,
    UnsupportedFormatError,
)
from ..observability import configure_logging, get_logger, new_request_id, request_id_var
from ..version import ANALYSIS_VERSION
from .routes import dj, health, tracks

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Exception -> (status, stable error code). Ordered most specific first;
# lookup walks the MRO so a subclass never falls through to a 500.
# Integer literals rather than starlette's constants: the names for 413 and
# 422 were renamed and the old spellings now emit deprecation warnings, and
# an error table is not worth breaking across versions over.
_ERROR_MAP: list[tuple[type[Exception], int, str]] = [
    (FileTooLargeError, 413, "file_too_large"),  # Content Too Large
    (UnsupportedFormatError, 415, "unsupported_format"),  # Unsupported Media Type
    (EmptyAudioError, 422, "empty_audio"),  # Unprocessable Content
    (AudioTooShortError, 422, "audio_too_short"),
    (AudioDecodeError, 422, "decode_failed"),
    (AudioIngestError, 422, "ingest_failed"),
    (ToolNotFoundError, 503, "ffmpeg_unavailable"),  # Service Unavailable
    (BackendUnavailableError, 503, "backend_unavailable"),
]


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Bind a request id for the life of the request.

    Accepts a client-supplied ``X-Request-ID`` so a trace survives a proxy,
    and always echoes the one in use -- a caller reporting "this track failed"
    can then be matched to the exact log lines.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming[:64] if incoming else new_request_id()
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


def _error_response(exc: Exception) -> JSONResponse:
    for exception_type, http_status, code in _ERROR_MAP:
        if isinstance(exc, exception_type):
            return JSONResponse(
                status_code=http_status,
                content={
                    "error": code,
                    "detail": str(exc),
                    "request_id": request_id_var.get(),
                },
            )
    raise exc  # pragma: no cover - not one of ours; let the server log it


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "service.started",
            extra={
                "analysis_version": ANALYSIS_VERSION,
                "key_engine": resolved.key_engine.value,
                "tempo_engine": resolved.tempo_engine.value,
                "fingerprint": resolved.analysis_fingerprint,
            },
        )
        yield
        log.info("service.stopped")

    app = FastAPI(
        title="DJ Track Intelligence",
        version=ANALYSIS_VERSION,
        summary="Key, Camelot, tempo and tonal structure for DJ libraries.",
        lifespan=lifespan,
    )
    # Routes read configuration from here, so an app built with explicit
    # settings actually enforces them. See api/dependencies.py.
    app.state.settings = resolved
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(AudioIngestError)
    async def _ingest_error(request: Request, exc: AudioIngestError) -> JSONResponse:
        log.warning("request.rejected", extra={"error": type(exc).__name__, "detail": str(exc)})
        return _error_response(exc)

    @app.exception_handler(ToolNotFoundError)
    async def _tool_error(request: Request, exc: ToolNotFoundError) -> JSONResponse:
        log.error("request.failed", extra={"error": type(exc).__name__, "detail": str(exc)})
        return _error_response(exc)

    @app.exception_handler(BackendUnavailableError)
    async def _backend_error(request: Request, exc: BackendUnavailableError) -> JSONResponse:
        log.error("request.failed", extra={"error": type(exc).__name__, "detail": str(exc)})
        return _error_response(exc)

    app.include_router(health.router)
    app.include_router(tracks.router)
    app.include_router(dj.router)
    return app
