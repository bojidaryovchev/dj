"""
Structured logging and stage timing.

Two output modes: ``json`` for anything that ships logs somewhere, ``console``
for a human at a terminal. Both carry the same fields, so a developer reading
the console form is reading the same records an aggregator would.

Every log line inside an analysis carries the request id, so a slow or
failing track can be reconstructed from a single grep. What never goes into a
log line is audio -- only its metadata.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

__all__ = [
    "configure_logging",
    "get_logger",
    "new_request_id",
    "request_id_var",
    "stage_timer",
    "use_request_id",
]

request_id_var: ContextVar[str | None] = ContextVar("dj_request_id", default=None)

_PACKAGE_LOGGER = __name__.split(".")[0]

# Attributes the stdlib puts on every record; anything else a caller attached
# via extra= is ours and belongs in the structured payload.
_STANDARD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


@contextmanager
def use_request_id(request_id: str | None = None) -> Iterator[str]:
    """Bind a request id for the duration of the block."""
    resolved = request_id or new_request_id()
    token = request_id_var.set(resolved)
    try:
        yield resolved
    finally:
        request_id_var.reset(token)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _STANDARD_ATTRS and key != "request_id"
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if request_id := getattr(record, "request_id", None):
            payload["request_id"] = request_id
        payload.update(_extras(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    _COLOURS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[38;5;203m",
    }
    _RESET = "\033[0m"
    _DIM = "\033[38;5;244m"

    def __init__(self, *, colour: bool) -> None:
        super().__init__()
        self._colour = colour

    def _paint(self, text: str, code: str) -> str:
        return f"{code}{text}{self._RESET}" if self._colour else text

    def format(self, record: logging.LogRecord) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = self._paint(f"{record.levelname:<7}", self._COLOURS.get(record.levelname, ""))
        head = f"{self._paint(stamp, self._DIM)} {level} {record.getMessage()}"

        parts = []
        if request_id := getattr(record, "request_id", None):
            parts.append(f"req={request_id}")
        parts += [f"{key}={value}" for key, value in _extras(record).items()]
        if parts:
            head += "  " + self._paint(" ".join(parts), self._DIM)
        if record.exc_info:
            head += "\n" + self.formatException(record.exc_info)
        return head


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    """
    Install the root handler. Idempotent -- calling it twice (CLI then API in
    the same process, or once per test) replaces rather than stacks handlers.

    ``level`` applies to **this package**, not to the world. Turning the root
    logger up to DEBUG also turns on numba's bytecode dumps, which bury a
    stage timing under a few hundred lines of disassembly and make ``-v``
    useless for the thing it exists to do. Third-party loggers inherit
    WARNING from the root instead, so `-v` shows our stages and nothing else.
    """
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        JsonFormatter() if fmt == "json" else ConsoleFormatter(colour=sys.stderr.isatty())
    )
    handler.addFilter(_RequestIdFilter())

    resolved = level.upper()

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    # Records are filtered at the logger they were created on, so a quiet root
    # silences third-party libraries without touching ours below.
    root.setLevel(max(logging.getLevelName(resolved), logging.WARNING))

    logging.getLogger(_PACKAGE_LOGGER).setLevel(resolved)

    # uvicorn installs its own colourised handlers; let ours own the output.
    # Its access log is INFO, so it needs an explicit level to survive a
    # WARNING root.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
        logger.setLevel(min(logging.getLevelName(resolved), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def stage_timer(
    logger: logging.Logger,
    stage: str,
    **fields: Any,
) -> Iterator[MutableMapping[str, Any]]:
    """
    Time one pipeline stage and log its start, end and duration.

    Yields a mapping the caller can add result fields to; whatever is in it
    when the block exits is logged with the completion record. On failure the
    duration is still logged, which is how a stage that hangs on one codec
    gets noticed.

        with stage_timer(log, "key") as fields:
            result = analyser.analyze(audio)
            fields["confidence"] = result.confidence
    """
    extra_fields: dict[str, Any] = {}
    logger.debug("stage.started", extra={"stage": stage, **fields})
    started = time.perf_counter()
    try:
        yield extra_fields
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.exception(
            "stage.failed",
            extra={"stage": stage, "duration_ms": round(elapsed_ms, 1), **fields},
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - started) * 1000
        extra_fields.setdefault("duration_ms", round(elapsed_ms, 1))
        logger.info("stage.completed", extra={"stage": stage, **fields, **extra_fields})
