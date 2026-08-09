"""
Receiving an uploaded file safely.

An upload is hostile until proven otherwise, so:

* **The name is never used as a path.** The temporary file is created by
  ``mkstemp`` in the system temp directory; at most a validated extension is
  carried over. ``../../etc/passwd`` and ``C:\\Windows\\System32\\x.mp3``
  are therefore not expressible.
* **The size limit is enforced while streaming**, not after. Checking the
  length of something you have already written to disk is not a limit.
* **The temporary file is always removed**, including when analysis raises.
* **The format is validated by ffprobe, not by the extension** -- that
  happens downstream in the decoder, and a ``.mp3`` that is really a ZIP is
  rejected there.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

from ..errors import FileTooLargeError

__all__ = ["received_upload", "safe_suffix"]

_CHUNK_BYTES: Final = 1024 * 1024
_SUFFIX_RE: Final = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


def safe_suffix(filename: str | None) -> str:
    """
    The uploaded name's extension, if it is plausibly one.

    ffmpeg detects container formats from content, so this only helps it pick
    a demuxer faster. Anything unusual is dropped rather than sanitised --
    there is no reason to try to rescue a strange extension.
    """
    if not filename:
        return ""
    suffix = Path(filename).suffix
    return suffix.lower() if _SUFFIX_RE.match(suffix) else ""


@asynccontextmanager
async def received_upload(
    read_chunk: Callable[[int], Awaitable[bytes]],
    *,
    filename: str | None,
    max_bytes: int,
) -> AsyncIterator[Path]:
    """
    Stream an upload to a temporary file and yield its path.

    ``read_chunk`` is Starlette's ``UploadFile.read``; taking it as a callable
    keeps this module testable without constructing an ASGI request.
    """
    handle, raw_path = tempfile.mkstemp(prefix="djti-", suffix=safe_suffix(filename))
    path = Path(raw_path)
    written = 0
    try:
        with os.fdopen(handle, "wb") as destination:
            while chunk := await read_chunk(_CHUNK_BYTES):
                written += len(chunk)
                if written > max_bytes:
                    raise FileTooLargeError(
                        f"upload exceeds the {max_bytes} byte limit (DJTI_MAX_UPLOAD_BYTES)"
                    )
                destination.write(chunk)
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()
