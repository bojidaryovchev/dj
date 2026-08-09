"""
Content hashing.

SHA-256 of the file's bytes, streamed so a 200 MB WAV does not become a
200 MB string. The hash identifies the *file*, not the recording: two rips of
the same track at different bitrates hash differently, which is correct for
"have I already analysed this exact file?" and wrong for "is this the same
song?". Audio fingerprinting is the answer to the second question and is on
the roadmap, not in this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

__all__ = ["CHUNK_BYTES", "sha256_file"]

CHUNK_BYTES: Final = 1024 * 1024


def sha256_file(path: Path | str, *, chunk_bytes: int = CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
