"""
Container and stream metadata, via ffprobe.

Probing before decoding does three things: it validates that the file really
is audio regardless of what its extension claims, it gives an exact duration
without decoding, and it records what the source actually was -- once the
decoder has normalised everything to mono float32, the original codec and
sample rate are gone.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import AudioDecodeError, UnsupportedFormatError
from .ffmpeg import redact_path, resolve_tool, run_tool

__all__ = ["SourceInfo", "probe"]

_PROBE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """What the file was before we normalised it."""

    duration_seconds: float | None
    sample_rate: int | None
    channels: int | None
    codec: str | None
    container: str | None
    bit_rate_bps: int | None


def _first_audio_stream(streams: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((s for s in streams if s.get("codec_type") == "audio"), None)


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed >= 0 else None  # reject NaN


def probe(path: Path, ffprobe_path: str = "ffprobe") -> SourceInfo:
    """
    Read stream metadata. Raises if the file is not decodable audio.

    Extension is never trusted: a ``.mp3`` that is really a JPEG fails here,
    before a decoder is handed it.
    """
    executable = resolve_tool(ffprobe_path)
    try:
        completed = run_tool(
            executable,
            [
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout=_PROBE_TIMEOUT_SECONDS,
            capture_stdout=True,
            nostdin=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"ffprobe timed out reading {path.name}") from exc
    except OSError as exc:
        raise AudioDecodeError(f"could not run ffprobe: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = redact_path(detail[-1], path) if detail else "unknown error"
        raise UnsupportedFormatError(f"ffprobe could not read {path.name}: {reason}")

    try:
        parsed = json.loads(completed.stdout or b"{}")
    except ValueError as exc:
        raise AudioDecodeError(f"ffprobe returned unparseable output for {path.name}") from exc

    stream = _first_audio_stream(parsed.get("streams") or [])
    if stream is None:
        raise UnsupportedFormatError(f"{path.name} contains no audio stream")

    container = parsed.get("format", {})
    duration = _as_float(stream.get("duration")) or _as_float(container.get("duration"))

    return SourceInfo(
        duration_seconds=duration,
        sample_rate=_as_int(stream.get("sample_rate")),
        channels=_as_int(stream.get("channels")),
        codec=stream.get("codec_name"),
        container=container.get("format_name"),
        bit_rate_bps=_as_int(stream.get("bit_rate")) or _as_int(container.get("bit_rate")),
    )
