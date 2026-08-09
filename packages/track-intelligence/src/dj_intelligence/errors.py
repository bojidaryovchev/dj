"""
Failure taxonomy.

The distinction that matters: *ingest* failures are errors (there is nothing
to analyse, the caller gets a 4xx), while *analysis* failures are results
(the audio was fine, we just could not make a confident claim about it). The
second kind never raises -- it comes back as a null value with
``reliable=false`` and a warning attached, because "we don't know" is a real
answer and silently substituting C major is not.
"""

from __future__ import annotations

__all__ = [
    "AudioDecodeError",
    "AudioIngestError",
    "AudioTooShortError",
    "BackendUnavailableError",
    "DJIntelligenceError",
    "EmptyAudioError",
    "FileTooLargeError",
    "ToolNotFoundError",
    "UnsupportedFormatError",
]


class DJIntelligenceError(Exception):
    """Base class for everything this package raises deliberately."""


class AudioIngestError(DJIntelligenceError):
    """The input could not be turned into samples. Caller's problem."""


class UnsupportedFormatError(AudioIngestError):
    """Container or codec ffmpeg will not decode."""


class AudioDecodeError(AudioIngestError):
    """ffmpeg failed or produced garbage: truncated, corrupt, encrypted."""


class EmptyAudioError(AudioIngestError):
    """Zero decodable samples."""


class AudioTooShortError(AudioIngestError):
    """Decodes, but there is not enough of it to analyse."""


class FileTooLargeError(AudioIngestError):
    """Upload exceeded the configured limit."""


class ToolNotFoundError(DJIntelligenceError):
    """ffmpeg or ffprobe is not installed or not on PATH."""


class BackendUnavailableError(DJIntelligenceError):
    """An analysis backend was requested explicitly but cannot be imported."""
