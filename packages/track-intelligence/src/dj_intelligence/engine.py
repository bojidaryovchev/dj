"""
The single entry point both the CLI and the API use.

There is exactly one implementation of "analyse a track" in this codebase and
this module holds the handle to it. The CLI formats its output and the API
serialises it, but neither has an analysis path of its own -- if they did,
they would drift, and the two would start disagreeing about the same file.

The pipeline is cached per configuration because building it can be
expensive (Essentia algorithm construction, librosa's numba warm-up) and
because a server analysing a queue of tracks should pay that once.
"""

from __future__ import annotations

from pathlib import Path

from .analysis.pipeline import AnalysisPipeline
from .config import Settings, get_settings
from .models import TrackAnalysis

__all__ = ["analyze", "get_pipeline", "reset_pipeline"]

_pipelines: dict[str, AnalysisPipeline] = {}


def get_pipeline(settings: Settings | None = None) -> AnalysisPipeline:
    """A pipeline for these settings, built once and reused."""
    resolved = settings or get_settings()
    fingerprint = resolved.analysis_fingerprint
    if fingerprint not in _pipelines:
        _pipelines[fingerprint] = AnalysisPipeline(resolved)
    return _pipelines[fingerprint]


def reset_pipeline() -> None:
    """Drop cached pipelines. For tests that vary configuration."""
    _pipelines.clear()


def analyze(
    path: Path | str,
    *,
    settings: Settings | None = None,
    display_name: str | None = None,
) -> TrackAnalysis:
    """Analyse an audio file and return the canonical result."""
    return get_pipeline(settings).analyze(path, display_name=display_name)
