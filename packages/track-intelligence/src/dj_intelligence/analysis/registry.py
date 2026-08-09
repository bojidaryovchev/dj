"""
Choosing backends.

One place decides which implementation satisfies each analyser interface, so
that swapping engines is a configuration change and every other module keeps
programming against the Protocol.

``auto`` prefers Essentia and falls back to the portable chroma/librosa pair
when it is not importable, which is the normal case on Windows. The fallback
is logged, not warned about in the result: the engine that ran is recorded in
``analysis.key_engine`` either way, and a warning on every single result would
be noise. Naming an engine explicitly and not having it is an error -- if you
asked for Essentia you want to know it is missing, not get a quiet
substitute.
"""

from __future__ import annotations

from ..config import EngineChoice, Settings
from ..errors import BackendUnavailableError
from ..observability import get_logger
from .base import (
    KeyAnalyzer,
    LoudnessAnalyzer,
    SegmentKeyAnalyzer,
    SupportsChromagram,
    TempoAnalyzer,
)
from .key.chroma import ChromaKeyAnalyzer
from .key.essentia import EssentiaKeyAnalyzer, essentia_available
from .key.segmentation import SlidingWindowKeyAnalyzer
from .key.tonal_content import TonalContentGate
from .loudness.ebur128 import EbuR128LoudnessAnalyzer
from .tempo.essentia_tempo import EssentiaTempoAnalyzer
from .tempo.librosa_tempo import LibrosaTempoAnalyzer

__all__ = [
    "available_engines",
    "build_key_analyzer",
    "build_loudness_analyzer",
    "build_segment_analyzer",
    "build_tempo_analyzer",
    "build_tonal_content_gate",
]

log = get_logger(__name__)


def available_engines() -> dict[str, bool]:
    """Which backends this installation can actually run. Used by /ready."""
    return {"chroma": True, "librosa": True, "essentia": essentia_available()}


def build_key_analyzer(settings: Settings) -> KeyAnalyzer:
    choice = settings.key_engine
    if choice is EngineChoice.ESSENTIA or (choice is EngineChoice.AUTO and essentia_available()):
        try:
            return EssentiaKeyAnalyzer(
                profile=settings.key_profile.value,
                sample_rate=settings.sample_rate,
                min_reliability=settings.key_min_reliability,
            )
        except BackendUnavailableError:
            if choice is EngineChoice.ESSENTIA:
                raise
            log.info("engine.fallback", extra={"requested": choice.value, "using": "chroma"})

    return ChromaKeyAnalyzer(
        profile=settings.key_profile.value,
        harmonic_separation=settings.key_harmonic_separation,
        min_reliability=settings.key_min_reliability,
        min_tonal_salience=settings.key_min_tonal_salience,
    )


def build_tempo_analyzer(settings: Settings) -> TempoAnalyzer:
    choice = settings.tempo_engine
    if choice is EngineChoice.ESSENTIA or (choice is EngineChoice.AUTO and essentia_available()):
        try:
            return EssentiaTempoAnalyzer(
                dj_bpm_min=settings.dj_bpm_min,
                dj_bpm_max=settings.dj_bpm_max,
                stability_max_cv=settings.tempo_stability_max_cv,
                min_reliability=settings.tempo_min_reliability,
            )
        except BackendUnavailableError:
            if choice is EngineChoice.ESSENTIA:
                raise
            log.info("engine.fallback", extra={"requested": choice.value, "using": "librosa"})

    return LibrosaTempoAnalyzer(
        dj_bpm_min=settings.dj_bpm_min,
        dj_bpm_max=settings.dj_bpm_max,
        stability_max_cv=settings.tempo_stability_max_cv,
        min_reliability=settings.tempo_min_reliability,
    )


def build_segment_analyzer(key_analyzer: KeyAnalyzer, settings: Settings) -> SegmentKeyAnalyzer:
    return SlidingWindowKeyAnalyzer(
        key_analyzer,
        window_seconds=settings.segment_window_seconds,
        hop_seconds=settings.segment_hop_seconds,
        min_confidence=settings.segment_min_confidence,
    )


def build_loudness_analyzer(settings: Settings) -> LoudnessAnalyzer | None:
    if not settings.loudness_enabled:
        return None
    return EbuR128LoudnessAnalyzer(
        ffmpeg_path=settings.ffmpeg_path,
        max_seconds=settings.max_analysis_seconds or None,
    )


def build_tonal_content_gate(key_analyzer: KeyAnalyzer, settings: Settings) -> TonalContentGate:
    """
    The "is this even tonal?" guard, wired to reuse the active analyser's
    chromagram when it has one. Backends without a chromagram (Essentia) get
    their own chroma pass, which is why this takes the analyser rather than
    assuming one.
    """
    source = (
        key_analyzer
        if isinstance(key_analyzer, SupportsChromagram)
        else ChromaKeyAnalyzer(profile=settings.key_profile.value)
    )
    return TonalContentGate(source, min_salience=settings.key_min_tonal_salience)
