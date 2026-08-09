"""
Analyser interfaces.

Every measurement the pipeline makes goes through one of these Protocols.
They exist so that "which library estimates the key?" is a configuration
question rather than a rewrite: Essentia, librosa, madmom, a trained model or
a third-party service all satisfy the same three methods, and the pipeline
cannot tell them apart.

Two rules every implementation follows:

* **Never invent a value.** If the audio does not support a claim, return the
  ``unknown()`` estimate with ``reliable=False``. Substituting C major
  because an algorithm shrugged is the single worst thing this system could
  do, because it is indistinguishable from a real answer downstream.
* **Describe yourself.** :meth:`describe` records the algorithm, the library
  version and every parameter that could move a number, so a stored result
  can be checked against the configuration that produced it.

Protocols rather than base classes: an adapter around someone else's engine
should not have to inherit from us.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ..audio.decoder import DecodedAudio
from ..models import (
    EngineInfo,
    KeyEstimate,
    LoudnessMeasurement,
    TempoEstimate,
    TonalSegment,
)

__all__ = [
    "Analyzer",
    "KeyAnalyzer",
    "LoudnessAnalyzer",
    "SegmentKeyAnalyzer",
    "SupportsChromagram",
    "TempoAnalysis",
    "TempoAnalyzer",
]


class Analyzer(Protocol):
    """Common to every analyser: a stable name and a self-description."""

    @property
    def name(self) -> str:
        """Backend identifier, e.g. ``"essentia"``. Recorded in results."""
        ...

    def describe(self) -> EngineInfo: ...


@dataclass(frozen=True, slots=True)
class TempoAnalysis:
    """
    A tempo analyser's full output.

    The beat grid is separate from the summary because it is two orders of
    magnitude larger and most consumers only want the BPM -- but phrase
    detection, cue points and mix-in points all need the grid, so it is
    carried rather than discarded.
    """

    estimate: TempoEstimate
    beats: list[float] = field(default_factory=list)
    downbeats: list[float] | None = None


class KeyAnalyzer(Analyzer, Protocol):
    """Estimates one key for a stretch of audio."""

    def analyze(self, audio: DecodedAudio) -> KeyEstimate: ...


class TempoAnalyzer(Analyzer, Protocol):
    """Estimates tempo and the beat grid."""

    def analyze(self, audio: DecodedAudio) -> TempoAnalysis: ...


class SegmentKeyAnalyzer(Analyzer, Protocol):
    """
    Splits a track into stretches that each read as one key.

    Deliberately a separate interface from :class:`KeyAnalyzer`: the sliding
    window shipped today is the crudest thing that works, and replacing it
    with real structural segmentation (novelty curves, self-similarity,
    phrase boundaries) should not disturb global key detection.
    """

    def analyze(self, audio: DecodedAudio) -> list[TonalSegment]: ...


class LoudnessAnalyzer(Analyzer, Protocol):
    def analyze(self, audio: DecodedAudio) -> LoudnessMeasurement: ...


@runtime_checkable
class SupportsChromagram(Protocol):
    """
    Optional fast path for segmentation.

    A key analyser that can hand over its chromagram lets the segment
    analyser compute the expensive front-end **once** for the whole track and
    then score windows of it, instead of re-running feature extraction for
    every overlapping window. On a 6-minute track with a 30 s window and a
    15 s hop that is 23 extractions collapsed into one.

    Backends that cannot expose their internals simply do not implement it,
    and segmentation falls back to slicing the audio.
    """

    def chromagram(self, audio: DecodedAudio) -> tuple[np.ndarray, float]:
        """``(chroma, frame_rate)`` -- shape (12, frames), frames per second."""
        ...

    def estimate_from_chroma(self, chroma: np.ndarray) -> KeyEstimate:
        """Score an already-extracted chroma block, shape (12, frames)."""
        ...
