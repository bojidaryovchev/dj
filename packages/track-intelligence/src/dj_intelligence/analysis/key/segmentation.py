"""
Time-windowed key analysis.

A global key is a summary, and summaries lie about tracks that move: a
breakdown that lifts to the relative major, a remix that modulates for the
last eight bars, a mashup that was never in one key. Knowing *where* the key
changes is what lets a DJ mix into the right part of a track rather than the
track as a whole.

What this is: a sliding window over the track, the same key analyser run on
each window, runs of identical results merged into segments. It is the
crudest thing that works and it is honest about that -- boundaries land on
the window grid, not on musical events.

What it is not: structural segmentation. Real boundaries come from novelty
detection over a self-similarity matrix, and finding them is a different
problem from labelling them. That is why this sits behind
:class:`~dj_intelligence.analysis.base.SegmentKeyAnalyzer` -- a better
implementation replaces this file and nothing else.

Cost: with a chroma-capable analyser the expensive front end runs **once**
for the whole track and each window is a matrix-vector product over a slice
of it. Backends that cannot expose a chromagram get audio slices instead, and
pay per window.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...audio.decoder import DecodedAudio
from ...models import EngineInfo, KeyEstimate, TonalSegment
from ...music.notes import Mode
from ..base import KeyAnalyzer, SupportsChromagram

__all__ = ["SlidingWindowKeyAnalyzer"]


@dataclass(frozen=True, slots=True)
class _Window:
    start: float
    end: float
    estimate: KeyEstimate

    @property
    def label(self) -> tuple[int, Mode] | None:
        """What this window is merged on: ``None`` means "unknown here"."""
        if not self.estimate.reliable or self.estimate.pitch_class is None:
            return None
        assert self.estimate.mode is not None
        return (self.estimate.pitch_class, self.estimate.mode)


class SlidingWindowKeyAnalyzer:
    def __init__(
        self,
        key_analyzer: KeyAnalyzer,
        *,
        window_seconds: float = 30.0,
        hop_seconds: float = 15.0,
        min_confidence: float = 0.3,
    ) -> None:
        if hop_seconds <= 0 or window_seconds <= 0:
            raise ValueError("window and hop must be positive")
        self._key_analyzer = key_analyzer
        self._window_seconds = window_seconds
        self._hop_seconds = hop_seconds
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return f"sliding_window[{self._key_analyzer.name}]"

    def describe(self) -> EngineInfo:
        inner = self._key_analyzer.describe()
        return EngineInfo(
            name=self.name,
            algorithm=f"sliding window over {inner.algorithm}",
            library_version=inner.library_version,
            parameters={
                "window_seconds": self._window_seconds,
                "hop_seconds": self._hop_seconds,
                "min_confidence": self._min_confidence,
            },
        )

    def analyze(self, audio: DecodedAudio) -> list[TonalSegment]:
        duration = audio.duration_seconds
        if duration < self._window_seconds:
            # Too short to say anything a global key does not already say.
            return []
        windows = (
            self._windows_via_chroma(audio, duration)
            if isinstance(self._key_analyzer, SupportsChromagram)
            else self._windows_via_slices(audio, duration)
        )
        return self._merge(windows, duration)

    # -- window scoring ----------------------------------------------------

    def _window_bounds(self, duration: float) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        start = 0.0
        while start + self._window_seconds <= duration + 1e-9:
            bounds.append((start, start + self._window_seconds))
            start += self._hop_seconds
        # Keep the tail if the last window would otherwise drop more than a
        # hop of the track -- an outro is exactly where a key change hides.
        if bounds and bounds[-1][1] < duration - self._hop_seconds * 0.5:
            bounds.append((max(0.0, duration - self._window_seconds), duration))
        return bounds

    def _windows_via_chroma(self, audio: DecodedAudio, duration: float) -> list[_Window]:
        analyzer: SupportsChromagram = self._key_analyzer  # type: ignore[assignment]
        chroma, frame_rate = analyzer.chromagram(audio)
        frames = chroma.shape[1]

        windows = []
        for start, end in self._window_bounds(duration):
            first = max(0, int(start * frame_rate))
            last = min(frames, round(end * frame_rate))
            if last - first < 2:
                continue
            windows.append(
                _Window(
                    start,
                    end,
                    self._threshold(analyzer.estimate_from_chroma(chroma[:, first:last])),
                )
            )
        return windows

    def _windows_via_slices(self, audio: DecodedAudio, duration: float) -> list[_Window]:
        return [
            _Window(
                start, end, self._threshold(self._key_analyzer.analyze(audio.window(start, end)))
            )
            for start, end in self._window_bounds(duration)
        ]

    def _threshold(self, estimate: KeyEstimate) -> KeyEstimate:
        """
        Apply the segment confidence floor.

        Segments use their own, usually stricter, threshold than the global
        key: 30 seconds of audio is far less evidence than six minutes, so a
        score that would be acceptable for a whole track is not enough to
        claim a key change.
        """
        if estimate.reliable and estimate.confidence < self._min_confidence:
            return estimate.model_copy(update={"reliable": False})
        return estimate

    # -- merging -----------------------------------------------------------

    def _merge(self, windows: list[_Window], duration: float) -> list[TonalSegment]:
        """
        Collapse runs of identically-labelled windows into segments.

        Windows overlap, so a run's raw bounds would overlap the next run's.
        Boundaries are placed at the midpoint between the end of one run and
        the start of the next, and the first and last segments are stretched
        to the ends of the track, so the segments tile the timeline exactly.
        """
        if not windows:
            return []

        runs: list[list[_Window]] = [[windows[0]]]
        for window in windows[1:]:
            if window.label == runs[-1][-1].label:
                runs[-1].append(window)
            else:
                runs.append([window])

        segments: list[TonalSegment] = []
        for index, run in enumerate(runs):
            start = 0.0 if index == 0 else segments[-1].end_seconds
            if index == len(runs) - 1:
                end = duration
            else:
                end = (run[-1].end + runs[index + 1][0].start) / 2.0
            end = max(end, start)

            confidences = [w.estimate.confidence for w in run]
            mean_confidence = sum(confidences) / len(confidences)
            representative = max(run, key=lambda w: w.estimate.confidence).estimate
            labelled = run[0].label is not None

            segments.append(
                TonalSegment(
                    start_seconds=round(start, 3),
                    end_seconds=round(end, 3),
                    pitch_class=representative.pitch_class if labelled else None,
                    mode=representative.mode if labelled else None,
                    confidence=round(mean_confidence, 4),
                    reliable=labelled,
                )
            )
        return segments
