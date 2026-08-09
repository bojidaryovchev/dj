"""
The mapping between audio time and musical time.

This is the primitive the whole preparation layer stands on. Everything that
matters to a DJ — jumping a bar, looping eight, warping a drifting record,
keeping two decks in phase — is a question about where musical position *n*
lives in a particular audio file, and that is exactly what a tempo map
answers, in both directions:

    source seconds  <--->  musical beats

It is deliberately a pure primitive. No audio, no librosa, no configuration,
no measurement. It is constructed *from* measurements and consumed by both
the analysis and the DJ layers, which is why it lives in neither.

Between anchors the mapping is linear; outside them it extrapolates at the
nearest known tempo. Anchors are the detected beats, so the map follows a
drifting record rather than averaging it into a single BPM — which is the
entire point.

Conventions, fixed here and relied on everywhere:

* beat index is zero-based and *musical*, not ordinal. If the tracker misses
  a beat, the following beat is still index 5 rather than index 4.
* bar index is zero-based, counted from the first downbeat.
* beat-in-bar is one-based, because that is how people count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

__all__ = ["BarPosition", "TempoMap"]


@dataclass(frozen=True, slots=True)
class BarPosition:
    """A musical position expressed the way it would be displayed."""

    bar: int
    beat_in_bar: int
    """One-based."""

    fraction: float = 0.0
    """How far past that beat, 0.0-1.0."""

    def __str__(self) -> str:
        return f"bar {self.bar} beat {self.beat_in_bar}"


@dataclass(frozen=True, slots=True)
class TempoMap:
    """
    A piecewise-linear correspondence between source time and musical beats.

    ``times`` and ``beats`` are parallel, both strictly increasing. Construct
    with :meth:`from_beats` for a detected grid or :meth:`constant` for an
    ideal one.
    """

    times: np.ndarray
    beats: np.ndarray
    beats_per_bar: int | None = None
    downbeat_beat: int | None = None
    """Musical beat index that starts bar 0. ``None`` if no phase is known."""

    def __post_init__(self) -> None:
        if self.times.shape != self.beats.shape:
            raise ValueError("times and beats must be the same length")
        if self.times.size < 2:
            raise ValueError("a tempo map needs at least two anchors")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("anchor times must be strictly increasing")
        if not np.all(np.diff(self.beats) > 0):
            raise ValueError("anchor beats must be strictly increasing")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_beats(
        cls,
        times: list[float] | np.ndarray,
        beat_indices: list[int] | np.ndarray | None = None,
        *,
        beats_per_bar: int | None = None,
        downbeat_beat: int | None = None,
    ) -> Self:
        """
        Build from detected beats.

        ``beat_indices`` are the *musical* indices of those beats. Pass them
        when the grid may have gaps — a dropped beat must advance the index by
        two, or every bar after it is off by one. When omitted the beats are
        assumed consecutive.
        """
        beat_times = np.asarray(times, dtype=np.float64)
        indices = (
            np.arange(beat_times.size, dtype=np.float64)
            if beat_indices is None
            else np.asarray(beat_indices, dtype=np.float64)
        )
        return cls(
            times=beat_times,
            beats=indices,
            beats_per_bar=beats_per_bar,
            downbeat_beat=downbeat_beat,
        )

    @classmethod
    def constant(
        cls,
        bpm: float,
        *,
        anchor_time: float = 0.0,
        anchor_beat: int = 0,
        beat_count: int = 2,
        beats_per_bar: int | None = None,
        downbeat_beat: int | None = None,
    ) -> Self:
        """
        An ideal constant-tempo map, pinned so ``anchor_beat`` sits at
        ``anchor_time``. This is the *target* timeline a warp aims at.
        """
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        seconds_per_beat = 60.0 / bpm
        count = max(2, beat_count)
        indices = np.arange(count, dtype=np.float64)
        return cls(
            times=anchor_time + (indices - anchor_beat) * seconds_per_beat,
            beats=indices,
            beats_per_bar=beats_per_bar,
            downbeat_beat=downbeat_beat,
        )

    # -- basic properties --------------------------------------------------

    @property
    def first_time(self) -> float:
        return float(self.times[0])

    @property
    def last_time(self) -> float:
        return float(self.times[-1])

    @property
    def first_beat(self) -> float:
        return float(self.beats[0])

    @property
    def last_beat(self) -> float:
        return float(self.beats[-1])

    @property
    def average_bpm(self) -> float:
        """Mean tempo across the mapped span."""
        span_beats = self.last_beat - self.first_beat
        span_time = self.last_time - self.first_time
        return 60.0 * span_beats / span_time

    def _leading_period(self) -> float:
        return float((self.times[1] - self.times[0]) / (self.beats[1] - self.beats[0]))

    def _trailing_period(self) -> float:
        return float((self.times[-1] - self.times[-2]) / (self.beats[-1] - self.beats[-2]))

    # -- the mapping -------------------------------------------------------

    def times_to_beats(self, times: np.ndarray) -> np.ndarray:
        """Vectorised ``time -> fractional beat``, extrapolating at the ends."""
        query = np.asarray(times, dtype=np.float64)
        mapped = np.interp(query, self.times, self.beats)

        before = query < self.times[0]
        if np.any(before):
            mapped[before] = (
                self.beats[0] + (query[before] - self.times[0]) / self._leading_period()
            )
        after = query > self.times[-1]
        if np.any(after):
            mapped[after] = (
                self.beats[-1] + (query[after] - self.times[-1]) / self._trailing_period()
            )
        return np.asarray(mapped, dtype=np.float64)

    def beats_to_times(self, beats: np.ndarray) -> np.ndarray:
        """Vectorised ``fractional beat -> time``, extrapolating at the ends."""
        query = np.asarray(beats, dtype=np.float64)
        mapped = np.interp(query, self.beats, self.times)

        before = query < self.beats[0]
        if np.any(before):
            mapped[before] = (
                self.times[0] + (query[before] - self.beats[0]) * self._leading_period()
            )
        after = query > self.beats[-1]
        if np.any(after):
            mapped[after] = (
                self.times[-1] + (query[after] - self.beats[-1]) * self._trailing_period()
            )
        return np.asarray(mapped, dtype=np.float64)

    def time_to_beat(self, time: float) -> float:
        """``91.243 -> 192.42``. Fractional, because musical position is."""
        return float(self.times_to_beats(np.array([time]))[0])

    def beat_to_time(self, beat: float) -> float:
        """``192 -> 91.107``. Accepts fractional beats."""
        return float(self.beats_to_times(np.array([beat]))[0])

    def local_bpm(self, time: float) -> float:
        """Tempo of the segment containing ``time``."""
        index = int(np.clip(np.searchsorted(self.times, time) - 1, 0, self.times.size - 2))
        span_time = float(self.times[index + 1] - self.times[index])
        span_beats = float(self.beats[index + 1] - self.beats[index])
        return 60.0 * span_beats / span_time

    # -- bars --------------------------------------------------------------

    @property
    def has_bars(self) -> bool:
        return self.beats_per_bar is not None and self.downbeat_beat is not None

    def _require_bars(self) -> tuple[int, int]:
        if self.beats_per_bar is None or self.downbeat_beat is None:
            raise ValueError(
                "this tempo map has no bar phase; downbeat detection did not "
                "establish one, so bar positions are not available"
            )
        return self.beats_per_bar, self.downbeat_beat

    def beat_to_bar(self, beat: float) -> BarPosition:
        """Musical beat -> bar and position within it."""
        beats_per_bar, downbeat = self._require_bars()
        offset = beat - downbeat
        whole = np.floor(offset)
        bar = int(np.floor(whole / beats_per_bar))
        beat_in_bar = int(whole - bar * beats_per_bar) + 1
        return BarPosition(bar=bar, beat_in_bar=beat_in_bar, fraction=float(offset - whole))

    def bar_to_beat(self, bar: int, beat_in_bar: int = 1) -> float:
        """Bar and one-based position -> musical beat index."""
        beats_per_bar, downbeat = self._require_bars()
        if not 1 <= beat_in_bar <= beats_per_bar:
            raise ValueError(f"beat_in_bar must be 1..{beats_per_bar}, got {beat_in_bar}")
        return float(downbeat + bar * beats_per_bar + (beat_in_bar - 1))

    def bar_to_time(self, bar: int, beat_in_bar: int = 1) -> float:
        return self.beat_to_time(self.bar_to_beat(bar, beat_in_bar))

    def time_to_bar(self, time: float) -> BarPosition:
        return self.beat_to_bar(self.time_to_beat(time))

    def bar_count(self, until: float | None = None) -> int:
        """Complete bars from bar 0 up to ``until`` (default: the last anchor)."""
        beats_per_bar, downbeat = self._require_bars()
        end_beat = self.time_to_beat(self.last_time if until is None else until)
        return max(0, int(np.floor((end_beat - downbeat) / beats_per_bar)))
