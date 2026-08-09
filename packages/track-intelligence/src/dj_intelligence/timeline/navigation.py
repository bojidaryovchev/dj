"""
Quantized navigation over a tempo map.

Pure timeline maths. Given a musical timeline and a position in it, work out
where "+16 bars", "the next bar line" or "the start of the next phrase"
actually is, in source seconds. No audio, no playback, no state — which is
what makes it testable to the millisecond and reusable by a deck later.

Three separate things live here, and they answer different questions:

``snap``
    "Where is the nearest / next / previous beat or bar from here?"
    Used for seeking: a user drags to 92.131 s and lands on the bar line.

``jump``
    "Where do I end up if I move 16 bars from here?"
    Preserves rhythmic phase: jumping from halfway through beat 3 lands
    halfway through beat 3 of the destination bar, so a loop stays in time.

``schedule``
    "I asked for this now; when should it happen?"
    The timing primitive a deck needs to make a jump land on a bar line
    instead of wherever the user's finger was. It returns times, not audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from .tempo_map import TempoMap

__all__ = [
    "DEFAULT_PHRASE_BARS",
    "Direction",
    "MusicalPosition",
    "Navigator",
    "PhraseWindow",
    "QuantizedAction",
    "Unit",
]

DEFAULT_PHRASE_BARS: Final = 8
# In *units*, not seconds: a position within this fraction of a boundary
# counts as being on it. Without it, a seek that lands a float's-width before
# a bar line would report "next bar" as the line it is already sitting on.
_BOUNDARY_EPSILON: Final = 1e-6


class Unit(StrEnum):
    BEAT = "beat"
    BAR = "bar"
    PHRASE = "phrase"


class Direction(StrEnum):
    NEAREST = "nearest"
    NEXT = "next"
    PREVIOUS = "previous"


@dataclass(frozen=True, slots=True)
class MusicalPosition:
    """A point on the timeline, in both time and musical coordinates."""

    time: float
    beat: float
    bar: int | None = None
    beat_in_bar: int | None = None

    def __str__(self) -> str:
        if self.bar is None:
            return f"{self.time:.3f}s (beat {self.beat:.2f})"
        return f"{self.time:.3f}s (bar {self.bar} beat {self.beat_in_bar})"


@dataclass(frozen=True, slots=True)
class QuantizedAction:
    """
    A jump requested at one instant and executed at the next boundary.

    Pure timing: a deck decides what to do with these numbers. ``execute_at``
    is when to act, ``destination`` is where to land.
    """

    requested_at: float
    execute_at: float
    destination: float
    quantization: Unit
    destination_position: MusicalPosition


@dataclass(frozen=True, slots=True)
class PhraseWindow:
    """One phrase in the deterministic bar grouping."""

    index: int
    start_bar: int
    bars: int
    start_time: float
    end_time: float


class Navigator:
    """
    Navigation over one track's tempo map.

    Bar- and phrase-level operations need a bar phase; if downbeat detection
    did not establish one they raise rather than silently assuming 4/4 and
    putting every jump in the wrong place. Beat-level operations always work.
    """

    def __init__(self, tempo_map: TempoMap, *, phrase_bars: int = DEFAULT_PHRASE_BARS) -> None:
        if phrase_bars <= 0:
            raise ValueError("phrase_bars must be positive")
        self.tempo_map = tempo_map
        self.phrase_bars = phrase_bars

    # -- positions ---------------------------------------------------------

    def position_at(self, time: float) -> MusicalPosition:
        beat = self.tempo_map.time_to_beat(time)
        if not self.tempo_map.has_bars:
            return MusicalPosition(time=time, beat=beat)
        bar_position = self.tempo_map.beat_to_bar(beat)
        return MusicalPosition(
            time=time,
            beat=beat,
            bar=bar_position.bar,
            beat_in_bar=bar_position.beat_in_bar,
        )

    def _position_at_beat(self, beat: float) -> MusicalPosition:
        return self.position_at(self.tempo_map.beat_to_time(beat))

    def _beats_per_unit(self, unit: Unit) -> float:
        if unit is Unit.BEAT:
            return 1.0
        beats_per_bar = self.tempo_map.beats_per_bar
        if beats_per_bar is None:
            raise ValueError(
                f"{unit.value} navigation needs a bar phase, and downbeat detection "
                f"did not establish one for this track"
            )
        return float(beats_per_bar * (self.phrase_bars if unit is Unit.PHRASE else 1))

    def _grid_origin(self, unit: Unit) -> float:
        """Musical beat the grid for this unit is counted from."""
        if unit is Unit.BEAT:
            return 0.0
        downbeat = self.tempo_map.downbeat_beat
        if downbeat is None:
            raise ValueError(f"{unit.value} navigation needs a downbeat phase")
        return float(downbeat)

    # -- snapping ----------------------------------------------------------

    def snap(
        self, time: float, unit: Unit = Unit.BEAT, direction: Direction = Direction.NEAREST
    ) -> MusicalPosition:
        """
        Move ``time`` onto the nearest, next or previous boundary of ``unit``.

        ``next`` from a position already exactly on a boundary returns the
        following one, which is what a user pressing "next bar" expects.
        """
        step = self._beats_per_unit(unit)
        origin = self._grid_origin(unit)
        offset = (self.tempo_map.time_to_beat(time) - origin) / step

        # NEXT and PREVIOUS mean the boundary strictly after / before this
        # position: from exactly on bar 12, "next bar" is 13, not 12.
        match direction:
            case Direction.NEAREST:
                snapped = np.round(offset)
            case Direction.NEXT:
                snapped = np.floor(offset + _BOUNDARY_EPSILON) + 1.0
            case Direction.PREVIOUS:
                snapped = np.ceil(offset - _BOUNDARY_EPSILON) - 1.0

        return self._position_at_beat(origin + float(snapped) * step)

    # -- jumping -----------------------------------------------------------

    def jump_beats(self, current_time: float, beats: int) -> MusicalPosition:
        """
        Move by whole beats, keeping the fractional position within the beat.

        Phase preservation is the point: jumping from 40% through a beat lands
        40% through the destination beat, so a loop taken mid-beat stays in
        time instead of shifting by a fraction on every jump.
        """
        return self._position_at_beat(self.tempo_map.time_to_beat(current_time) + beats)

    def jump_bars(self, current_time: float, bars: int) -> MusicalPosition:
        """Move by whole bars, keeping the position within the bar."""
        beats_per_bar = self.tempo_map.beats_per_bar
        if beats_per_bar is None:
            raise ValueError("bar jumps need a bar phase, which this track does not have")
        return self._position_at_beat(
            self.tempo_map.time_to_beat(current_time) + bars * beats_per_bar
        )

    def jump_phrases(self, current_time: float, phrases: int) -> MusicalPosition:
        return self.jump_bars(current_time, phrases * self.phrase_bars)

    # -- phrases -----------------------------------------------------------

    def phrase_grid(self, duration: float, *, phrase_bars: int | None = None) -> list[PhraseWindow]:
        """
        The deterministic phrase grid: bars grouped in fixed-size blocks from
        the first downbeat. Arithmetic, not detection -- see
        ``analysis.structure`` for evidence-based boundaries.
        """
        bars_per_phrase = phrase_bars or self.phrase_bars
        if not self.tempo_map.has_bars:
            return []

        windows: list[PhraseWindow] = []
        index = 0
        while True:
            start_bar = index * bars_per_phrase
            start_time = self.tempo_map.bar_to_time(start_bar)
            if start_time >= duration:
                break
            end_time = min(self.tempo_map.bar_to_time(start_bar + bars_per_phrase), duration)
            windows.append(
                PhraseWindow(
                    index=index,
                    start_bar=start_bar,
                    bars=bars_per_phrase,
                    start_time=round(start_time, 4),
                    end_time=round(end_time, 4),
                )
            )
            if end_time >= duration:
                break
            index += 1
        return windows

    def next_boundary(self, time: float, *, bars: int) -> MusicalPosition:
        """The next multiple-of-``bars`` bar line strictly after ``time``."""
        beats_per_bar = self.tempo_map.beats_per_bar
        downbeat = self.tempo_map.downbeat_beat
        if beats_per_bar is None or downbeat is None:
            raise ValueError("phrase boundaries need a bar phase")
        step = beats_per_bar * bars
        offset = (self.tempo_map.time_to_beat(time) - downbeat) / step
        return self._position_at_beat(
            downbeat + (float(np.floor(offset + _BOUNDARY_EPSILON)) + 1.0) * step
        )

    # -- scheduling --------------------------------------------------------

    def schedule(
        self,
        requested_at: float,
        destination: float,
        *,
        quantization: Unit = Unit.BAR,
    ) -> QuantizedAction:
        """
        Defer a jump to the next boundary.

        The deck asks for a jump the moment the user hits the button; this
        says when to actually do it so the result stays in phase. Both the
        execution instant and the landing point are quantised, because
        executing on the bar line but landing mid-bar would still break phase.
        """
        execute_at = self.snap(requested_at, quantization, Direction.NEXT)
        landing = self.snap(destination, quantization, Direction.NEAREST)
        return QuantizedAction(
            requested_at=requested_at,
            execute_at=execute_at.time,
            destination=landing.time,
            quantization=quantization,
            destination_position=landing,
        )
