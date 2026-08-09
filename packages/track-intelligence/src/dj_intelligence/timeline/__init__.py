"""
Musical time.

Pure primitives that map between audio seconds and musical position, and
navigate in the latter. Like ``music/``, nothing here measures anything or
touches audio: these are constructed *from* measurements and consumed by both
the analysis and the DJ layers, which is why they live in neither.

    TempoMap    source seconds <-> musical beats, in both directions
    Navigator   snapping, beat/bar/phrase jumps, quantised scheduling
    WarpMap     a plan for moving a drifting grid onto a constant one
"""

from .navigation import (
    DEFAULT_PHRASE_BARS,
    Direction,
    MusicalPosition,
    Navigator,
    PhraseWindow,
    QuantizedAction,
    Unit,
)
from .tempo_map import BarPosition, TempoMap
from .warp_map import WarpParameters, build_warp_map, target_times_for

__all__ = [
    "DEFAULT_PHRASE_BARS",
    "BarPosition",
    "Direction",
    "MusicalPosition",
    "Navigator",
    "PhraseWindow",
    "QuantizedAction",
    "TempoMap",
    "Unit",
    "WarpParameters",
    "build_warp_map",
    "target_times_for",
]
