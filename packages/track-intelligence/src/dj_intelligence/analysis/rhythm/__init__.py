"""
Rhythmic measurement: bars, local tempo, and the assembled grid.

Beat *detection* stays in ``analysis/tempo`` where it always was — this
package answers the questions that come after it. Which beat is beat one, how
many beats are in a bar, how the tempo moves, and how much of the grid we
actually believe.
"""

from .downbeats import BeatSyncDownbeatAnalyzer, DownbeatEstimate
from .grid import beat_observations, build_grid, downbeat_list, grid_regions
from .refine import GridOffset, estimate_grid_offset
from .stage import RhythmResult, RhythmStage
from .tempo_curve import (
    DEFAULT_WINDOW_BEATS,
    DRIFT_THRESHOLDS,
    classify_drift,
    local_tempo_curve,
)

__all__ = [
    "DEFAULT_WINDOW_BEATS",
    "DRIFT_THRESHOLDS",
    "BeatSyncDownbeatAnalyzer",
    "DownbeatEstimate",
    "GridOffset",
    "RhythmResult",
    "RhythmStage",
    "beat_observations",
    "build_grid",
    "classify_drift",
    "downbeat_list",
    "estimate_grid_offset",
    "grid_regions",
    "local_tempo_curve",
]
