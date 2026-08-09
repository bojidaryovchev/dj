"""
Checking that a render actually did what the map said.

Rendering audio and assuming it worked is not a pipeline, it is a hope. The
stretcher can round a segment length, a marker can be misplaced, a beat
tracker can have been wrong about the source in the first place — and every
one of those failures produces a file that looks fine and is out of time.

So the rendered file is analysed again, from scratch, by the same engine. Its
beats are compared against the grid the warp was aiming at, and the same
measurement is taken on the input for comparison. That last part matters: an
absolute error of 6 ms means nothing on its own, but 24 ms before and 3 ms
after is evidence.

Comparison is done in *musical* terms. Each rendered beat is matched to the
nearest ideal grid position and the residual is the error, which is robust to
the tracker finding one beat more or fewer than expected.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np

from ..models import WarpMap, WarpVerification
from ..observability import get_logger

__all__ = ["BeatSource", "grid_error_ms", "verify_render"]

BeatSource = Callable[[Path], list[float]]
"""Something that returns beat times for a path -- normally the pipeline."""

log = get_logger(__name__)

_MS: Final = 1000.0
_MIN_BEATS: Final = 8


def grid_error_ms(
    beat_times: np.ndarray | list[float],
    *,
    target_bpm: float,
    anchor_time: float,
) -> np.ndarray:
    """
    Distance from each beat to the nearest position on an ideal grid.

    The grid is infinite in both directions from ``anchor_time``, so this does
    not care where the track starts or how many beats were found.
    """
    times = np.asarray(beat_times, dtype=np.float64)
    if times.size == 0:
        return np.array([])
    period = 60.0 / target_bpm
    offsets = (times - anchor_time) / period
    return np.abs(offsets - np.round(offsets)) * period * _MS


def verify_render(
    rendered_path: Path,
    warp_map: WarpMap,
    *,
    analyze_beats: BeatSource,
    source_beats: np.ndarray | list[float] | None = None,
    threshold_ms: float = 15.0,
) -> WarpVerification:
    """
    Re-analyse ``rendered_path`` and score it against the target grid.

    ``analyze_beats`` is a callable returning beat times for a path; the
    engine passes its own pipeline, so verification uses exactly the detector
    that produced the map rather than a second opinion that might disagree for
    unrelated reasons.
    """
    rendered_beats = analyze_beats(rendered_path)
    errors = grid_error_ms(
        rendered_beats, target_bpm=warp_map.target_bpm, anchor_time=warp_map.anchor_time
    )
    if errors.size < _MIN_BEATS:
        return WarpVerification(
            target_bpm=warp_map.target_bpm,
            mean_grid_error_ms=float("nan"),
            p95_grid_error_ms=float("nan"),
            max_grid_error_ms=float("nan"),
            beats_compared=int(errors.size),
            threshold_ms=threshold_ms,
            passed=False,
        )

    mean_error = float(np.mean(errors))
    source_mean: float | None = None
    improvement: float | None = None
    if source_beats is not None and len(source_beats) >= _MIN_BEATS:
        source_errors = grid_error_ms(
            source_beats, target_bpm=warp_map.target_bpm, anchor_time=warp_map.anchor_time
        )
        source_mean = round(float(np.mean(source_errors)), 3)
        if mean_error > 0:
            improvement = round(source_mean / mean_error, 3)

    return WarpVerification(
        target_bpm=warp_map.target_bpm,
        mean_grid_error_ms=round(mean_error, 3),
        p95_grid_error_ms=round(float(np.percentile(errors, 95)), 3),
        max_grid_error_ms=round(float(np.max(errors)), 3),
        beats_compared=int(errors.size),
        threshold_ms=threshold_ms,
        passed=bool(np.mean(errors) <= threshold_ms),
        source_mean_grid_error_ms=source_mean,
        improvement_factor=improvement,
    )
