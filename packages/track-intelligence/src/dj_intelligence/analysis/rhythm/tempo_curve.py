"""
Local tempo through the track, and whether it can be called constant.

One BPM number describes a sequenced record perfectly and a live one not at
all. This module measures tempo *locally* so a drifting track is described
rather than averaged, and then classifies the drift against documented
thresholds.

Two decisions matter here.

**Local tempo is fitted to detected beats, not re-detected per window.**
Running a beat tracker again inside every 30-second window would be slow, and
worse, would let each window pick its own metrical level — producing a
"tempo curve" that jumps between 63 and 126 BPM on a perfectly steady track.
Fitting the beats we already have keeps every window on the same grid.

**The fit is least squares over the window, not the interval median.** This
is the lesson from the 2.4% BPM error in the previous phase: beat times are
quantised to onset frames, so any estimator that looks at one interval
inherits the full quantisation. Regression over a window of beats averages it
down, and the residual is what sets the noise floor below.

**The noise floor sets the thresholds.** On a generated constant-tempo track
the local estimates still scatter a little, purely from frame quantisation.
Reporting that scatter as "drift" would flag every sequenced track in a
library. The ``stable`` cut-off is therefore set above the measured floor,
not at zero.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...models import DriftClassification, TempoCurvePoint, TempoDrift

__all__ = [
    "DEFAULT_WINDOW_BEATS",
    "DRIFT_THRESHOLDS",
    "classify_drift",
    "local_tempo_curve",
]

# 64 beats is ~30 s at 126 BPM. Long enough that quantisation noise in the
# fitted slope is well under 0.1%, short enough to follow a real tempo ramp.
DEFAULT_WINDOW_BEATS: Final = 64
DEFAULT_HOP_BEATS: Final = 32

_MIN_BEATS_FOR_FIT: Final = 8

# Relative range of local tempo, (max - min) / nominal, as a fraction.
DRIFT_THRESHOLDS: Final[tuple[tuple[float, DriftClassification], ...]] = (
    (0.002, DriftClassification.STABLE),
    (0.010, DriftClassification.MINOR_DRIFT),
    (0.030, DriftClassification.VARIABLE_TEMPO),
)


def _fit_bpm(times: np.ndarray, indices: np.ndarray) -> float | None:
    """
    Least-squares tempo over one window.

    ``indices`` are musical beat indices, so a dropped beat inside the window
    widens the gap instead of bending the line.
    """
    if times.size < 2:
        return None
    variance = float(np.var(indices))
    if variance <= 0:
        return None
    covariance = float(np.mean((indices - indices.mean()) * (times - times.mean())))
    period = covariance / variance
    if not np.isfinite(period) or period <= 0:
        return None
    return 60.0 / period


def local_tempo_curve(
    beat_times: np.ndarray,
    beat_indices: np.ndarray,
    *,
    window_beats: int = DEFAULT_WINDOW_BEATS,
    hop_beats: int = DEFAULT_HOP_BEATS,
) -> list[TempoCurvePoint]:
    """
    Tempo over successive overlapping windows of the beat grid.

    Windows are measured in beats rather than seconds so that every estimate
    is fitted over the same amount of *musical* evidence regardless of tempo.
    """
    times = np.asarray(beat_times, dtype=np.float64)
    indices = np.asarray(beat_indices, dtype=np.float64)
    if times.size < _MIN_BEATS_FOR_FIT:
        return []

    window = max(_MIN_BEATS_FOR_FIT, min(window_beats, times.size))
    hop = max(1, min(hop_beats, window))

    points: list[TempoCurvePoint] = []
    start = 0
    while start < times.size:
        end = min(start + window, times.size)
        if end - start < _MIN_BEATS_FOR_FIT:
            break
        bpm = _fit_bpm(times[start:end], indices[start:end])
        if bpm is not None:
            points.append(
                TempoCurvePoint(
                    start_time=round(float(times[start]), 3),
                    end_time=round(float(times[end - 1]), 3),
                    bpm=round(bpm, 3),
                    beat_count=int(end - start),
                )
            )
        if end >= times.size:
            break
        start += hop
    return points


def classify_drift(curve: list[TempoCurvePoint], nominal_bpm: float | None) -> TempoDrift:
    """
    Turn the tempo curve into a verdict, with the metrics that produced it.

    The classification is a convenience. ``relative_percent`` and the local
    minimum and maximum are the actual evidence, and they are always present
    so a caller can apply its own threshold.
    """
    if not curve or nominal_bpm is None or nominal_bpm <= 0:
        return TempoDrift(nominal_bpm=nominal_bpm, classification=DriftClassification.UNKNOWN)

    values = np.array([point.bpm for point in curve], dtype=np.float64)
    local_min = float(np.min(values))
    local_max = float(np.max(values))
    relative = (local_max - local_min) / nominal_bpm
    max_delta = float(np.max(np.abs(values - nominal_bpm)))

    classification = DriftClassification.HIGHLY_VARIABLE
    for threshold, label in DRIFT_THRESHOLDS:
        if relative < threshold:
            classification = label
            break

    return TempoDrift(
        nominal_bpm=round(nominal_bpm, 3),
        local_bpm_min=round(local_min, 3),
        local_bpm_max=round(local_max, 3),
        max_absolute_bpm_delta=round(max_delta, 4),
        relative_percent=round(relative * 100.0, 4),
        classification=classification,
        tempo_stable=classification is DriftClassification.STABLE,
    )
