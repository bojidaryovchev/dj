"""
Tempo reasoning shared by every backend.

Two things belong here rather than in a backend, because they are properties
of tempo itself and not of any library:

**Stability.** A single BPM number says nothing about whether the grid holds.
The coefficient of variation of the inter-beat intervals does: near zero for
a sequenced record, a few percent for a tight live band, large for anything
with rubato, a tempo ramp, or a beat tracker that lost the plot halfway.

**Metrical ambiguity.** 70 and 140 BPM describe the same pulse train. No
algorithm can choose between them from the signal alone -- the choice is a
genre convention, and getting it wrong is the single most common tempo
"error" in DJ software. So the reading the algorithm produced is reported
unchanged, and the alternatives are listed beside it for the DJ layer to
pick from, with the reason recorded.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...models import TempoCandidate, TempoRelation

__all__ = [
    "CV_AT_ZERO_CONFIDENCE",
    "bpm_from_beats",
    "consistency_confidence",
    "interval_stats",
    "tempo_candidates",
]

# Coefficient of variation at which a derived confidence reaches zero.
# 0.25 means intervals scattered by a quarter of their own length, which is
# well past "found a beat" and into "found something periodic-ish".
CV_AT_ZERO_CONFIDENCE: Final = 0.25

_MIN_BEATS_FOR_STATS: Final = 4


def interval_stats(beats: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    """
    ``(median interval, coefficient of variation)`` of the beat grid.

    The median locates the tempo; one missed beat doubles a single interval
    and a mean is defenceless against that. The CV then measures how tightly
    the grid holds -- but it is computed over intervals within +/-50% of the
    median only. Dropped and doubled beats are tracker artefacts, not tempo
    instability, and leaving them in would report a rock-solid sequenced
    track as unstable because the tracker blinked twice in six minutes.

    Returns ``(None, None)`` when there are too few beats to say anything.
    """
    times = np.asarray(beats, dtype=np.float64)
    if times.size < _MIN_BEATS_FOR_STATS:
        return None, None

    intervals = np.diff(times)
    intervals = intervals[intervals > 0]
    if intervals.size < _MIN_BEATS_FOR_STATS - 1:
        return None, None

    median = float(np.median(intervals))
    if median <= 0:
        return None, None

    inliers = intervals[np.abs(intervals - median) <= 0.5 * median]
    if inliers.size < 2:
        return median, None

    mean = float(np.mean(inliers))
    if mean <= 0:
        return median, None
    return median, float(np.std(inliers) / mean)


def bpm_from_beats(beats: list[float] | np.ndarray) -> float | None:
    """
    BPM implied by the detected beat grid, fitted across the whole track.

    Not the median interval. Beat trackers place beats on onset *frames*, and
    a frame is coarse: librosa's default hop is 512 samples at 22.05 kHz,
    23.2 ms. A 126 BPM beat lasts 20.5 of those, so every individual interval
    has to round to 20 frames (129.20 BPM) or 21 (123.05 BPM). The median
    picks one of those two and is therefore wrong by up to 2.4%, which is far
    outside what a DJ would accept and was exactly what this returned before.

    Fitting a line through (beat index, beat time) instead spreads that
    quantisation over every beat in the track, so the error falls roughly as
    1/N and a few hundred beats put it comfortably inside 0.05%. Indices come
    from the median interval rather than from position, so a dropped beat
    leaves a gap of two rather than shifting everything after it.

    For a track whose tempo genuinely drifts this returns the average, which
    is the right summary; ``interval_stats`` reports the drift separately.
    """
    times = np.asarray(beats, dtype=np.float64)
    median_interval, _ = interval_stats(times)
    if median_interval is None:
        return None

    # Indices accumulate from each interval in turn, rather than from each
    # beat's absolute position. Position-based indexing looks simpler and is
    # wrong: the median it divides by is itself quantised, so a 1% bias
    # compounds until, a minute in, beats are assigned to the wrong index
    # entirely and the fit is dragged further off than the median it was
    # meant to improve on. Per-interval rounding only needs each single
    # interval to land nearest its own beat count, which survives both the
    # bias and a dropped beat (an interval of two beats rounds to 2).
    steps = np.round(np.diff(times) / median_interval)
    steps = np.where(steps < 1, 1, steps)
    indices = np.concatenate([[0.0], np.cumsum(steps)])

    spread = float(indices[-1])
    if spread <= 0:
        return 60.0 / median_interval

    # Closed-form least squares: slope = cov(index, time) / var(index).
    variance = float(np.var(indices))
    if variance <= 0:
        return 60.0 / median_interval
    covariance = float(np.mean((indices - indices.mean()) * (times - times.mean())))
    period = covariance / variance

    if not np.isfinite(period) or period <= 0:
        return 60.0 / median_interval
    return 60.0 / period


def consistency_confidence(cv: float | None) -> float:
    """
    Turn interval spread into a 0-1 confidence.

    Derived, not reported: librosa's beat tracker gives no confidence of its
    own, and how regular the beats are is the best available proxy. It is a
    proxy with a known blind spot -- a tracker locked onto a steady but
    *wrong* grid (half-time, or the off-beat) scores just as high -- which is
    why the value is tagged ``beat_interval_consistency`` and not passed off
    as a probability.
    """
    if cv is None:
        return 0.0
    return float(np.clip(1.0 - cv / CV_AT_ZERO_CONFIDENCE, 0.0, 1.0))


def tempo_candidates(bpm: float, *, dj_min: float, dj_max: float) -> list[TempoCandidate]:
    """
    The metrically equivalent readings of one tempo, primary first.

    Only half and double are offered. Triplet readings (2/3, 3/2) are
    metrically real but are almost never what a DJ means by "the BPM", and
    listing them would add noise to every result to serve a rare case.
    """
    readings: list[tuple[float, TempoRelation]] = [
        (bpm, TempoRelation.PRIMARY),
        (bpm / 2.0, TempoRelation.HALF_TIME),
        (bpm * 2.0, TempoRelation.DOUBLE_TIME),
    ]
    return [
        TempoCandidate(
            bpm=round(value, 2),
            relation=relation,
            in_dj_range=dj_min <= value <= dj_max,
        )
        for value, relation in readings
    ]
