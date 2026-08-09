"""
Building a warp map: which source instants to pin to which target instants.

A warp map is a *plan*, not an edit. It says nothing about audio; it says that
if you pinned these source times to these target times and stretched linearly
between them, every beat would land on a constant-tempo grid.

Two ideas carry the whole module.

**The target grid is anchored, not started from zero.** Beat *b* belongs at
``anchor_time + (b - anchor_beat) * 60/target_bpm``. The anchor is the first
reliable downbeat, so bar lines stay where they are and the intro is not
squeezed to make room for a timeline that begins at zero. Nothing before the
anchor is moved.

**Markers are inserted only where they buy something.** The obvious
implementation puts a marker on every beat, and it is wrong: it pins detector
noise into the render, forces a different stretch ratio on every half-second
of audio, and smears the transients it was supposed to preserve. Instead the
mapping is a curve, and we simplify it — walk forward from the last marker
and only place a new one when linear interpolation would put some beat
further than ``max_grid_error_ms`` from where it belongs. On a track that
drifts smoothly, a handful of markers reproduce the whole curve to within a
few milliseconds.

The error measured during simplification is the error the *render* will have,
not the distance between two curves in the abstract: for each beat we compute
where its source instant would actually come out under the simplified map,
and compare that to where it should be.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..models import WarpMap, WarpMarker, WarpMetrics
from .tempo_map import TempoMap

__all__ = [
    "WarpParameters",
    "build_warp_map",
    "fitted_bpm",
    "smooth_source_times",
    "target_times_for",
]

_MS: Final = 1000.0


@dataclass(frozen=True, slots=True)
class WarpParameters:
    """
    Knobs for marker generation. Defaults are measured, not guessed.

    ``max_grid_error_ms`` at 10 ms: comfortably below the ~20 ms at which a
    timing offset becomes audible as flam against another deck, and well above
    the ~3 ms of noise a beat tracker leaves on a clean grid, so it does not
    chase its own tail.

    ``max_marker_distance_bars`` at 32 bars caps how far the renderer can go
    on one ratio even when the fit is perfect, which bounds the damage if the
    grid is wrong somewhere in the middle.

    ``min_marker_distance_beats`` at 16 keeps markers at least four bars
    apart. Closer than that and beat-detection jitter, not tempo, decides the
    stretch ratio: a +/-20 ms wobble at each end of a four-beat segment is a
    2% ratio error, which is a real audible stretch applied to fix noise.

    ``smooth_beats`` fits the source grid before planning against it, for the
    same reason. Warping exists to remove *drift*; detector jitter is not
    drift and must not be rendered into the audio.
    """

    smooth_beats: bool = True
    smoothing_window_beats: int = 33

    target_bpm: float | None = None
    """``None`` keeps the track's own nominal tempo."""

    max_grid_error_ms: float = 10.0
    max_marker_distance_bars: int = 32
    min_marker_distance_beats: int = 16
    min_safe_stretch_ratio: float = 0.9
    max_safe_stretch_ratio: float = 1.1
    tolerance_ms: float = 15.0
    """Below this worst-case source error, warping is not worth doing."""


def target_times_for(
    beat_indices: np.ndarray, *, target_bpm: float, anchor_beat: float, anchor_time: float
) -> np.ndarray:
    """Where each musical beat belongs on the constant-tempo grid."""
    seconds_per_beat = 60.0 / target_bpm
    return (
        anchor_time + (np.asarray(beat_indices, dtype=np.float64) - anchor_beat) * seconds_per_beat
    )


def fitted_bpm(source: np.ndarray, beats: np.ndarray) -> float | None:
    """
    Least-squares tempo over the whole grid.

    Not ``TempoMap.average_bpm``, which divides the span by the two endpoint
    beats and therefore inherits their jitter. A +/-20 ms wobble on the first
    and last beat of a two-minute track is a 0.03% tempo error, and a target
    grid built on it pivots away from the source by ~40 ms by the end -- which
    reads as drift, and would have this system warping perfectly sequenced
    tracks to correct its own arithmetic. Fitting across every beat drops that
    to the millisecond.
    """
    if source.size < 2:
        return None
    variance = float(np.var(beats))
    if variance <= 0:
        return None
    covariance = float(np.mean((beats - beats.mean()) * (source - source.mean())))
    period = covariance / variance
    if not np.isfinite(period) or period <= 0:
        return None
    return 60.0 / period


def smooth_source_times(
    source: np.ndarray, beats: np.ndarray, window_beats: int = 33
) -> np.ndarray:
    """
    Fit the source grid locally, so markers follow tempo rather than noise.

    A moving least-squares line through (beat index, source time). Inside the
    window the beats of a real record lie on a straight line to within a few
    milliseconds, so the fit tracks genuine tempo movement while averaging out
    the tracker's per-beat wobble. Endpoints use the nearest full window
    rather than a shrinking one, which stops the fit from degenerating where
    it has least data.
    """
    count = source.size
    window = max(5, min(window_beats | 1, count if count % 2 else count - 1))
    if count < window:
        return source.copy()

    half = window // 2
    smoothed = np.empty_like(source)
    for index in range(count):
        start = min(max(0, index - half), count - window)
        end = start + window
        local_beats = beats[start:end]
        local_times = source[start:end]
        variance = float(np.var(local_beats))
        if variance <= 0:
            smoothed[index] = source[index]
            continue
        slope = (
            float(np.mean((local_beats - local_beats.mean()) * (local_times - local_times.mean())))
            / variance
        )
        intercept = float(local_times.mean() - slope * local_beats.mean())
        smoothed[index] = slope * beats[index] + intercept
    return smoothed


def _rendered_error_ms(
    source: np.ndarray,
    target: np.ndarray,
    start: int,
    end: int,
) -> float:
    """
    Worst error, in ms, if beats ``start..end`` share one linear segment.

    This is what the render will actually do: source content between two
    markers is mapped linearly onto the span between their target times, so a
    beat's rendered position is its source position scaled by the segment's
    ratio.
    """
    source_span = source[end] - source[start]
    target_span = target[end] - target[start]
    if source_span <= 0:
        return float("inf")
    ratio = target_span / source_span
    rendered = target[start] + (source[start + 1 : end] - source[start]) * ratio
    if rendered.size == 0:
        return 0.0
    return float(np.max(np.abs(rendered - target[start + 1 : end]))) * _MS


def build_warp_map(
    tempo_map: TempoMap,
    *,
    parameters: WarpParameters | None = None,
    anchor_beat: int | None = None,
) -> WarpMap:
    """
    Plan a correction of ``tempo_map`` onto a constant-tempo grid.

    Returns markers, the stretch each segment implies, and the metrics a
    caller needs to decide whether to render at all. It does not decide -- see
    ``dj.warp_advice``.
    """
    params = parameters or WarpParameters()

    raw_source = np.asarray(tempo_map.times, dtype=np.float64)
    beats = np.asarray(tempo_map.beats, dtype=np.float64)
    source = (
        smooth_source_times(raw_source, beats, params.smoothing_window_beats)
        if params.smooth_beats
        else raw_source
    )
    target_bpm = params.target_bpm or fitted_bpm(source, beats) or tempo_map.average_bpm

    resolved_anchor = (
        float(anchor_beat)
        if anchor_beat is not None
        else float(tempo_map.downbeat_beat if tempo_map.downbeat_beat is not None else beats[0])
    )
    # From the smoothed grid, not the raw one: the markers are built from
    # smoothed source times, and an anchor taken from a raw beat would leave
    # the anchor marker mapping to a target a millisecond or two away -- i.e.
    # the one point that is supposed to stay put would move.
    anchor_time = float(np.interp(resolved_anchor, beats, source))
    target = target_times_for(
        beats, target_bpm=target_bpm, anchor_beat=resolved_anchor, anchor_time=anchor_time
    )

    warnings: list[str] = []

    beats_per_bar = tempo_map.beats_per_bar or 4
    max_gap_beats = max(
        params.min_marker_distance_beats, params.max_marker_distance_bars * beats_per_bar
    )

    kept = _simplify(
        source,
        target,
        beats,
        max_error_ms=params.max_grid_error_ms,
        min_gap_beats=params.min_marker_distance_beats,
        max_gap_beats=max_gap_beats,
    )

    markers = [
        WarpMarker(
            source_time=round(float(source[position]), 6),
            source_beat=int(beats[position]),
            target_time=round(float(target[position]), 6),
        )
        for position in kept
    ]

    metrics, ratio_warnings = _metrics(source, target, kept, raw_source=raw_source, params=params)
    warnings.extend(ratio_warnings)

    return WarpMap(
        target_bpm=round(target_bpm, 4),
        anchor_beat=int(resolved_anchor),
        anchor_time=round(anchor_time, 6),
        markers=markers,
        metrics=metrics,
        warnings=warnings,
    )


def _simplify(
    source: np.ndarray,
    target: np.ndarray,
    beats: np.ndarray,
    *,
    max_error_ms: float,
    min_gap_beats: int,
    max_gap_beats: int,
) -> list[int]:
    """
    Greedy forward line simplification over the source-to-target curve.

    From the current marker, extend as far as possible while every beat in
    between still renders within tolerance, then plant the next marker. Linear
    in the number of beats if the search doubles, but a plain forward scan is
    fast enough here (a six-minute track has ~750 beats) and much easier to
    reason about.
    """
    count = source.size
    if count < 2:
        return list(range(count))

    kept = [0]
    current = 0
    while current < count - 1:
        furthest_within_budget = current + 1
        first_far_enough = count - 1
        candidate = current + 1

        while candidate < count:
            gap = beats[candidate] - beats[current]
            if gap > max_gap_beats:
                break
            if gap >= min_gap_beats:
                first_far_enough = min(first_far_enough, candidate)
                if _rendered_error_ms(source, target, current, candidate) > max_error_ms:
                    break
            furthest_within_budget = candidate
            candidate += 1

        # Two constraints pull in opposite directions: accuracy wants the
        # marker early, and musical spacing wants it at least a few bars on.
        # Spacing wins, because a marker every other beat renders detector
        # noise into the audio. When the budget cannot be met that far out,
        # the residual is reported rather than chased.
        furthest = max(furthest_within_budget, min(first_far_enough, count - 1))
        furthest = max(furthest, current + 1)
        kept.append(furthest)
        current = furthest

    if kept[-1] != count - 1:
        kept.append(count - 1)
    return kept


def _metrics(
    source: np.ndarray,
    target: np.ndarray,
    kept: list[int],
    *,
    raw_source: np.ndarray,
    params: WarpParameters,
) -> tuple[WarpMetrics, list[str]]:
    """
    Stretch ratios, corrections, and the residual after simplifying.

    Two error figures, and the distinction decides whether anything gets
    rendered. ``max_correction_ms`` measures the raw detected beats against
    the target and therefore includes detector jitter. ``systematic_error_ms``
    measures the *smoothed* grid, so it is the part of the error that is
    really the track drifting -- the only part warping can fix.
    """
    warnings: list[str] = []
    ratios: list[float] = []
    residual = 0.0

    for start, end in itertools.pairwise(kept):
        source_span = float(source[end] - source[start])
        target_span = float(target[end] - target[start])
        if source_span <= 0:
            continue
        ratios.append(target_span / source_span)
        residual = max(residual, _rendered_error_ms(source, target, start, end))

    ratio_array = np.array(ratios, dtype=np.float64) if ratios else np.array([1.0])
    minimum = float(np.min(ratio_array))
    maximum = float(np.max(ratio_array))

    if minimum < params.min_safe_stretch_ratio or maximum > params.max_safe_stretch_ratio:
        warnings.append("warp_requires_large_local_stretch")

    corrections = np.abs(raw_source - target) * _MS
    systematic = np.abs(source - target) * _MS
    return (
        WarpMetrics(
            marker_count=len(kept),
            systematic_error_ms=round(float(np.max(systematic)), 3),
            mean_stretch_ratio=round(float(np.mean(ratio_array)), 6),
            min_stretch_ratio=round(minimum, 6),
            max_stretch_ratio=round(maximum, 6),
            max_correction_ms=round(float(np.max(corrections)), 3),
            mean_correction_ms=round(float(np.mean(corrections)), 3),
            residual_grid_error_ms=round(residual, 3),
        ),
        warnings,
    )
