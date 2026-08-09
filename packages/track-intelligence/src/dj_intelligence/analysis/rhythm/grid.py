"""
Assembling the beat grid.

Takes what the separate stages produced — beat times from the tracker, bar
phase from the downbeat detector, local tempo from the curve — and builds the
one structure everything downstream reads: which source instant is which
musical beat, in which bar, and how much of that we believe where.

Grid confidence is per region, not per track. A five-minute record with a
ninety-second ambient intro has one stretch with no rhythmic evidence at all
and another with a beat every 476 ms, and averaging those into a single
number would misrepresent both. Regions with weak evidence are reported with
low confidence and a reason, rather than being given invented beats.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...models import BeatGrid, BeatObservation, Downbeat, GridRegion
from ...timeline.tempo_map import TempoMap

__all__ = ["beat_observations", "build_grid", "downbeat_list", "grid_regions"]

# A gap wider than this many beat periods means the tracker found nothing
# there — an intro, a breakdown, or a passage with no percussion.
_SPARSE_GAP_BEATS: Final = 2.5
# Regions shorter than this are not worth reporting separately.
_MIN_REGION_SECONDS: Final = 4.0
_LOW_EVIDENCE_CONFIDENCE: Final = 0.2


def beat_observations(
    times: np.ndarray,
    indices: np.ndarray,
    *,
    beats_per_bar: int | None,
    downbeat_beat: int | None,
    confidence: float,
) -> list[BeatObservation]:
    """
    The indexed beat list.

    Bars are attached here rather than recomputed by every consumer, because
    getting the phase right once is the whole point of the downbeat stage.
    """
    observations: list[BeatObservation] = []
    for position in range(times.size):
        beat_index = int(indices[position])
        bar: int | None = None
        beat_in_bar: int | None = None
        if beats_per_bar is not None and downbeat_beat is not None:
            offset = beat_index - downbeat_beat
            bar = int(np.floor(offset / beats_per_bar))
            beat_in_bar = int(offset - bar * beats_per_bar) + 1
        observations.append(
            BeatObservation(
                index=beat_index,
                time=round(float(times[position]), 4),
                bar=bar,
                beat_in_bar=beat_in_bar,
                confidence=round(confidence, 4),
            )
        )
    return observations


def downbeat_list(observations: list[BeatObservation], confidence: float) -> list[Downbeat]:
    """Bar lines, taken from the beats already labelled as beat one."""
    return [
        Downbeat(
            bar=observation.bar,
            beat_index=observation.index,
            time=observation.time,
            confidence=round(confidence, 4),
        )
        for observation in observations
        if observation.beat_in_bar == 1 and observation.bar is not None and observation.bar >= 0
    ]


def grid_regions(
    times: np.ndarray,
    duration: float,
    *,
    period: float,
    base_confidence: float,
) -> list[GridRegion]:
    """
    Split the timeline into stretches with and without rhythmic evidence.

    Detects three things: a beatless lead-in before the first beat, gaps in
    the middle where the tracker found nothing for several beats, and a
    beatless tail after the last beat.
    """
    if times.size == 0 or duration <= 0:
        return [
            GridRegion(
                start=0.0, end=round(duration, 3), confidence=0.0, reason="no_beats_detected"
            )
        ]

    sparse_gap = _SPARSE_GAP_BEATS * period
    regions: list[GridRegion] = []

    def add(start: float, end: float, confidence: float, reason: str | None) -> None:
        if end - start >= _MIN_REGION_SECONDS:
            regions.append(
                GridRegion(
                    start=round(max(0.0, start), 3),
                    end=round(min(duration, end), 3),
                    confidence=round(confidence, 4),
                    reason=reason,
                )
            )

    add(0.0, float(times[0]), _LOW_EVIDENCE_CONFIDENCE, "beatless_intro")

    # Walk the beats, breaking the tracked span wherever the tracker skipped.
    span_start = float(times[0])
    gaps = np.diff(times)
    for position, gap in enumerate(gaps):
        if gap > sparse_gap:
            add(span_start, float(times[position]), base_confidence, None)
            add(
                float(times[position]),
                float(times[position + 1]),
                _LOW_EVIDENCE_CONFIDENCE,
                "sparse_beat_evidence",
            )
            span_start = float(times[position + 1])
    add(span_start, float(times[-1]), base_confidence, None)

    add(float(times[-1]), duration, _LOW_EVIDENCE_CONFIDENCE, "beatless_outro")

    if not regions:
        add(0.0, duration, base_confidence, None)
    return sorted(regions, key=lambda region: region.start)


def build_grid(
    times: np.ndarray,
    indices: np.ndarray,
    *,
    duration: float,
    beats_per_bar: int | None,
    downbeat_beat: int | None,
    beat_confidence: float,
    phase_confidence: float,
) -> tuple[BeatGrid, list[BeatObservation], list[Downbeat], TempoMap | None]:
    """
    Build the grid, the indexed beats, the bar lines and the tempo map.

    Returns the tempo map too because it is derived from exactly this data and
    building it anywhere else would risk a second, subtly different grid.
    """
    observations = beat_observations(
        times,
        indices,
        beats_per_bar=beats_per_bar,
        downbeat_beat=downbeat_beat,
        confidence=beat_confidence,
    )
    downbeats = downbeat_list(observations, phase_confidence)

    period = float(np.median(np.diff(times))) if times.size > 1 else 0.5
    regions = grid_regions(times, duration, period=period, base_confidence=beat_confidence)

    # One headline number, weighted by how much of the track each region
    # covers -- so a long beatless intro genuinely lowers grid confidence.
    covered = sum(region.end - region.start for region in regions)
    overall = (
        sum(region.confidence * (region.end - region.start) for region in regions) / covered
        if covered > 0
        else 0.0
    )
    if beats_per_bar is None:
        # Without a bar phase the grid cannot place a bar line, and most of
        # what a DJ wants from it is unavailable.
        overall *= 0.5

    tempo_map: TempoMap | None = None
    if times.size >= 2:
        tempo_map = TempoMap.from_beats(
            times,
            indices,
            beats_per_bar=beats_per_bar,
            downbeat_beat=(
                int(indices[downbeat_beat])
                if downbeat_beat is not None and downbeat_beat < indices.size
                else None
            ),
        )

    grid = BeatGrid(
        beats_per_bar=beats_per_bar,
        first_downbeat_time=downbeats[0].time if downbeats else None,
        first_downbeat_beat_index=downbeats[0].beat_index if downbeats else None,
        bar_count=len(downbeats),
        confidence=round(float(np.clip(overall, 0.0, 1.0)), 4),
        regions=regions,
    )
    return grid, observations, downbeats, tempo_map
