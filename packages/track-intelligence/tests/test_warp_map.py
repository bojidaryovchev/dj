"""
Warp map planning: markers, smoothing, metrics and the decision to warp.

No audio here — planning is arithmetic over a tempo map, and testing it
without a decoder is what makes it possible to check the exact marker
positions for a grid whose drift is known to the microsecond.
"""

from __future__ import annotations

import numpy as np
import pytest

from dj_intelligence.dj.warp_advice import WarpAdviceRules, recommend_warp
from dj_intelligence.models import DriftClassification, WarpSkipReason
from dj_intelligence.timeline import TempoMap, WarpParameters, build_warp_map
from dj_intelligence.timeline.warp_map import fitted_bpm, smooth_source_times

SPB = 60.0 / 126.0


def steady_map(beats: int = 400, bpm: float = 126.0, start: float = 1.5) -> TempoMap:
    return TempoMap.from_beats(
        np.arange(beats) * (60.0 / bpm) + start, beats_per_bar=4, downbeat_beat=0
    )


def drifting_map(start_bpm: float = 125.0, end_bpm: float = 127.0, beats: int = 400) -> TempoMap:
    periods = 60.0 / np.linspace(start_bpm, end_bpm, beats)
    times = np.concatenate([[1.5], 1.5 + np.cumsum(periods[:-1])])
    return TempoMap.from_beats(times, beats_per_bar=4, downbeat_beat=0)


def jittered_map(seed: int = 0, sigma: float = 0.007, beats: int = 400) -> TempoMap:
    """A perfect grid seen through a beat tracker's noise."""
    rng = np.random.default_rng(seed)
    times = np.arange(beats) * SPB + 1.5 + rng.normal(0.0, sigma, beats)
    return TempoMap.from_beats(np.sort(times), beats_per_bar=4, downbeat_beat=0)


# -- tempo fitting ----------------------------------------------------------


def test_fitted_bpm_beats_the_endpoint_average() -> None:
    """
    Why the target tempo is fitted rather than taken from the span.

    ``average_bpm`` divides by the first and last beat only, so their jitter
    becomes a tempo error that accumulates into fake drift. This was
    recommending warps on perfectly sequenced tracks.
    """
    tempo_map = jittered_map()
    fitted = fitted_bpm(np.asarray(tempo_map.times), np.asarray(tempo_map.beats))
    assert fitted is not None
    assert abs(fitted - 126.0) < abs(tempo_map.average_bpm - 126.0)
    assert fitted == pytest.approx(126.0, abs=0.01)


def test_smoothing_removes_jitter_but_keeps_drift() -> None:
    jittered = jittered_map(sigma=0.01)
    smoothed = smooth_source_times(np.asarray(jittered.times), np.asarray(jittered.beats), 33)
    ideal = np.arange(400) * SPB + 1.5
    assert np.std(smoothed - ideal) < np.std(np.asarray(jittered.times) - ideal) / 2

    drifting = drifting_map()
    kept = smooth_source_times(np.asarray(drifting.times), np.asarray(drifting.beats), 33)
    assert np.max(np.abs(kept - np.asarray(drifting.times))) < 0.01


# -- markers ----------------------------------------------------------------


def test_a_steady_track_needs_almost_no_markers() -> None:
    warp_map = build_warp_map(steady_map())
    assert warp_map.metrics.marker_count <= 8
    assert warp_map.metrics.systematic_error_ms < 5.0
    assert warp_map.metrics.min_stretch_ratio == pytest.approx(1.0, abs=1e-3)


def test_drift_is_described_by_a_handful_of_markers() -> None:
    warp_map = build_warp_map(drifting_map(), parameters=WarpParameters(target_bpm=126.0))
    assert 3 <= warp_map.metrics.marker_count <= 20, "should simplify, not mark every beat"
    assert warp_map.metrics.systematic_error_ms > 100.0
    assert warp_map.metrics.residual_grid_error_ms <= 10.0


def test_markers_stay_within_the_error_budget() -> None:
    """The simplification promise: whatever is dropped, the plan still lands
    every beat inside max_grid_error_ms."""
    for budget in (5.0, 10.0, 25.0):
        warp_map = build_warp_map(
            drifting_map(120.0, 132.0),
            parameters=WarpParameters(target_bpm=126.0, max_grid_error_ms=budget),
        )
        assert warp_map.metrics.residual_grid_error_ms <= budget + 1e-6


def test_a_tighter_budget_costs_more_markers() -> None:
    loose = build_warp_map(
        drifting_map(120.0, 132.0), parameters=WarpParameters(max_grid_error_ms=40.0)
    )
    tight = build_warp_map(
        drifting_map(120.0, 132.0), parameters=WarpParameters(max_grid_error_ms=3.0)
    )
    assert tight.metrics.marker_count > loose.metrics.marker_count


def test_markers_are_ordered_and_on_real_beats() -> None:
    warp_map = build_warp_map(drifting_map())
    sources = [marker.source_time for marker in warp_map.markers]
    targets = [marker.target_time for marker in warp_map.markers]
    assert sources == sorted(sources)
    assert targets == sorted(targets)
    assert all(marker.source_beat >= 0 for marker in warp_map.markers)


def test_markers_are_never_closer_than_the_minimum() -> None:
    warp_map = build_warp_map(
        drifting_map(110.0, 140.0),
        parameters=WarpParameters(min_marker_distance_beats=16, max_grid_error_ms=1.0),
    )
    beats = [marker.source_beat for marker in warp_map.markers]
    gaps = np.diff(beats)
    # The final marker is pinned to the last beat and may fall short.
    assert all(gap >= 16 for gap in gaps[:-1])


def test_the_anchor_does_not_move() -> None:
    """
    Bar lines stay where they are: the anchor maps exactly to itself.

    The anchor is taken from the *smoothed* grid, so it sits within a few
    milliseconds of the raw beat rather than exactly on it — that is the
    point, since the markers are smoothed too and a raw anchor would leave
    the one fixed point of the whole map needing to move.
    """
    tempo_map = drifting_map()
    warp_map = build_warp_map(tempo_map, anchor_beat=0)

    assert warp_map.markers[0].source_beat == 0
    assert warp_map.markers[0].target_time == pytest.approx(
        warp_map.markers[0].source_time, abs=1e-6
    )
    assert warp_map.anchor_time == pytest.approx(tempo_map.beat_to_time(0), abs=0.02)


def test_target_bpm_is_honoured() -> None:
    warp_map = build_warp_map(steady_map(), parameters=WarpParameters(target_bpm=128.0))
    assert warp_map.target_bpm == pytest.approx(128.0)
    spacing = 60.0 / 128.0
    first, second = warp_map.markers[0], warp_map.markers[1]
    beats = second.source_beat - first.source_beat
    assert (second.target_time - first.target_time) == pytest.approx(beats * spacing, abs=1e-6)


def test_extreme_stretch_is_flagged() -> None:
    warp_map = build_warp_map(steady_map(bpm=126.0), parameters=WarpParameters(target_bpm=200.0))
    assert "warp_requires_large_local_stretch" in warp_map.warnings


# -- the recommendation (DJ interpretation) ---------------------------------


def advise(tempo_map: TempoMap, **kwargs: object) -> object:
    warp_map = build_warp_map(tempo_map, parameters=WarpParameters(**kwargs))  # type: ignore[arg-type]
    return recommend_warp(
        warp_map,
        grid_confidence=0.9,
        drift=DriftClassification.STABLE,
        tempo_reliable=True,
    )


def test_a_good_track_is_left_alone() -> None:
    recommendation = advise(jittered_map())
    assert recommendation.required is False
    assert recommendation.skip_reason is WarpSkipReason.ALREADY_ALIGNED


def test_a_drifting_track_is_worth_warping() -> None:
    warp_map = build_warp_map(drifting_map(), parameters=WarpParameters(target_bpm=126.0))
    recommendation = recommend_warp(
        warp_map,
        grid_confidence=0.9,
        drift=DriftClassification.VARIABLE_TEMPO,
        tempo_reliable=True,
    )
    assert recommendation.required is True


def test_an_untrusted_grid_is_never_warped() -> None:
    warp_map = build_warp_map(drifting_map())
    recommendation = recommend_warp(
        warp_map,
        grid_confidence=0.1,
        drift=DriftClassification.VARIABLE_TEMPO,
        tempo_reliable=True,
    )
    assert recommendation.required is False
    assert recommendation.skip_reason is WarpSkipReason.NO_GRID


def test_an_unreliable_tempo_is_never_warped() -> None:
    """Half-time protection: warping a 128 track read as 64 would double it."""
    warp_map = build_warp_map(drifting_map())
    recommendation = recommend_warp(
        warp_map,
        grid_confidence=0.9,
        drift=DriftClassification.VARIABLE_TEMPO,
        tempo_reliable=False,
    )
    assert recommendation.required is False
    assert recommendation.skip_reason is WarpSkipReason.TEMPO_UNRELIABLE


def test_a_violent_correction_is_refused() -> None:
    warp_map = build_warp_map(steady_map(), parameters=WarpParameters(target_bpm=170.0))
    recommendation = recommend_warp(
        warp_map,
        grid_confidence=0.95,
        drift=DriftClassification.STABLE,
        tempo_reliable=True,
        target_bpm_requested=True,
    )
    assert recommendation.required is False
    assert recommendation.skip_reason is WarpSkipReason.UNSAFE_STRETCH


def test_an_explicit_target_overrides_already_aligned() -> None:
    """A user asking for 128 does not want to hear that 126 is tidy."""
    warp_map = build_warp_map(steady_map(), parameters=WarpParameters(target_bpm=128.0))
    recommendation = recommend_warp(
        warp_map,
        grid_confidence=0.95,
        drift=DriftClassification.STABLE,
        tempo_reliable=True,
        target_bpm_requested=True,
    )
    assert recommendation.required is True


def test_tolerance_is_configurable() -> None:
    warp_map = build_warp_map(drifting_map(125.9, 126.1))
    strict = recommend_warp(
        warp_map,
        grid_confidence=0.9,
        drift=DriftClassification.MINOR_DRIFT,
        tempo_reliable=True,
        rules=WarpAdviceRules(tolerance_ms=1.0),
    )
    lenient = recommend_warp(
        warp_map,
        grid_confidence=0.9,
        drift=DriftClassification.MINOR_DRIFT,
        tempo_reliable=True,
        rules=WarpAdviceRules(tolerance_ms=500.0),
    )
    assert strict.required is True
    assert lenient.required is False
