"""Tempo statistics and metrical candidates -- the parts with no audio in them."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from dj_intelligence.analysis.tempo.common import (
    bpm_from_beats,
    consistency_confidence,
    interval_stats,
    tempo_candidates,
)
from dj_intelligence.dj.interpret import preferred_mix_bpm
from dj_intelligence.models import ConfidenceType, TempoEstimate, TempoRelation


def grid(bpm: float, count: int, jitter: float = 0.0, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    interval = 60.0 / bpm
    times = np.arange(count) * interval
    if jitter:
        times = times + rng.normal(0.0, jitter * interval, size=count)
    return [float(t) for t in np.sort(times)]


def test_bpm_recovered_from_a_perfect_grid() -> None:
    assert bpm_from_beats(grid(126.0, 100)) == pytest.approx(126.0)


def test_a_perfect_grid_has_no_spread() -> None:
    _, cv = interval_stats(grid(128.0, 100))
    assert cv == pytest.approx(0.0, abs=1e-9)
    assert consistency_confidence(cv) == pytest.approx(1.0)


def test_jitter_lowers_confidence() -> None:
    _, steady = interval_stats(grid(128.0, 200, jitter=0.005))
    _, loose = interval_stats(grid(128.0, 200, jitter=0.08))
    assert consistency_confidence(steady) > consistency_confidence(loose)


def test_a_dropped_beat_does_not_wreck_the_tempo() -> None:
    """One missing beat doubles an interval. The median must ignore it."""
    beats = grid(128.0, 60)
    del beats[30]
    assert bpm_from_beats(beats) == pytest.approx(128.0)
    _, cv = interval_stats(beats)
    assert cv == pytest.approx(0.0, abs=1e-6)  # trimmed as a tracker artefact


def test_bpm_survives_frame_quantised_beats() -> None:
    """
    The regression test for a 2.4% tempo error.

    Beat trackers place beats on onset frames. At librosa's default hop that
    is 23.2 ms, so a 126 BPM beat lasts 20.5 frames and every interval has to
    round to 20 (129.20 BPM) or 21 (123.05 BPM). Taking the median interval
    picks one of those and is wrong by up to 2.4%; fitting across the whole
    grid recovers the true tempo from the mixture.
    """
    frame = 512 / 22050
    true_interval = 60.0 / 126.0
    beats = [round(i * true_interval / frame) * frame for i in range(200)]

    intervals = {round(b - a, 6) for a, b in itertools.pairwise(beats)}
    assert len(intervals) > 1, "fixture should contain both 20- and 21-frame gaps"

    assert bpm_from_beats(beats) == pytest.approx(126.0, rel=0.001)


def test_bpm_fit_is_not_dragged_off_by_a_biased_median() -> None:
    """
    Beat indices accumulate per interval, not from absolute position.

    Position-based indexing divides by the (quantised, biased) median, and
    over a few minutes the bias compounds until beats land on the wrong index
    and the fit is worse than the median it replaced.
    """
    frame = 256 / 22050
    true_interval = 60.0 / 128.0  # 40.4 frames: median rounds to 40, a 1% bias
    beats = [round(i * true_interval / frame) * frame for i in range(400)]
    assert bpm_from_beats(beats) == pytest.approx(128.0, rel=0.001)


def test_bpm_fit_tolerates_dropped_beats() -> None:
    beats = grid(124.0, 120)
    for index in (90, 60, 30):  # descending, so earlier deletes stay valid
        del beats[index]
    assert bpm_from_beats(beats) == pytest.approx(124.0, rel=0.002)


def test_too_few_beats_says_so() -> None:
    assert interval_stats([]) == (None, None)
    assert interval_stats([0.0, 0.5]) == (None, None)
    assert bpm_from_beats([0.0, 0.5]) is None


def test_confidence_of_nothing_is_zero() -> None:
    assert consistency_confidence(None) == 0.0
    assert consistency_confidence(10.0) == 0.0


def test_candidates_cover_half_and_double() -> None:
    candidates = tempo_candidates(63.0, dj_min=70.0, dj_max=180.0)
    by_relation = {c.relation: c for c in candidates}
    assert by_relation[TempoRelation.PRIMARY].bpm == 63.0
    assert by_relation[TempoRelation.DOUBLE_TIME].bpm == 126.0
    assert by_relation[TempoRelation.HALF_TIME].bpm == 31.5
    assert by_relation[TempoRelation.PRIMARY].in_dj_range is False
    assert by_relation[TempoRelation.DOUBLE_TIME].in_dj_range is True


# -- interpretation ---------------------------------------------------------


def estimate(bpm: float) -> TempoEstimate:
    return TempoEstimate(
        bpm=bpm,
        confidence=0.9,
        confidence_type=ConfidenceType.BEAT_INTERVAL_CONSISTENCY,
        reliable=True,
        candidates=tempo_candidates(bpm, dj_min=70.0, dj_max=180.0),
    )


def test_a_normal_tempo_is_left_alone() -> None:
    bpm, relation = preferred_mix_bpm(estimate(126.04), dj_bpm_min=70.0, dj_bpm_max=180.0)
    assert bpm == 126.04
    assert relation is TempoRelation.PRIMARY


@pytest.mark.parametrize(("measured", "expected"), [(63.0, 126.0), (70.0, 70.0), (35.0, 70.0)])
def test_out_of_range_tempos_are_folded_not_measured_differently(
    measured: float, expected: float
) -> None:
    bpm, _ = preferred_mix_bpm(estimate(measured), dj_bpm_min=70.0, dj_bpm_max=180.0)
    assert bpm == pytest.approx(expected)


def test_folding_reports_which_reading_it_chose() -> None:
    _, relation = preferred_mix_bpm(estimate(63.0), dj_bpm_min=70.0, dj_bpm_max=180.0)
    assert relation is TempoRelation.DOUBLE_TIME


def test_measurement_is_never_overwritten() -> None:
    """dj.mix_bpm may differ from tempo.bpm; tempo.bpm stays as measured."""
    measured = estimate(63.0)
    preferred_mix_bpm(measured, dj_bpm_min=70.0, dj_bpm_max=180.0)
    assert measured.bpm == 63.0


def test_unknown_tempo_folds_to_nothing() -> None:
    assert preferred_mix_bpm(TempoEstimate.unknown(), dj_bpm_min=70.0, dj_bpm_max=180.0) == (
        None,
        None,
    )
