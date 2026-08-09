"""
The tempo map: audio time <-> musical time.

This is the primitive everything rhythmic stands on, so it is tested harder
than anything else here. A bug in ``beat_to_time`` is a bug in every bar jump,
every phrase boundary and every warp marker in the system.
"""

from __future__ import annotations

import numpy as np
import pytest

from dj_intelligence.timeline import TempoMap

SPB = 60.0 / 126.0


@pytest.fixture
def steady() -> TempoMap:
    """126 BPM, 400 beats, first beat at 1.5 s, bar lines on beat 0."""
    return TempoMap.from_beats(np.arange(400) * SPB + 1.5, beats_per_bar=4, downbeat_beat=0)


@pytest.fixture
def drifting() -> TempoMap:
    """125 -> 127 BPM over 400 beats."""
    periods = 60.0 / np.linspace(125.0, 127.0, 400)
    times = np.concatenate([[1.5], 1.5 + np.cumsum(periods[:-1])])
    return TempoMap.from_beats(times, beats_per_bar=4, downbeat_beat=0)


# -- construction -----------------------------------------------------------


def test_needs_at_least_two_anchors() -> None:
    with pytest.raises(ValueError, match="at least two anchors"):
        TempoMap.from_beats([1.0])


def test_anchors_must_increase() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TempoMap.from_beats([1.0, 0.5, 2.0])


def test_constant_map_is_exact() -> None:
    tempo_map = TempoMap.constant(120.0, anchor_time=10.0, anchor_beat=4, beat_count=64)
    assert tempo_map.beat_to_time(4) == pytest.approx(10.0)
    assert tempo_map.beat_to_time(8) == pytest.approx(12.0)
    assert tempo_map.average_bpm == pytest.approx(120.0)


def test_constant_map_rejects_nonsense_tempo() -> None:
    with pytest.raises(ValueError, match="positive"):
        TempoMap.constant(0.0)


# -- the mapping ------------------------------------------------------------


def test_round_trips_to_floating_point_precision(steady: TempoMap) -> None:
    for time in np.linspace(0.0, 200.0, 400):
        assert steady.beat_to_time(steady.time_to_beat(time)) == pytest.approx(time, abs=1e-9)


def test_round_trips_on_a_drifting_grid(drifting: TempoMap) -> None:
    for beat in np.linspace(0, 399, 250):
        assert drifting.time_to_beat(drifting.beat_to_time(beat)) == pytest.approx(beat, abs=1e-9)


def test_positions_are_fractional(steady: TempoMap) -> None:
    """Musical position is continuous; a seek lands between beats."""
    assert steady.time_to_beat(1.5 + 0.42 * SPB) == pytest.approx(0.42)


def test_known_positions(steady: TempoMap) -> None:
    assert steady.time_to_beat(1.5) == pytest.approx(0.0)
    assert steady.beat_to_time(64) == pytest.approx(1.5 + 64 * SPB)


def test_extrapolates_before_the_first_beat(steady: TempoMap) -> None:
    """A track has audio before its first detected beat, and a DJ can seek
    into it. Beat -1 is a real position, not an error."""
    assert steady.beat_to_time(-4) == pytest.approx(1.5 - 4 * SPB)
    assert steady.time_to_beat(0.0) == pytest.approx(-1.5 / SPB)


def test_extrapolates_after_the_last_beat(steady: TempoMap) -> None:
    last = steady.last_beat
    assert steady.beat_to_time(last + 8) == pytest.approx(steady.last_time + 8 * SPB)


def test_local_bpm_follows_a_drift(drifting: TempoMap) -> None:
    early = drifting.local_bpm(drifting.first_time + 1.0)
    late = drifting.local_bpm(drifting.last_time - 1.0)
    assert early == pytest.approx(125.0, abs=0.5)
    assert late == pytest.approx(127.0, abs=0.5)
    assert late > early


def test_vectorised_matches_scalar(steady: TempoMap) -> None:
    times = np.linspace(2.0, 100.0, 50)
    assert np.allclose(steady.times_to_beats(times), [steady.time_to_beat(t) for t in times])


# -- musical indices --------------------------------------------------------


def test_a_dropped_beat_does_not_shift_the_bars() -> None:
    """
    The reason beat indices are supplied rather than inferred from position.

    If the tracker misses beat 5, numbering the survivors 0..n puts every
    later bar line one beat early for the rest of the track.
    """
    times = np.arange(64) * SPB
    kept = np.delete(times, 5)
    indices = np.delete(np.arange(64), 5)

    naive = TempoMap.from_beats(kept, beats_per_bar=4, downbeat_beat=0)
    correct = TempoMap.from_beats(kept, indices, beats_per_bar=4, downbeat_beat=0)

    assert correct.beat_to_time(60) == pytest.approx(60 * SPB, abs=1e-9)
    assert naive.beat_to_time(60) != pytest.approx(60 * SPB, abs=1e-3)


# -- bars -------------------------------------------------------------------


def test_bar_positions(steady: TempoMap) -> None:
    assert steady.beat_to_bar(0).bar == 0
    assert steady.beat_to_bar(0).beat_in_bar == 1
    assert steady.beat_to_bar(3).beat_in_bar == 4
    assert steady.beat_to_bar(4).bar == 1
    assert steady.beat_to_bar(64).bar == 16


def test_beat_in_bar_is_one_based(steady: TempoMap) -> None:
    for beat in range(16):
        position = steady.beat_to_bar(beat)
        assert 1 <= position.beat_in_bar <= 4


def test_bar_and_beat_round_trip(steady: TempoMap) -> None:
    for bar in range(0, 40):
        for beat_in_bar in range(1, 5):
            beat = steady.bar_to_beat(bar, beat_in_bar)
            position = steady.beat_to_bar(beat)
            assert (position.bar, position.beat_in_bar) == (bar, beat_in_bar)


def test_beats_before_the_first_downbeat_are_negative_bars() -> None:
    """A pickup belongs before bar zero, not on it."""
    tempo_map = TempoMap.from_beats(np.arange(64) * SPB, beats_per_bar=4, downbeat_beat=2)
    assert tempo_map.beat_to_bar(2).bar == 0
    assert tempo_map.beat_to_bar(0).bar == -1
    assert tempo_map.beat_to_bar(0).beat_in_bar == 3


def test_bar_operations_refuse_without_a_phase() -> None:
    """No silent 4/4 assumption: a wrong bar line is worse than none."""
    tempo_map = TempoMap.from_beats(np.arange(64) * SPB)
    assert tempo_map.has_bars is False
    with pytest.raises(ValueError, match="no bar phase"):
        tempo_map.beat_to_bar(4)
    with pytest.raises(ValueError, match="no bar phase"):
        tempo_map.bar_to_time(2)


def test_beat_in_bar_is_range_checked(steady: TempoMap) -> None:
    with pytest.raises(ValueError, match=r"1\.\.4"):
        steady.bar_to_beat(0, 5)


def test_bar_count(steady: TempoMap) -> None:
    assert steady.bar_count() == pytest.approx(99, abs=1)
