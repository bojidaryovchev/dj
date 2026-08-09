"""Quantized navigation: snapping, jumping, phrases and scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from dj_intelligence.timeline import Direction, Navigator, TempoMap, Unit

SPB = 60.0 / 126.0
FIRST_BEAT = 1.5


@pytest.fixture
def navigator() -> Navigator:
    tempo_map = TempoMap.from_beats(
        np.arange(400) * SPB + FIRST_BEAT, beats_per_bar=4, downbeat_beat=0
    )
    return Navigator(tempo_map, phrase_bars=8)


def beat_time(index: float) -> float:
    return FIRST_BEAT + index * SPB


# -- snapping ---------------------------------------------------------------


def test_snap_to_nearest_beat(navigator: Navigator) -> None:
    just_after = beat_time(40) + 0.05
    assert navigator.snap(just_after, Unit.BEAT).time == pytest.approx(beat_time(40))
    just_before = beat_time(40) - 0.05
    assert navigator.snap(just_before, Unit.BEAT).time == pytest.approx(beat_time(40))


def test_snap_next_and_previous_beat(navigator: Navigator) -> None:
    midway = beat_time(40) + 0.5 * SPB
    assert navigator.snap(midway, Unit.BEAT, Direction.NEXT).time == pytest.approx(beat_time(41))
    assert navigator.snap(midway, Unit.BEAT, Direction.PREVIOUS).time == pytest.approx(
        beat_time(40)
    )


def test_snap_next_from_exactly_on_a_boundary_moves_on(navigator: Navigator) -> None:
    """Pressing "next bar" on a bar line means the *following* bar."""
    on_the_line = navigator.tempo_map.bar_to_time(12)
    assert navigator.snap(on_the_line, Unit.BAR, Direction.NEXT).bar == 13
    assert navigator.snap(on_the_line, Unit.BAR, Direction.PREVIOUS).bar == 11
    assert navigator.snap(on_the_line, Unit.BAR, Direction.NEAREST).bar == 12


def test_snap_to_bar_lands_on_beat_one(navigator: Navigator) -> None:
    for time in np.linspace(5.0, 90.0, 40):
        snapped = navigator.snap(time, Unit.BAR)
        assert snapped.beat_in_bar == 1


def test_snap_to_phrase_lands_on_a_phrase_boundary(navigator: Navigator) -> None:
    snapped = navigator.snap(63.0, Unit.PHRASE)
    assert snapped.bar is not None
    assert snapped.bar % 8 == 0


def test_snapping_is_idempotent(navigator: Navigator) -> None:
    once = navigator.snap(47.3, Unit.BAR)
    twice = navigator.snap(once.time, Unit.BAR)
    assert twice.time == pytest.approx(once.time)


# -- jumping ----------------------------------------------------------------


def test_jump_beats(navigator: Navigator) -> None:
    assert navigator.jump_beats(beat_time(10), 4).time == pytest.approx(beat_time(14))
    assert navigator.jump_beats(beat_time(10), -4).time == pytest.approx(beat_time(6))


@pytest.mark.parametrize("bars", [1, -1, 8, 16, 32, -8])
def test_jump_bars_moves_exactly_that_many(navigator: Navigator, bars: int) -> None:
    start = navigator.tempo_map.bar_to_time(40)
    landed = navigator.jump_bars(start, bars)
    assert landed.bar == 40 + bars
    assert landed.beat_in_bar == 1


def test_jumping_preserves_rhythmic_phase(navigator: Navigator) -> None:
    """
    The whole point of jumping in musical units.

    Starting 40% through beat 3, a 16-bar jump must land 40% through beat 3.
    Landing on the bar line instead would silently shift a loop every time.
    """
    start = navigator.tempo_map.bar_to_time(20, 3) + 0.4 * SPB
    landed = navigator.jump_bars(start, 16)
    before = navigator.tempo_map.time_to_beat(start) % 1
    after = navigator.tempo_map.time_to_beat(landed.time) % 1
    assert after == pytest.approx(before, abs=1e-9)
    assert landed.beat_in_bar == 3


def test_jump_phrases(navigator: Navigator) -> None:
    start = navigator.tempo_map.bar_to_time(0)
    assert navigator.jump_phrases(start, 2).bar == 16


def test_bar_jumps_refuse_without_a_phase() -> None:
    plain = Navigator(TempoMap.from_beats(np.arange(64) * SPB))
    with pytest.raises(ValueError, match="bar phase"):
        plain.jump_bars(1.0, 4)
    # Beat-level navigation still works without bars.
    assert plain.jump_beats(1.0, 4).time == pytest.approx(1.0 + 4 * SPB)


# -- phrases ----------------------------------------------------------------


def test_phrase_grid_tiles_from_the_first_downbeat(navigator: Navigator) -> None:
    windows = navigator.phrase_grid(120.0)
    assert windows[0].start_bar == 0
    assert all(window.bars == 8 for window in windows)
    for index, window in enumerate(windows[1:], start=1):
        assert window.start_bar == index * 8
        assert window.start_time == pytest.approx(windows[index - 1].end_time, abs=1e-3)


def test_phrase_grid_respects_a_custom_size(navigator: Navigator) -> None:
    windows = navigator.phrase_grid(120.0, phrase_bars=16)
    assert all(window.bars == 16 for window in windows)


def test_phrase_grid_is_empty_without_bars() -> None:
    plain = Navigator(TempoMap.from_beats(np.arange(64) * SPB))
    assert plain.phrase_grid(30.0) == []


@pytest.mark.parametrize("bars", [4, 8, 16, 32])
def test_next_boundary_is_a_multiple(navigator: Navigator, bars: int) -> None:
    boundary = navigator.next_boundary(37.0, bars=bars)
    assert boundary.bar is not None
    assert boundary.bar % bars == 0
    assert boundary.time > 37.0


# -- scheduling -------------------------------------------------------------


def test_schedule_defers_to_the_next_boundary(navigator: Navigator) -> None:
    action = navigator.schedule(91.320, 122.222, quantization=Unit.BAR)
    assert action.requested_at == 91.320
    assert action.execute_at > 91.320
    # Both ends are quantised: executing on the bar but landing mid-bar would
    # still break phase.
    assert navigator.position_at(action.execute_at).beat_in_bar == 1
    assert action.destination_position.beat_in_bar == 1
    assert action.quantization is Unit.BAR


def test_schedule_can_quantize_to_beats(navigator: Navigator) -> None:
    action = navigator.schedule(50.0, 80.0, quantization=Unit.BEAT)
    assert action.execute_at == pytest.approx(navigator.snap(50.0, Unit.BEAT, Direction.NEXT).time)


def test_position_reports_both_coordinate_systems(navigator: Navigator) -> None:
    position = navigator.position_at(navigator.tempo_map.bar_to_time(9, 2))
    assert position.bar == 9
    assert position.beat_in_bar == 2
    assert "bar 9 beat 2" in str(position)
