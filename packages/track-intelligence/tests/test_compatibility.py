"""Track-to-track compatibility scoring."""

from __future__ import annotations

import pytest

from dj_intelligence.dj.compatibility import RULES_VERSION, ScoringRules, score_pair
from dj_intelligence.models.analysis import TempoRelation
from dj_intelligence.models.compatibility import TrackReference


def ref(camelot: str | None = None, bpm: float | None = None) -> TrackReference:
    return TrackReference(camelot=camelot, bpm=bpm)


def test_the_specification_example() -> None:
    result = score_pair(ref("4A", 126), ref("5A", 126.5))
    assert result.comparable
    assert result.harmonic_relationship == "adjacent_plus"
    assert result.components.harmonic == pytest.approx(0.95)
    assert result.score > 0.9
    assert any("Adjacent" in reason for reason in result.reasons)
    assert any("0.4%" in reason for reason in result.reasons)


def test_identical_tracks_score_one() -> None:
    assert score_pair(ref("4A", 126), ref("4A", 126)).score == pytest.approx(1.0)


def test_scoring_is_deterministic() -> None:
    first = score_pair(ref("8A", 128), ref("9A", 129))
    second = score_pair(ref("8A", 128), ref("9A", 129))
    assert first == second


def test_harmonic_ordering_follows_the_rules() -> None:
    """Same key beats adjacent beats relative beats a two-step beats distant."""
    same = score_pair(ref("4A", 126), ref("4A", 126)).components.harmonic
    adjacent = score_pair(ref("4A", 126), ref("5A", 126)).components.harmonic
    relative = score_pair(ref("4A", 126), ref("4B", 126)).components.harmonic
    boost = score_pair(ref("4A", 126), ref("6A", 126)).components.harmonic
    distant = score_pair(ref("4A", 126), ref("10A", 126)).components.harmonic
    assert same > adjacent > relative > boost > distant


def test_tempo_score_falls_off_with_distance() -> None:
    close = score_pair(ref("4A", 126), ref("4A", 126.5)).components.tempo
    further = score_pair(ref("4A", 126), ref("4A", 129)).components.tempo
    beyond = score_pair(ref("4A", 126), ref("4A", 150)).components.tempo
    assert close > further > beyond
    assert beyond == 0.0


def test_relative_not_absolute_tempo_difference() -> None:
    """2 BPM at 174 is a smaller move than 2 BPM at 90, and scores higher."""
    fast = score_pair(ref("4A", 174), ref("4A", 176)).components.tempo
    slow = score_pair(ref("4A", 90), ref("4A", 92)).components.tempo
    assert fast > slow


def test_half_time_is_matched_and_penalised() -> None:
    result = score_pair(ref("4A", 174), ref("4A", 87))
    assert result.tempo_relation is TempoRelation.DOUBLE_TIME
    assert result.components.tempo == pytest.approx(ScoringRules().half_double_penalty)
    assert any("half-time" in r or "double-time" in r for r in result.reasons)
    # A straight match still wins over a metrical one.
    assert score_pair(ref("4A", 174), ref("4A", 174)).components.tempo > result.components.tempo


def test_missing_key_scores_on_tempo_alone() -> None:
    result = score_pair(ref(None, 126), ref("5A", 126))
    assert result.comparable
    assert result.components.harmonic is None
    assert result.components.tempo is not None
    # Renormalised, not diluted: a perfect tempo match is not dragged down
    # by a component we could not compute.
    assert result.score == pytest.approx(result.components.tempo)
    assert any("key unknown" in reason for reason in result.reasons)


def test_missing_tempo_scores_on_key_alone() -> None:
    result = score_pair(ref("4A"), ref("4A"))
    assert result.comparable
    assert result.components.tempo is None
    assert result.score == pytest.approx(1.0)


def test_nothing_to_compare_is_reported_not_guessed() -> None:
    result = score_pair(ref(), ref())
    assert result.comparable is False
    assert result.score == 0.0


def test_invalid_camelot_is_an_error() -> None:
    with pytest.raises(ValueError, match="Camelot"):
        score_pair(ref("99Z", 126), ref("4A", 126))


def test_rules_are_configurable() -> None:
    tempo_only = ScoringRules(harmonic_weight=0.0, tempo_weight=1.0)
    result = score_pair(ref("4A", 126), ref("10A", 126), tempo_only)
    assert result.score == pytest.approx(1.0)  # distant key, ignored by these rules


def test_result_records_its_rules_version() -> None:
    assert score_pair(ref("4A", 126), ref("4A", 126)).rules_version == RULES_VERSION


def test_scores_stay_in_range() -> None:
    for camelot in ("1A", "4A", "7B", "12B"):
        for bpm in (60.0, 100.0, 128.0, 174.0, 200.0):
            result = score_pair(ref("4A", 126), ref(camelot, bpm))
            assert 0.0 <= result.score <= 1.0
