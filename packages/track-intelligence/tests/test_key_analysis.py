"""The chroma key analyser and the profile scoring underneath it."""

from __future__ import annotations

import numpy as np
import pytest

from dj_intelligence.analysis.key.chroma import ChromaKeyAnalyzer, _tonal_salience
from dj_intelligence.analysis.key.profiles import PROFILES, KeyScores
from dj_intelligence.models import ConfidenceType
from dj_intelligence.music.notes import Mode

# -- profile scoring (no audio) --------------------------------------------


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_a_profile_scores_highest_against_itself(profile: str) -> None:
    """The sanity check for the whole approach: feed a key's own template in
    and that key must win, for every key and every profile."""
    scores = KeyScores(profile)
    major, minor = PROFILES[profile]
    for tonic in range(12):
        for mode, template in ((Mode.MAJOR, major), (Mode.MINOR, minor)):
            rotated = np.roll(np.asarray(template), tonic)
            correlations = scores.correlate(rotated)
            best = int(np.argmax(correlations))
            assert KeyScores.pitch_class_of(best) == tonic
            assert scores.mode_of(best) is mode
            assert correlations[best] == pytest.approx(1.0)


def test_correlation_is_bounded() -> None:
    scores = KeyScores("edma")
    rng = np.random.default_rng(0)
    for _ in range(50):
        correlations = scores.correlate(rng.random(12))
        assert np.all(correlations >= -1.0001) and np.all(correlations <= 1.0001)


def test_a_flat_vector_correlates_with_nothing() -> None:
    assert np.all(KeyScores("edma").correlate(np.ones(12)) == 0.0)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown key profile"):
        KeyScores("definitely-not-a-profile")


def test_every_profile_has_two_twelve_element_templates() -> None:
    for major, minor in PROFILES.values():
        assert len(major) == len(minor) == 12
        assert all(weight >= 0 for weight in (*major, *minor))


# -- tonal salience ---------------------------------------------------------


def test_a_flat_chroma_has_no_salience() -> None:
    assert _tonal_salience(np.ones(12)) == pytest.approx(0.0, abs=1e-12)


def test_a_single_pitch_class_is_maximally_salient() -> None:
    spike = np.zeros(12)
    spike[3] = 1.0
    assert _tonal_salience(spike) == pytest.approx(1.0)


def test_salience_rises_as_energy_concentrates() -> None:
    flat = np.ones(12)
    slight = np.ones(12)
    slight[0] = 1.5
    strong = np.ones(12)
    strong[[0, 4, 7]] = 6.0
    assert _tonal_salience(flat) < _tonal_salience(slight) < _tonal_salience(strong)


def test_salience_of_nothing_is_zero() -> None:
    assert _tonal_salience(np.zeros(12)) == 0.0


# -- the analyser -----------------------------------------------------------


def make_chroma(pitch_classes: list[int], frames: int = 100) -> np.ndarray:
    """A chromagram with energy on the given pitch classes."""
    chroma = np.full((12, frames), 0.05)
    for pitch_class in pitch_classes:
        chroma[pitch_class, :] = 1.0
    return chroma


def test_a_clear_triad_is_identified() -> None:
    analyzer = ChromaKeyAnalyzer()
    # F minor triad: F, Ab, C -> 5, 8, 0
    estimate = analyzer.estimate_from_chroma(make_chroma([5, 8, 0]))
    assert estimate.pitch_class == 5
    assert estimate.mode is Mode.MINOR
    assert estimate.confidence_type is ConfidenceType.KEY_PROFILE_CORRELATION


def test_flat_chroma_yields_no_key_however_well_a_template_fits() -> None:
    """The regression test for the noise bug: correlation ignores magnitude,
    so only the salience gate stops a key being invented here."""
    analyzer = ChromaKeyAnalyzer()
    rng = np.random.default_rng(1)
    # Essentially flat, with a whisper of structure -- enough for correlation
    # to pick a favourite, nowhere near enough to be a key.
    chroma = np.ones((12, 200)) + rng.normal(0.0, 0.002, size=(12, 200))
    assert analyzer.estimate_from_chroma(chroma).key is None


def test_the_gate_is_configurable() -> None:
    chroma = np.ones((12, 50))
    chroma[7, :] = 1.02
    assert ChromaKeyAnalyzer(min_tonal_salience=0.5).estimate_from_chroma(chroma).key is None
    assert ChromaKeyAnalyzer(min_tonal_salience=0.0).estimate_from_chroma(chroma).key is not None


def test_empty_chroma_yields_no_key() -> None:
    assert ChromaKeyAnalyzer().estimate_from_chroma(np.zeros((12, 0))).key is None


def test_alternatives_are_ranked_below_the_winner() -> None:
    estimate = ChromaKeyAnalyzer().estimate_from_chroma(make_chroma([5, 8, 0]))
    assert estimate.alternatives
    scores = [candidate.score for candidate in estimate.alternatives]
    assert scores == sorted(scores, reverse=True)
    assert estimate.confidence >= scores[0]


def test_reliability_respects_the_threshold() -> None:
    chroma = make_chroma([5, 8, 0])
    assert ChromaKeyAnalyzer(min_reliability=0.99).estimate_from_chroma(chroma).reliable is False
    assert ChromaKeyAnalyzer(min_reliability=0.10).estimate_from_chroma(chroma).reliable is True


def test_profile_choice_is_recorded() -> None:
    analyzer = ChromaKeyAnalyzer(profile="temperley")
    assert analyzer.parameters()["profile"] == "temperley"
    assert analyzer.describe().name == "chroma"
    assert "librosa" in analyzer.describe().algorithm


def test_bad_aggregate_is_rejected() -> None:
    with pytest.raises(ValueError, match="median"):
        ChromaKeyAnalyzer(aggregate="mode")
