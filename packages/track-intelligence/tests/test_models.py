"""Result models: serialisation, spelling, and refusing to invent values."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dj_intelligence.config import Settings
from dj_intelligence.models import (
    ConfidenceType,
    KeyCandidate,
    KeyEstimate,
    TempoEstimate,
    TonalSegment,
)
from dj_intelligence.music.notes import Mode
from dj_intelligence.version import ANALYSIS_VERSION, SCHEMA_VERSION


def test_key_estimate_spells_its_tonic() -> None:
    estimate = KeyEstimate(
        pitch_class=8,
        mode=Mode.MINOR,
        confidence=0.7,
        confidence_type=ConfidenceType.KEY_PROFILE_CORRELATION,
        reliable=True,
    )
    assert estimate.key == "G#"  # minor spelling, not Ab


def test_unknown_key_is_null_everywhere_not_defaulted() -> None:
    """The single most important behaviour in the schema: a failed key
    detection must not look like C major."""
    estimate = KeyEstimate.unknown(confidence=0.12)
    assert estimate.key is None
    assert estimate.pitch_class is None
    assert estimate.mode is None
    assert estimate.reliable is False
    assert estimate.confidence == 0.12
    assert estimate.confidence_type is ConfidenceType.NONE

    payload = json.loads(estimate.model_dump_json())
    assert payload["key"] is None and payload["mode"] is None


def test_unknown_tempo_is_null() -> None:
    tempo = TempoEstimate.unknown()
    assert tempo.bpm is None
    assert tempo.reliable is False
    assert tempo.candidates == []


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        KeyEstimate(confidence=1.5, confidence_type=ConfidenceType.NONE, reliable=False)
    with pytest.raises(ValidationError):
        KeyEstimate(confidence=-0.1, confidence_type=ConfidenceType.NONE, reliable=False)


def test_pitch_class_is_bounded() -> None:
    with pytest.raises(ValidationError):
        KeyCandidate(pitch_class=12, mode=Mode.MAJOR, score=0.5)


def test_models_reject_unknown_fields() -> None:
    """Typos in a payload fail loudly rather than being silently dropped."""
    with pytest.raises(ValidationError):
        KeyEstimate(
            confidence=0.5,
            confidence_type=ConfidenceType.NONE,
            reliable=False,
            camelot="4A",  # not a field here -- Camelot lives in the DJ layer
        )


def test_segment_duration() -> None:
    segment = TonalSegment(
        start_seconds=32.0,
        end_seconds=96.0,
        pitch_class=5,
        mode=Mode.MINOR,
        confidence=0.8,
        reliable=True,
    )
    assert segment.duration_seconds == 64.0
    assert segment.key == "F"


def test_enums_serialise_as_their_values() -> None:
    estimate = KeyEstimate(
        pitch_class=5,
        mode=Mode.MINOR,
        confidence=0.8,
        confidence_type=ConfidenceType.ESSENTIA_KEY_STRENGTH,
        reliable=True,
    )
    payload = json.loads(estimate.model_dump_json())
    assert payload["mode"] == "minor"
    assert payload["confidence_type"] == "essentia_key_strength"


# -- versioning and determinism --------------------------------------------


def test_versions_are_semver_shaped() -> None:
    assert ANALYSIS_VERSION.count(".") == 2
    assert SCHEMA_VERSION.count(".") == 1


def test_fingerprint_is_stable_for_equal_settings() -> None:
    assert Settings().analysis_fingerprint == Settings().analysis_fingerprint


def test_fingerprint_tracks_analysis_settings() -> None:
    baseline = Settings().analysis_fingerprint
    assert Settings(key_profile="temperley").analysis_fingerprint != baseline
    assert Settings(segment_window_seconds=15.0).analysis_fingerprint != baseline
    assert Settings(key_min_reliability=0.9).analysis_fingerprint != baseline


def test_fingerprint_ignores_transport_settings() -> None:
    """Moving the API to another port must not invalidate a library."""
    baseline = Settings().analysis_fingerprint
    assert Settings(port=9999).analysis_fingerprint == baseline
    assert Settings(log_level="DEBUG").analysis_fingerprint == baseline
    assert Settings(max_upload_bytes=1).analysis_fingerprint == baseline


def test_settings_validate_their_ranges() -> None:
    with pytest.raises(ValidationError):
        Settings(dj_bpm_min=180.0, dj_bpm_max=70.0)
    with pytest.raises(ValidationError):
        Settings(log_level="LOUD")
    with pytest.raises(ValidationError):
        Settings(key_min_reliability=2.0)
