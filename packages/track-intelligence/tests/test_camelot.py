"""
The Camelot mapping, exhaustively.

The implementation *derives* the wheel from the circle of fifths. These tests
assert the literal 24-row table from the Camelot specification against it, so
the arithmetic and the table are independent statements of the same fact --
which is the only arrangement in which a test of a formula is worth anything.
"""

from __future__ import annotations

import pytest

from dj_intelligence.music.camelot import WHEEL_POSITIONS, CamelotKey
from dj_intelligence.music.notes import InvalidKeyError, Mode, canonical_key_name, parse_pitch_class

# Transcribed from the specification, both spellings where it lists two.
MINOR_KEYS = [
    ("G#", "1A"),
    ("Ab", "1A"),
    ("D#", "2A"),
    ("Eb", "2A"),
    ("A#", "3A"),
    ("Bb", "3A"),
    ("F", "4A"),
    ("C", "5A"),
    ("G", "6A"),
    ("D", "7A"),
    ("A", "8A"),
    ("E", "9A"),
    ("B", "10A"),
    ("F#", "11A"),
    ("Gb", "11A"),
    ("C#", "12A"),
    ("Db", "12A"),
]

MAJOR_KEYS = [
    ("B", "1B"),
    ("F#", "2B"),
    ("Gb", "2B"),
    ("C#", "3B"),
    ("Db", "3B"),
    ("G#", "4B"),
    ("Ab", "4B"),
    ("D#", "5B"),
    ("Eb", "5B"),
    ("A#", "6B"),
    ("Bb", "6B"),
    ("F", "7B"),
    ("C", "8B"),
    ("G", "9B"),
    ("D", "10B"),
    ("A", "11B"),
    ("E", "12B"),
]


@pytest.mark.parametrize(("tonic", "expected"), MINOR_KEYS)
def test_minor_keys_map_to_the_specified_position(tonic: str, expected: str) -> None:
    assert CamelotKey.from_key(tonic, Mode.MINOR).notation == expected


@pytest.mark.parametrize(("tonic", "expected"), MAJOR_KEYS)
def test_major_keys_map_to_the_specified_position(tonic: str, expected: str) -> None:
    assert CamelotKey.from_key(tonic, Mode.MAJOR).notation == expected


def test_every_wheel_position_is_covered_exactly_once() -> None:
    """All 24 positions are reachable and no two keys collide."""
    produced = {
        CamelotKey.from_key(pitch_class, mode).notation
        for pitch_class in range(12)
        for mode in Mode
    }
    expected = {f"{n}{letter}" for n in range(1, WHEEL_POSITIONS + 1) for letter in "AB"}
    assert produced == expected


@pytest.mark.parametrize(
    ("left", "right", "mode"),
    [
        ("Ab", "G#", Mode.MAJOR),
        ("Db", "C#", Mode.MAJOR),
        ("Gb", "F#", Mode.MAJOR),
        ("Eb", "D#", Mode.MINOR),
        ("Bb", "A#", Mode.MINOR),
        ("Cb", "B", Mode.MAJOR),  # crosses the octave
        ("B#", "C", Mode.MAJOR),  # and back
        ("E#", "F", Mode.MINOR),
        ("Fb", "E", Mode.MINOR),
    ],
)
def test_enharmonic_spellings_collapse(left: str, right: str, mode: Mode) -> None:
    assert CamelotKey.from_key(left, mode) == CamelotKey.from_key(right, mode)


def test_unicode_accidentals_are_accepted() -> None:
    assert parse_pitch_class("A♭") == parse_pitch_class("Ab")
    assert parse_pitch_class("F♯") == parse_pitch_class("F#")


@pytest.mark.parametrize("notation", ["4A", "12B", "1a", " 7b ", "10A"])
def test_notation_round_trips(notation: str) -> None:
    key = CamelotKey.parse(notation)
    assert CamelotKey.from_key(key.tonic, key.mode) == key


@pytest.mark.parametrize("bad", ["", "H", "13A", "0A", "4C", "A4", "4", "--", "4AA"])
def test_invalid_notation_is_rejected(bad: str) -> None:
    with pytest.raises(InvalidKeyError):
        CamelotKey.parse(bad)


@pytest.mark.parametrize("bad", ["", "H", "Cbb#", "8", "sharp"])
def test_invalid_note_names_are_rejected(bad: str) -> None:
    with pytest.raises(InvalidKeyError):
        parse_pitch_class(bad)


def test_position_out_of_range_is_rejected() -> None:
    with pytest.raises(InvalidKeyError):
        CamelotKey(position=0, mode=Mode.MINOR)
    with pytest.raises(InvalidKeyError):
        CamelotKey(position=13, mode=Mode.MAJOR)


# -- spelling ---------------------------------------------------------------


def test_spelling_follows_the_key_signature_not_the_pitch_class() -> None:
    """Pitch class 8 is Ab in major and G# in minor -- as every DJ app shows."""
    assert canonical_key_name(8, Mode.MAJOR) == "Ab"
    assert canonical_key_name(8, Mode.MINOR) == "G#"
    assert CamelotKey.parse("4B").key_name == "Ab major"
    assert CamelotKey.parse("1A").key_name == "G# minor"


def test_tonic_matches_the_specification_labels() -> None:
    assert CamelotKey.parse("4A").tonic == "F"
    assert CamelotKey.parse("8B").tonic == "C"
    assert CamelotKey.parse("2A").tonic == "Eb"
    assert CamelotKey.parse("11A").tonic == "F#"


# -- wheel arithmetic -------------------------------------------------------


def test_shifting_wraps_around_the_wheel() -> None:
    assert CamelotKey.parse("12A").shifted(1).notation == "1A"
    assert CamelotKey.parse("1A").shifted(-1).notation == "12A"
    assert CamelotKey.parse("4A").shifted(12).notation == "4A"


def test_shifting_preserves_mode() -> None:
    for position in range(1, 13):
        key = CamelotKey(position=position, mode=Mode.MINOR)
        assert key.shifted(3).mode is Mode.MINOR


def test_flipping_gives_the_relative_key() -> None:
    """Relative major/minor share a key signature: A minor and C major."""
    a_minor = CamelotKey.from_key("A", Mode.MINOR)
    assert a_minor.flipped().key_name == "C major"
    assert a_minor.flipped().flipped() == a_minor


def test_relative_keys_are_three_semitones_apart() -> None:
    for position in range(1, 13):
        minor = CamelotKey(position=position, mode=Mode.MINOR)
        major = minor.flipped()
        assert (major.pitch_class - minor.pitch_class) % 12 == 3


def test_adjacent_positions_are_a_fifth_apart() -> None:
    """One step clockwise is up a perfect fifth. That is what the wheel is."""
    for position in range(1, 13):
        key = CamelotKey(position=position, mode=Mode.MAJOR)
        assert (key.shifted(1).pitch_class - key.pitch_class) % 12 == 7


def test_distance_takes_the_short_way_round() -> None:
    assert CamelotKey.parse("12A").distance(CamelotKey.parse("1A")) == 1
    assert CamelotKey.parse("1A").distance(CamelotKey.parse("7A")) == 6
    assert CamelotKey.parse("4A").distance(CamelotKey.parse("4B")) == 0
    for position in range(1, 13):
        key = CamelotKey(position=position, mode=Mode.MAJOR)
        assert 0 <= key.distance(CamelotKey.parse("1A")) <= 6


def test_keys_are_hashable_and_comparable() -> None:
    assert len({CamelotKey.parse("4A"), CamelotKey.from_key("F", Mode.MINOR)}) == 1
    assert CamelotKey.parse("4A") != CamelotKey.parse("4B")
