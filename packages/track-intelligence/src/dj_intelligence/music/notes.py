"""
Pitch classes, enharmonic normalisation and conventional key spelling.

Pure music theory: no audio, no numpy, no configuration. Everything here is a
total function of its arguments, which is what makes the Camelot mapping
testable without touching a decoder.

A *pitch class* is an integer 0-11 with C = 0. Two spellings that sound the
same (G# and Ab) share a pitch class; the difference between them is
orthographic, and orthography is what we normalise away before mapping to
Camelot.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

__all__ = [
    "PITCH_CLASS_COUNT",
    "InvalidKeyError",
    "Mode",
    "canonical_key_name",
    "parse_mode",
    "parse_pitch_class",
]

PITCH_CLASS_COUNT: Final = 12


class Mode(StrEnum):
    """Major or minor. The only two modes the Camelot wheel represents."""

    MAJOR = "major"
    MINOR = "minor"


class InvalidKeyError(ValueError):
    """Raised when a string cannot be read as a musical key or mode."""


# Natural note letters to pitch class.
_LETTER_TO_PC: Final[dict[str, int]] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# Accept the unicode accidentals as well as ASCII; tag editors emit both.
_ACCIDENTAL_OFFSET: Final[dict[str, int]] = {
    "#": 1,
    "♯": 1,  # ♯
    "b": -1,
    "♭": -1,  # ♭
    "x": 2,  # double sharp, occasionally seen from notation software
}

_NOTE_RE: Final = re.compile(r"^([A-Ga-g])([#b♯♭x]*)$")

_MODE_ALIASES: Final[dict[str, Mode]] = {
    "major": Mode.MAJOR,
    "maj": Mode.MAJOR,
    "M": Mode.MAJOR,
    "ionian": Mode.MAJOR,
    "minor": Mode.MINOR,
    "min": Mode.MINOR,
    "m": Mode.MINOR,
    "aeolian": Mode.MINOR,
}


def parse_pitch_class(name: str) -> int:
    """
    Read a note name into a pitch class 0-11.

    Accepts any real accidental spelling, including ones that cross the
    octave boundary such as ``Cb`` (= B, pitch class 11) and ``B#`` (= C, 0),
    and doubles such as ``Cbb`` and ``F##``.

        >>> parse_pitch_class("Ab") == parse_pitch_class("G#")
        True

    Rejects notation no musician writes: sharps and flats mixed on one note
    (``Cbb#``), or more than a double accidental. Both are well-defined as
    arithmetic and meaningless as music, and quietly accepting them would
    turn a caller's parsing bug into a plausible wrong key.
    """
    match = _NOTE_RE.match(name.strip())
    if match is None:
        raise InvalidKeyError(f"not a note name: {name!r}")

    letter, accidentals = match.groups()
    offsets = [_ACCIDENTAL_OFFSET[accidental] for accidental in accidentals]
    if any(offset > 0 for offset in offsets) and any(offset < 0 for offset in offsets):
        raise InvalidKeyError(f"note name mixes sharps and flats: {name!r}")

    total = sum(offsets)
    if abs(total) > 2:
        raise InvalidKeyError(f"more than a double accidental: {name!r}")

    return (_LETTER_TO_PC[letter.upper()] + total) % PITCH_CLASS_COUNT


def parse_mode(name: str) -> Mode:
    """Read a mode name. Case-insensitive except for the bare ``M``/``m``."""
    stripped = name.strip()
    if stripped in _MODE_ALIASES:  # exact first, so "M" != "m"
        return _MODE_ALIASES[stripped]
    lowered = stripped.lower()
    if lowered in _MODE_ALIASES:
        return _MODE_ALIASES[lowered]
    raise InvalidKeyError(f"not a mode: {name!r}")


# The spelling DJ software actually displays for each key.
#
# Neither "all sharps" nor "all flats" matches what a DJ sees: Rekordbox,
# Serato and Mixed In Key all use the spelling with the fewest accidentals in
# the key signature, which is why 2A reads Ebm but 11A reads F#m. These two
# tables are that convention, indexed by pitch class.
_MAJOR_SPELLING: Final[tuple[str, ...]] = (
    "C",  # 0   8B
    "Db",  # 1   3B   (5 flats beats 7 sharps)
    "D",  # 2   10B
    "Eb",  # 3   5B
    "E",  # 4   12B
    "F",  # 5   7B
    "F#",  # 6   2B   (6 sharps ties 6 flats; F# is the common label)
    "G",  # 7   9B
    "Ab",  # 8   4B
    "A",  # 9   11B
    "Bb",  # 10  6B
    "B",  # 11  1B
)

_MINOR_SPELLING: Final[tuple[str, ...]] = (
    "C",  # 0   5A
    "C#",  # 1   12A
    "D",  # 2   7A
    "Eb",  # 3   2A
    "E",  # 4   9A
    "F",  # 5   4A
    "F#",  # 6   11A
    "G",  # 7   6A
    "G#",  # 8   1A
    "A",  # 9   8A
    "Bb",  # 10  3A
    "B",  # 11  10A
)


def canonical_key_name(pitch_class: int, mode: Mode) -> str:
    """
    The conventional spelling of a key's tonic, e.g. ``(8, MINOR) -> "G#"``.

    Mode matters: pitch class 8 is ``Ab`` in major (4B) but ``G#`` in minor
    (1A), because that is what the key signatures say and what every DJ
    application prints.
    """
    pc = pitch_class % PITCH_CLASS_COUNT
    table = _MAJOR_SPELLING if mode is Mode.MAJOR else _MINOR_SPELLING
    return table[pc]
