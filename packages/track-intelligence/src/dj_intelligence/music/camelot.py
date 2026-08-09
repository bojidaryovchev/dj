"""
The Camelot wheel: key <-> notation, and arithmetic on the wheel itself.

The 24 mappings are *derived*, not tabulated. The wheel is the circle of
fifths with the relative minor folded onto the same number, so:

    major pitch class of position n = (11 + 7 * (n - 1)) mod 12
    minor pitch class of position n = major - 3 semitones

which reproduces B major = 1B, F# major = 2B ... and G# minor = 1A, Eb minor
= 2A ... exactly. Tests assert the literal 24-row table from the Camelot
specification against these formulas, so a mistake in the arithmetic cannot
hide behind a matching hand-written table.

Positions run 1-12 and wrap; the letter is A for minor, B for major.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Self

from .notes import InvalidKeyError, Mode, canonical_key_name, parse_mode, parse_pitch_class

__all__ = ["WHEEL_POSITIONS", "CamelotKey"]

WHEEL_POSITIONS: Final = 12

# Position 1B is B major.
_MAJOR_PC_AT_POSITION_ONE: Final = 11
_FIFTH: Final = 7
# 7 is its own inverse mod 12 (7 * 7 = 49 = 1 mod 12), which is what makes the
# reverse mapping below a multiplication rather than a search.
_FIFTH_INVERSE: Final = 7
_RELATIVE_MINOR_OFFSET: Final = -3

_CAMELOT_RE: Final = re.compile(r"^\s*(\d{1,2})\s*([ABab])\s*$")


@dataclass(frozen=True, slots=True)
class CamelotKey:
    """
    A position on the Camelot wheel.

    Frozen and hashable so it can be a dict key in a compatibility matrix or
    a set member when de-duplicating neighbours.
    """

    position: int
    mode: Mode

    def __post_init__(self) -> None:
        if not 1 <= self.position <= WHEEL_POSITIONS:
            raise InvalidKeyError(f"Camelot position out of range 1-12: {self.position}")

    # -- construction ------------------------------------------------------

    @classmethod
    def from_key(cls, tonic: str | int, mode: Mode | str) -> Self:
        """
        Build from a musical key, e.g. ``("Ab", "major") -> 4B``.

        ``tonic`` is a note name in any enharmonic spelling, or a pitch class
        integer. Enharmonic equivalents collapse: ``Ab major`` and ``G# major``
        both give 4B.
        """
        pitch_class = tonic if isinstance(tonic, int) else parse_pitch_class(tonic)
        resolved_mode = mode if isinstance(mode, Mode) else parse_mode(mode)

        major_pc = (
            pitch_class if resolved_mode is Mode.MAJOR else pitch_class - _RELATIVE_MINOR_OFFSET
        )
        position = ((major_pc - _MAJOR_PC_AT_POSITION_ONE) * _FIFTH_INVERSE) % WHEEL_POSITIONS + 1
        return cls(position=position, mode=resolved_mode)

    @classmethod
    def parse(cls, notation: str) -> Self:
        """Read ``"4A"`` / ``"12b"`` into a CamelotKey."""
        match = _CAMELOT_RE.match(notation)
        if match is None:
            raise InvalidKeyError(f"not Camelot notation: {notation!r}")
        position = int(match.group(1))
        if not 1 <= position <= WHEEL_POSITIONS:
            raise InvalidKeyError(f"Camelot position out of range 1-12: {notation!r}")
        mode = Mode.MINOR if match.group(2).upper() == "A" else Mode.MAJOR
        return cls(position=position, mode=mode)

    # -- projection back to music -----------------------------------------

    @property
    def pitch_class(self) -> int:
        """Pitch class of this key's tonic."""
        major_pc = (_MAJOR_PC_AT_POSITION_ONE + _FIFTH * (self.position - 1)) % WHEEL_POSITIONS
        if self.mode is Mode.MAJOR:
            return major_pc
        return (major_pc + _RELATIVE_MINOR_OFFSET) % WHEEL_POSITIONS

    @property
    def tonic(self) -> str:
        """Conventional spelling of the tonic, e.g. ``"G#"`` for 1A."""
        return canonical_key_name(self.pitch_class, self.mode)

    @property
    def letter(self) -> str:
        return "A" if self.mode is Mode.MINOR else "B"

    @property
    def notation(self) -> str:
        return f"{self.position}{self.letter}"

    @property
    def key_name(self) -> str:
        """Human key label, e.g. ``"G# minor"``."""
        return f"{self.tonic} {self.mode.value}"

    def __str__(self) -> str:
        return self.notation

    # -- wheel arithmetic --------------------------------------------------

    def shifted(self, steps: int) -> CamelotKey:
        """Move ``steps`` positions clockwise (negative = anticlockwise)."""
        position = (self.position - 1 + steps) % WHEEL_POSITIONS + 1
        return CamelotKey(position=position, mode=self.mode)

    def flipped(self) -> CamelotKey:
        """Same position, other mode: the relative major/minor."""
        other = Mode.MAJOR if self.mode is Mode.MINOR else Mode.MINOR
        return CamelotKey(position=self.position, mode=other)

    def distance(self, other: CamelotKey) -> int:
        """
        Shortest number of positions between two keys, ignoring mode: 0-6.

        Distance on a 12-position ring, so 12A -> 1A is 1, not 11.
        """
        raw = abs(self.position - other.position)
        return min(raw, WHEEL_POSITIONS - raw)
