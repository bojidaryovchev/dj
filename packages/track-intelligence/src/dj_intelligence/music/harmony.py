"""
Harmonic relationships between Camelot keys.

Still pure theory: given a key, which keys mix with it and *why*. The "why"
is the point -- a bare list of neighbours is not much use to a DJ interface,
and downstream scoring needs the relationship, not the notation, to decide
how good a transition is.

All of it is wheel arithmetic. No per-key tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .camelot import WHEEL_POSITIONS, CamelotKey
from .notes import Mode

__all__ = ["HarmonicRelation", "HarmonicRelationship", "classify", "compatible_keys"]


class HarmonicRelationship(StrEnum):
    """How two Camelot positions relate. Ordered loosely by mixability."""

    SAME_KEY = "same_key"
    """Identical key. Always mixes."""

    ADJACENT_PLUS = "adjacent_plus"
    """One step clockwise: up a perfect fifth. The classic energy lift."""

    ADJACENT_MINUS = "adjacent_minus"
    """One step anticlockwise: down a fifth. Softens."""

    RELATIVE_MAJOR = "relative_major"
    """Same position, minor -> major. Same notes, brighter mood."""

    RELATIVE_MINOR = "relative_minor"
    """Same position, major -> minor. Same notes, darker mood."""

    ENERGY_BOOST = "energy_boost"
    """Two steps clockwise: a deliberate, audible jump. Use on a drop."""

    DIAGONAL = "diagonal"
    """One step *and* a mode change. Workable, not a safe default."""

    DISTANT = "distant"
    """No standard harmonic relationship."""


@dataclass(frozen=True, slots=True)
class HarmonicRelation:
    """A neighbouring key together with the reason it is one."""

    camelot: CamelotKey
    relationship: HarmonicRelationship

    @property
    def key(self) -> str:
        return self.camelot.tonic

    @property
    def mode(self) -> Mode:
        return self.camelot.mode


def signed_steps(source: CamelotKey, target: CamelotKey) -> int:
    """
    Clockwise steps from ``source`` to ``target``, folded to -5..+6.

    Signed because direction is musically meaningful: +1 is a lift, -1 is a
    release, and a DJ interface should not conflate them.
    """
    steps = (target.position - source.position) % WHEEL_POSITIONS
    return steps - WHEEL_POSITIONS if steps > WHEEL_POSITIONS // 2 else steps


def classify(source: CamelotKey, target: CamelotKey) -> HarmonicRelationship:
    """Name the relationship between any two keys on the wheel."""
    steps = signed_steps(source, target)
    same_mode = source.mode is target.mode

    if same_mode:
        match steps:
            case 0:
                return HarmonicRelationship.SAME_KEY
            case 1:
                return HarmonicRelationship.ADJACENT_PLUS
            case -1:
                return HarmonicRelationship.ADJACENT_MINUS
            case 2:
                return HarmonicRelationship.ENERGY_BOOST
            case _:
                return HarmonicRelationship.DISTANT

    if steps == 0:
        return (
            HarmonicRelationship.RELATIVE_MAJOR
            if target.mode is Mode.MAJOR
            else HarmonicRelationship.RELATIVE_MINOR
        )
    if abs(steps) == 1:
        return HarmonicRelationship.DIAGONAL
    return HarmonicRelationship.DISTANT


def compatible_keys(source: CamelotKey, *, extended: bool = False) -> list[HarmonicRelation]:
    """
    The keys that mix with ``source``, most conventional first.

    The default four are the standard harmonic mixing rules every DJ uses:
    the same key, either neighbour on the wheel, and the relative
    major/minor. ``extended=True`` adds the moves that work but need
    intent -- the +2 energy boost and the two diagonals.
    """
    targets = [source, source.shifted(-1), source.shifted(1), source.flipped()]
    if extended:
        targets += [
            source.shifted(2),
            source.flipped().shifted(1),
            source.flipped().shifted(-1),
        ]
    return [
        HarmonicRelation(camelot=target, relationship=classify(source, target))
        for target in targets
    ]
