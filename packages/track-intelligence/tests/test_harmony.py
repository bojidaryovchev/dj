"""Harmonic relationships and the neighbour list."""

from __future__ import annotations

import pytest

from dj_intelligence.music.camelot import CamelotKey
from dj_intelligence.music.harmony import (
    HarmonicRelationship,
    classify,
    compatible_keys,
    signed_steps,
)
from dj_intelligence.music.notes import Mode


def test_the_specified_neighbours_of_4a() -> None:
    """The exact list, in the exact order, from the specification."""
    relations = compatible_keys(CamelotKey.parse("4A"))
    assert [(r.camelot.notation, r.relationship.value) for r in relations] == [
        ("4A", "same_key"),
        ("3A", "adjacent_minus"),
        ("5A", "adjacent_plus"),
        ("4B", "relative_major"),
    ]


def test_every_key_has_four_distinct_neighbours() -> None:
    for position in range(1, 13):
        for mode in Mode:
            key = CamelotKey(position=position, mode=mode)
            relations = compatible_keys(key)
            assert len(relations) == 4
            assert len({r.camelot for r in relations}) == 4
            assert {r.relationship for r in relations} == {
                HarmonicRelationship.SAME_KEY,
                HarmonicRelationship.ADJACENT_MINUS,
                HarmonicRelationship.ADJACENT_PLUS,
                (
                    HarmonicRelationship.RELATIVE_MAJOR
                    if mode is Mode.MINOR
                    else HarmonicRelationship.RELATIVE_MINOR
                ),
            }


def test_neighbours_wrap_at_the_seam() -> None:
    assert [r.camelot.notation for r in compatible_keys(CamelotKey.parse("1A"))] == [
        "1A",
        "12A",
        "2A",
        "1B",
    ]
    assert [r.camelot.notation for r in compatible_keys(CamelotKey.parse("12B"))] == [
        "12B",
        "11B",
        "1B",
        "12A",
    ]


def test_compatibility_is_symmetric_in_membership() -> None:
    """If B mixes with A, A mixes with B. Wheel moves are all invertible."""
    for position in range(1, 13):
        for mode in Mode:
            key = CamelotKey(position=position, mode=mode)
            for relation in compatible_keys(key):
                back = {r.camelot for r in compatible_keys(relation.camelot)}
                assert key in back


def test_extended_adds_the_deliberate_moves() -> None:
    relations = compatible_keys(CamelotKey.parse("4A"), extended=True)
    by_notation = {r.camelot.notation: r.relationship for r in relations}
    assert by_notation["6A"] is HarmonicRelationship.ENERGY_BOOST
    assert by_notation["5B"] is HarmonicRelationship.DIAGONAL
    assert by_notation["3B"] is HarmonicRelationship.DIAGONAL
    assert len(relations) == 7


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("4A", "4A", HarmonicRelationship.SAME_KEY),
        ("4A", "5A", HarmonicRelationship.ADJACENT_PLUS),
        ("4A", "3A", HarmonicRelationship.ADJACENT_MINUS),
        ("4A", "4B", HarmonicRelationship.RELATIVE_MAJOR),
        ("4B", "4A", HarmonicRelationship.RELATIVE_MINOR),
        ("4A", "6A", HarmonicRelationship.ENERGY_BOOST),
        ("4A", "5B", HarmonicRelationship.DIAGONAL),
        ("4A", "10A", HarmonicRelationship.DISTANT),
        ("12A", "1A", HarmonicRelationship.ADJACENT_PLUS),
        ("1A", "12A", HarmonicRelationship.ADJACENT_MINUS),
    ],
)
def test_classify(source: str, target: str, expected: HarmonicRelationship) -> None:
    assert classify(CamelotKey.parse(source), CamelotKey.parse(target)) is expected


def test_signed_steps_keeps_direction() -> None:
    assert signed_steps(CamelotKey.parse("4A"), CamelotKey.parse("5A")) == 1
    assert signed_steps(CamelotKey.parse("5A"), CamelotKey.parse("4A")) == -1
    assert signed_steps(CamelotKey.parse("12A"), CamelotKey.parse("1A")) == 1
    assert signed_steps(CamelotKey.parse("1A"), CamelotKey.parse("12A")) == -1
    for position in range(1, 13):
        steps = signed_steps(CamelotKey.parse("1A"), CamelotKey(position=position, mode=Mode.MINOR))
        assert -5 <= steps <= 6
