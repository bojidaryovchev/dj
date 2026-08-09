"""
Deterministic compatibility scoring between two tracks.

This is a **rule system, not a model**. It does not claim to predict whether
a mix will sound good; it encodes what harmonic mixing and beatmatching
conventionally consider workable, so that a library can be sorted by it. Every
rule is a number in :class:`ScoringRules`, every number is documented, and the
same two inputs always produce the same output.

Two components today:

*harmonic* -- distance on the Camelot wheel, with mode changes penalised
separately from position changes, because they fail differently: a wrong
position clashes, a wrong mode merely changes the mood.

*tempo* -- relative difference, scored against the pitch range of a CDJ
rather than an absolute BPM tolerance, because 2 BPM at 174 is a smaller
adjustment than 2 BPM at 90. Half- and double-time matches are considered and
penalised rather than ignored: 87 into 174 is a real transition.

Deliberately absent: energy, phrase structure, vocals, spectral overlap. They
are on the roadmap and each will arrive as another component with its own
weight. The version string exists so that stored scores can be recognised as
having been produced by today's rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from ..models.analysis import TempoRelation
from ..models.compatibility import CompatibilityComponents, CompatibilityScore, TrackReference
from ..music.camelot import CamelotKey
from ..music.harmony import HarmonicRelationship, classify
from ..music.notes import InvalidKeyError

__all__ = ["RULES_VERSION", "ScoringRules", "score_pair"]

RULES_VERSION: Final = "1.0.0"

# Score by distance on the wheel (0-6), for a pair in the same mode.
# 1 step is a fifth and the standard move; 2 steps is the deliberate energy
# jump; beyond that there is no convention to appeal to.
_SAME_MODE_BY_DISTANCE: Final[dict[int, float]] = {
    0: 1.00,
    1: 0.95,
    2: 0.75,
    3: 0.45,
    4: 0.30,
    5: 0.25,
    6: 0.20,
}

# The same, when one track is major and the other minor. Position 0 is the
# relative major/minor -- identical notes, so nearly as safe as the same key.
# Everything else compounds a position change with a mode change.
_CROSS_MODE_BY_DISTANCE: Final[dict[int, float]] = {
    0: 0.90,
    1: 0.60,
    2: 0.45,
    3: 0.30,
    4: 0.25,
    5: 0.20,
    6: 0.15,
}


@dataclass(frozen=True, slots=True)
class ScoringRules:
    """Every knob, in one place, so the scoring can be tuned without edits."""

    harmonic_weight: float = 0.6
    tempo_weight: float = 0.4

    tempo_max_relative_difference: float = 0.08
    """Relative tempo difference at which the tempo score reaches zero.
    0.08 is the +/-8% pitch range of a standard CDJ: past it, the tracks
    cannot be beatmatched without time-stretching them out of shape."""

    half_double_penalty: float = 0.85
    """Multiplier when the best tempo match needed a half- or double-time
    reading. Such a mix works, but it is a different move from a straight
    beatmatch and should not outrank one."""

    same_mode_scores: dict[int, float] = field(default_factory=lambda: dict(_SAME_MODE_BY_DISTANCE))
    cross_mode_scores: dict[int, float] = field(
        default_factory=lambda: dict(_CROSS_MODE_BY_DISTANCE)
    )


_RELATIONSHIP_REASONS: Final[dict[HarmonicRelationship, str]] = {
    HarmonicRelationship.SAME_KEY: "Same key",
    HarmonicRelationship.ADJACENT_PLUS: "Adjacent Camelot key, up a fifth (energy lift)",
    HarmonicRelationship.ADJACENT_MINUS: "Adjacent Camelot key, down a fifth",
    HarmonicRelationship.RELATIVE_MAJOR: "Relative major",
    HarmonicRelationship.RELATIVE_MINOR: "Relative minor",
    HarmonicRelationship.ENERGY_BOOST: "Two steps on the wheel (energy boost)",
    HarmonicRelationship.DIAGONAL: "One step plus a mode change",
    HarmonicRelationship.DISTANT: "No standard harmonic relationship",
}


def _harmonic_component(
    source: CamelotKey, target: CamelotKey, rules: ScoringRules
) -> tuple[float, HarmonicRelationship, str]:
    relationship = classify(source, target)
    distance = source.distance(target)
    table = rules.same_mode_scores if source.mode is target.mode else rules.cross_mode_scores
    score = table.get(distance, min(table.values()))
    return score, relationship, _RELATIONSHIP_REASONS[relationship]


def _tempo_component(
    bpm_a: float, bpm_b: float, rules: ScoringRules
) -> tuple[float, TempoRelation, float, str]:
    """
    Best of the straight, half-time and double-time readings of track B.

    Returns ``(score, relation, relative difference as a percentage, reason)``.
    """
    options: list[tuple[float, TempoRelation]] = [
        (bpm_b, TempoRelation.PRIMARY),
        (bpm_b * 2.0, TempoRelation.DOUBLE_TIME),
        (bpm_b / 2.0, TempoRelation.HALF_TIME),
    ]

    best_relative = float("inf")
    best_relation = TempoRelation.PRIMARY
    for candidate, relation in options:
        relative = abs(bpm_a - candidate) / ((bpm_a + candidate) / 2.0)
        if relative < best_relative:
            best_relative, best_relation = relative, relation

    score = max(0.0, 1.0 - best_relative / rules.tempo_max_relative_difference)
    if best_relation is not TempoRelation.PRIMARY:
        score *= rules.half_double_penalty

    percent = best_relative * 100.0
    if best_relation is TempoRelation.PRIMARY:
        reason = f"Tempo difference {percent:.1f}%"
    else:
        wording = "double-time" if best_relation is TempoRelation.DOUBLE_TIME else "half-time"
        reason = f"Tempo matches at {wording}, difference {percent:.1f}%"
    return score, best_relation, percent, reason


def score_pair(
    track_a: TrackReference,
    track_b: TrackReference,
    rules: ScoringRules | None = None,
) -> CompatibilityScore:
    """
    Score mixing ``track_b`` after ``track_a``.

    Components that cannot be computed are omitted rather than defaulted, and
    the remaining weights are renormalised -- a track with no detected key is
    scored on tempo alone instead of being punished for a measurement we
    failed to make.
    """
    rules = rules or ScoringRules()
    reasons: list[str] = []

    harmonic: float | None = None
    relationship: HarmonicRelationship | None = None
    if track_a.camelot and track_b.camelot:
        try:
            source = CamelotKey.parse(track_a.camelot)
            target = CamelotKey.parse(track_b.camelot)
        except InvalidKeyError as exc:
            raise ValueError(str(exc)) from exc
        harmonic, relationship, reason = _harmonic_component(source, target, rules)
        reasons.append(reason)
    else:
        reasons.append("Harmonic score skipped: key unknown for at least one track")

    tempo: float | None = None
    tempo_relation: TempoRelation | None = None
    bpm_difference_percent: float | None = None
    if track_a.bpm and track_b.bpm:
        tempo, tempo_relation, bpm_difference_percent, reason = _tempo_component(
            track_a.bpm, track_b.bpm, rules
        )
        reasons.append(reason)
    else:
        reasons.append("Tempo score skipped: BPM unknown for at least one track")

    weighted: list[tuple[float, float]] = []
    if harmonic is not None:
        weighted.append((harmonic, rules.harmonic_weight))
    if tempo is not None:
        weighted.append((tempo, rules.tempo_weight))

    total_weight = sum(weight for _, weight in weighted)
    comparable = total_weight > 0
    overall = (
        sum(value * weight for value, weight in weighted) / total_weight if comparable else 0.0
    )

    return CompatibilityScore(
        score=round(min(1.0, max(0.0, overall)), 4),
        comparable=comparable,
        components=CompatibilityComponents(
            harmonic=None if harmonic is None else round(harmonic, 4),
            tempo=None if tempo is None else round(tempo, 4),
        ),
        reasons=reasons,
        harmonic_relationship=relationship.value if relationship else None,
        tempo_relation=tempo_relation,
        bpm_difference_percent=(
            None if bpm_difference_percent is None else round(bpm_difference_percent, 3)
        ),
        rules_version=RULES_VERSION,
    )
