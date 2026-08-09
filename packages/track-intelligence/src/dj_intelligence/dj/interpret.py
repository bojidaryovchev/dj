"""
Turning measurements into what a DJ reads.

The analysis layer produces facts about a signal: a tonic, a mode, a tempo,
a list of tonal stretches. None of those are Camelot notation, none of them
know that 87 BPM and 174 BPM are the same record, and none of them care which
key a breakdown lifts into. That interpretation happens here, once, at the
end, and it is the only place in the package that maps a measurement onto DJ
convention.

The split has a practical payoff: swapping the key backend cannot change what
"4A" means, and changing what neighbours we recommend cannot change a
measured key.
"""

from __future__ import annotations

from ..models import (
    CompatibleKey,
    DJInterpretation,
    DJSegment,
    KeyEstimate,
    TempoEstimate,
    TempoRelation,
    TonalSegment,
)
from ..music.camelot import CamelotKey
from ..music.harmony import classify, compatible_keys

__all__ = ["camelot_for", "interpret", "preferred_mix_bpm"]


def camelot_for(estimate: KeyEstimate | TonalSegment) -> CamelotKey | None:
    """Camelot position for a measurement, or ``None`` if there was no key.

    An *unreliable* key still gets a Camelot value: the caller can see
    ``reliable=false`` beside it and decide. Hiding it would throw away the
    only estimate we have.
    """
    if estimate.pitch_class is None or estimate.mode is None:
        return None
    return CamelotKey.from_key(estimate.pitch_class, estimate.mode)


def preferred_mix_bpm(
    tempo: TempoEstimate, *, dj_bpm_min: float, dj_bpm_max: float
) -> tuple[float | None, TempoRelation | None]:
    """
    The tempo a DJ would actually beatmatch to.

    The measured value wins whenever it is already in the usual range. Only
    when it is not -- a 63 BPM reading of a 126 BPM record, a 174 read as 87 --
    does a half or double reading get promoted, and the relation is returned
    so the result can say which one it chose. ``tempo.bpm`` is never
    overwritten: the measurement and the interpretation are both kept.
    """
    if tempo.bpm is None:
        return None, None
    if dj_bpm_min <= tempo.bpm <= dj_bpm_max:
        return tempo.bpm, TempoRelation.PRIMARY

    for candidate in tempo.candidates:
        if candidate.relation is not TempoRelation.PRIMARY and candidate.in_dj_range:
            return candidate.bpm, candidate.relation
    return tempo.bpm, TempoRelation.PRIMARY


def interpret(
    *,
    key: KeyEstimate,
    tempo: TempoEstimate,
    segments: list[TonalSegment],
    dj_bpm_min: float,
    dj_bpm_max: float,
    extended_neighbours: bool = False,
) -> DJInterpretation:
    """Assemble the DJ-facing view of one analysed track."""
    camelot = camelot_for(key)

    neighbours = [
        CompatibleKey(
            camelot=relation.camelot.notation,
            relationship=relation.relationship.value,
            key=relation.camelot.tonic,
            mode=relation.camelot.mode,
        )
        for relation in (compatible_keys(camelot, extended=extended_neighbours) if camelot else [])
    ]

    mix_bpm, mix_relation = preferred_mix_bpm(tempo, dj_bpm_min=dj_bpm_min, dj_bpm_max=dj_bpm_max)

    dj_segments = []
    for segment in segments:
        segment_camelot = camelot_for(segment)
        dj_segments.append(
            DJSegment(
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                camelot=segment_camelot.notation if segment_camelot else None,
                key_label=segment_camelot.key_name if segment_camelot else None,
                # The useful part: not just what key this stretch is in, but
                # how it sits against the track's overall key. "relative
                # major from 3:12" is a mix point; "same key" is not news.
                relationship_to_global=(
                    classify(camelot, segment_camelot).value
                    if camelot and segment_camelot
                    else None
                ),
                reliable=segment.reliable,
            )
        )

    return DJInterpretation(
        camelot=camelot.notation if camelot else None,
        key_label=camelot.key_name if camelot else None,
        compatible_keys=neighbours,
        mix_bpm=mix_bpm,
        mix_bpm_relation=mix_relation,
        segments=dj_segments,
    )
