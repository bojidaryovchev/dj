"""
The DJ interpretation endpoints.

Both are pure functions of their input -- no audio, no I/O, no state. They
are here because a library UI needs them constantly ("what mixes with 4A?",
"how well do these two go together?") and re-implementing Camelot arithmetic
in a front end is how two parts of a system start disagreeing about music
theory.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status

from ...dj.compatibility import score_pair
from ...models import CompatibleKey
from ...models.compatibility import CompatibilityScore
from ...music.camelot import CamelotKey
from ...music.harmony import compatible_keys
from ...music.notes import InvalidKeyError
from ..schemas import CompatibilityRequest

router = APIRouter(prefix="/v1/dj", tags=["dj"])


@router.get(
    "/camelot/{camelot}/compatible",
    response_model=list[CompatibleKey],
    summary="Keys that mix with a Camelot key",
)
def camelot_neighbours(
    camelot: str,
    extended: bool = Query(default=False, description="Include energy-boost and diagonal moves."),
) -> list[CompatibleKey]:
    try:
        source = CamelotKey.parse(camelot)
    except InvalidKeyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return [
        CompatibleKey(
            camelot=relation.camelot.notation,
            relationship=relation.relationship.value,
            key=relation.camelot.tonic,
            mode=relation.camelot.mode,
        )
        for relation in compatible_keys(source, extended=extended)
    ]


@router.post(
    "/compatibility",
    response_model=CompatibilityScore,
    summary="Score mixing track B after track A",
)
def compatibility(payload: CompatibilityRequest) -> CompatibilityScore:
    try:
        return score_pair(payload.track_a, payload.track_b)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
