"""
Structure: where the track changes, and the bar grid a DJ counts in.

Two different things, deliberately not merged.

**The phrase grid** is deterministic. Bars grouped in fours, eights, sixteens
and thirty-twos, counted from the first downbeat. It is arithmetic over the
beat grid and it is always right if the grid is right. It is what "jump 16
bars" means.

**Structural boundaries** are evidence-based. They come from measuring where
the music actually changes — timbre, harmony, energy — and they land where
they land, which is usually but not always on a phrase boundary.

Boundaries are reported as ``structural_boundary`` and nothing more. Calling
one of them "the drop" would be a semantic claim the evidence does not
support: novelty detection finds *that* something changed, not *what*. Naming
sections needs a classifier, and inventing the labels without one would put
confident nonsense into a DJ's library.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BoundaryKind",
    "PhraseGridEntry",
    "StructuralBoundary",
    "StructureAnalysis",
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BoundaryKind(StrEnum):
    """
    What kind of boundary this is.

    Only one value today, and that is the honest state of the art here.
    Section naming arrives when there is a classifier behind it.
    """

    STRUCTURAL = "structural_boundary"


class StructuralBoundary(_Model):
    """A point where the music measurably changes."""

    time: float
    bar: int | None = Field(default=None, description="Zero-based bar, if the grid reaches here.")
    beat_index: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    kind: BoundaryKind = BoundaryKind.STRUCTURAL
    snapped_to_downbeat: bool = Field(
        default=False,
        description="True when the raw peak was moved onto the nearest bar line.",
    )


class PhraseGridEntry(_Model):
    """One phrase in the deterministic bar grouping."""

    index: int = Field(description="Zero-based phrase number.")
    start_bar: int
    bars: int
    start_time: float
    end_time: float


class StructureAnalysis(_Model):
    boundaries: list[StructuralBoundary] = Field(default_factory=list)
    phrase_grid: list[PhraseGridEntry] = Field(default_factory=list)
    phrase_bars: int | None = Field(
        default=None, description="Bars per phrase used to build `phrase_grid`."
    )
