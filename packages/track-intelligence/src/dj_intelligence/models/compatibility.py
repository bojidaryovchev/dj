"""Result models for track-to-track compatibility scoring."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .analysis import TempoRelation

__all__ = ["CompatibilityComponents", "CompatibilityScore", "TrackReference"]


class TrackReference(BaseModel):
    """
    The two facts scoring needs. Deliberately not a whole analysis: a library
    stores millions of these, and a caller comparing a candidate against 5000
    tracks should not have to hydrate 5000 documents.
    """

    model_config = ConfigDict(extra="forbid")

    camelot: str | None = None
    bpm: float | None = Field(default=None, gt=0)


class CompatibilityComponents(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harmonic: float | None = Field(default=None, ge=0.0, le=1.0)
    tempo: float | None = Field(default=None, ge=0.0, le=1.0)


class CompatibilityScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    comparable: bool = Field(
        description=(
            "False when neither track supplied a usable key or tempo; `score` is then meaningless."
        )
    )
    components: CompatibilityComponents
    reasons: list[str] = Field(default_factory=list)
    harmonic_relationship: str | None = None
    tempo_relation: TempoRelation | None = Field(
        default=None, description="Which metrical reading of track B was matched against track A."
    )
    bpm_difference_percent: float | None = None
    rules_version: str
