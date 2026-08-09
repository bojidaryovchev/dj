"""Request and response shapes that exist only at the HTTP boundary."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..models import RhythmAnalysis, TempoEstimate, TrackIdentity, WarpMap
from ..models.compatibility import TrackReference
from ..version import ANALYSIS_VERSION, SCHEMA_VERSION

__all__ = ["CompatibilityRequest", "ErrorResponse", "HealthResponse", "ReadyResponse"]


class CompatibilityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    track_a: TrackReference = Field(description="The outgoing track.")
    track_b: TrackReference = Field(description="The track being mixed in.")


class HealthResponse(BaseModel):
    status: str = "ok"
    analysis_version: str = ANALYSIS_VERSION
    schema_version: str = SCHEMA_VERSION


class ReadyResponse(BaseModel):
    ready: bool
    ffmpeg: str | None = None
    ffprobe: str | None = None
    engines: dict[str, bool] = Field(default_factory=dict)
    key_engine: str
    tempo_engine: str
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Every 4xx/5xx this service raises deliberately looks like this."""

    error: str = Field(description="Stable machine-readable code, e.g. `unsupported_format`.")
    detail: str
    request_id: str | None = None


class WarpPlanResponse(BaseModel):
    """
    What a warp would do, without doing it.

    Deliberately a subset of the full analysis: a caller deciding whether to
    spend two seconds rendering needs the grid and the recommendation, not the
    key, the loudness or the tonal segments.
    """

    track: TrackIdentity
    tempo: TempoEstimate
    rhythm: RhythmAnalysis
    warp: WarpMap | None = None
