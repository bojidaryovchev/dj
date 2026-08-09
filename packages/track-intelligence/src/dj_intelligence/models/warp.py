"""
Warp models: the mapping from a source timeline to a constant-tempo one.

A warp map is *not* a beat grid. The grid says where the beats are; the warp
map says where they ought to be and which source instants to pin to which
target instants to get them there. Nothing here touches audio — rendering is
a separate stage that consumes this.

The recommendation is DJ interpretation and is marked as such: "should this
track be warped" is a judgement about tolerances and risk, not a measurement.
It lives in this module next to the map it refers to, but it is produced by
``dj/``, not by ``analysis/``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "WarpMap",
    "WarpMarker",
    "WarpMetrics",
    "WarpProvenance",
    "WarpRecommendation",
    "WarpRenderReport",
    "WarpSkipReason",
    "WarpVerification",
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class WarpMarker(_Model):
    """
    One pinned correspondence between source time and target time.

    Between two markers the mapping is linear, so a marker is where the
    stretch ratio is allowed to change.
    """

    source_time: float = Field(description="Instant in the source audio, in seconds.")
    source_beat: int = Field(description="Musical beat index at that instant.")
    target_time: float = Field(description="Where that instant lands on the constant-tempo grid.")


class WarpMetrics(_Model):
    """
    How violent the correction is.

    Timestamp error is not the only thing that matters: a map that nails every
    beat by micro-stretching each bar in a different direction will sound
    worse than a slightly looser one. These are the numbers that expose that,
    and the reason marker count is minimised rather than maximised.
    """

    marker_count: int = 0
    mean_stretch_ratio: float = 1.0
    min_stretch_ratio: float = 1.0
    max_stretch_ratio: float = 1.0
    max_correction_ms: float = Field(
        default=0.0,
        description=(
            "Largest distance any *detected* beat is from the target grid. "
            "Includes beat-detection jitter, so it overstates how wrong the track is."
        ),
    )
    systematic_error_ms: float = Field(
        default=0.0,
        description=(
            "The same measurement against the smoothed grid: the part of the error "
            "that is genuine drift rather than detector noise. This is what decides "
            "whether warping is worth doing."
        ),
    )
    mean_correction_ms: float = 0.0
    residual_grid_error_ms: float = Field(
        default=0.0,
        description="Worst error the simplified marker set still leaves behind.",
    )


class WarpSkipReason(StrEnum):
    """Why warping was not recommended."""

    ALREADY_ALIGNED = "source_grid_already_within_tolerance"
    NO_GRID = "no_reliable_beat_grid"
    UNSAFE_STRETCH = "required_stretch_exceeds_safe_threshold"
    TEMPO_UNRELIABLE = "tempo_interpretation_not_trusted"


class WarpRecommendation(_Model):
    """
    Whether to warp. **DJ interpretation, not measurement.**

    A modern electronic track usually needs nothing done to it, and stretching
    it anyway trades a perfect grid for degraded transients. The default is
    therefore to leave audio alone and say why.
    """

    required: bool = False
    reason: str | None = None
    skip_reason: WarpSkipReason | None = None
    source_grid_error_ms: float | None = Field(
        default=None,
        description="Worst beat error against the target grid with no warping at all.",
    )
    tolerance_ms: float | None = None


class WarpProvenance(_Model):
    """Enough to reproduce a warp, or to recognise one made by older rules."""

    algorithm_version: str
    target_bpm: float
    renderer: str | None = None
    renderer_version: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)


class WarpMap(_Model):
    """The full warp description for one track at one target tempo."""

    target_bpm: float
    anchor_beat: int = Field(
        default=0,
        description="Beat pinned in place; the target grid is built outwards from it.",
    )
    anchor_time: float = Field(
        default=0.0, description="Source time of the anchor beat, which does not move."
    )
    markers: list[WarpMarker] = Field(default_factory=list)
    metrics: WarpMetrics = Field(default_factory=WarpMetrics)
    recommendation: WarpRecommendation = Field(default_factory=WarpRecommendation)
    provenance: WarpProvenance | None = None
    warnings: list[str] = Field(default_factory=list)


class WarpVerification(_Model):
    """
    Did the render actually do what the map said?

    Produced by re-analysing the rendered audio and comparing its beats to the
    target grid. Without this the pipeline would be asserting its own success.
    """

    target_bpm: float
    mean_grid_error_ms: float
    p95_grid_error_ms: float
    max_grid_error_ms: float
    beats_compared: int
    threshold_ms: float
    passed: bool
    source_mean_grid_error_ms: float | None = Field(
        default=None, description="The same measurement on the input, for comparison."
    )
    improvement_factor: float | None = Field(
        default=None, description="source mean error / rendered mean error."
    )


class WarpRenderReport(_Model):
    """What a rendering run did."""

    output_path: str
    renderer: str
    renderer_version: str | None = None
    target_bpm: float
    source_duration_seconds: float
    output_duration_seconds: float
    expected_duration_seconds: float
    marker_count: int
    segment_count: int
    min_stretch_ratio: float
    max_stretch_ratio: float
    mean_stretch_ratio: float
    pitch_shift_cents: float = Field(
        default=0.0, description="Always 0: time is changed, pitch is not."
    )
    crossfade_ms: float
    render_seconds: float
    verification: WarpVerification | None = None
    warnings: list[str] = Field(default_factory=list)
