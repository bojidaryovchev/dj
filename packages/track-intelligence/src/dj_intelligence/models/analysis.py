"""
The canonical result document.

Layering, which is the whole point of this module:

* ``tempo`` / ``tonality`` / ``loudness`` / ``tonal_segments`` are
  **measurements**. They say what is in the audio and how sure we are.
* ``dj`` is **interpretation**. Camelot notation, harmonic neighbours and the
  BPM a DJ would actually beatmatch to are conventions layered on top of the
  measurements, not properties of the signal.

Keeping the two apart in the schema mirrors keeping them apart in the code
(``analysis/`` never imports ``dj/``), so a second opinion on the musical
layer can be swapped in without touching anything a DJ interface reads.

Nothing here invents a value. A measurement that could not be made is
``None`` with ``reliable=false`` and a warning explaining why.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..music.notes import Mode, canonical_key_name
from ..version import ANALYSIS_VERSION, SCHEMA_VERSION
from .rhythm import RhythmAnalysis
from .structure import StructureAnalysis
from .warp import WarpMap

__all__ = [
    "AnalysisMetadata",
    "AnalysisWarning",
    "AudioProperties",
    "CompatibleKey",
    "ConfidenceType",
    "DJInterpretation",
    "DJSegment",
    "EngineInfo",
    "KeyCandidate",
    "KeyEstimate",
    "LoudnessMeasurement",
    "StageTiming",
    "TempoCandidate",
    "TempoEstimate",
    "TempoRelation",
    "TonalSegment",
    "TrackAnalysis",
    "TrackIdentity",
    "WarningCode",
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------
# confidence
# --------------------------------------------------------------------------


class ConfidenceType(StrEnum):
    """
    What a ``confidence`` number actually is.

    None of these are probabilities, and the field exists so that no consumer
    can mistake them for one. A 0.8 correlation against a key profile and a
    0.8 normalised Essentia beat confidence are not the same quantity and
    should not be averaged, compared or thresholded as if they were. The
    README documents each.
    """

    KEY_PROFILE_CORRELATION = "key_profile_correlation"
    """Pearson correlation of the chroma vector with the winning key profile,
    rescaled from [-1, 1] to [0, 1]. Higher means the track's pitch content
    fits that key better than the alternatives."""

    ESSENTIA_KEY_STRENGTH = "essentia_key_strength"
    """Essentia ``KeyExtractor``'s ``strength`` output, unmodified. Also a
    profile correlation, but computed over Essentia's own HPCP."""

    ESSENTIA_BEAT_CONFIDENCE = "essentia_beat_confidence"
    """Essentia ``RhythmExtractor2013`` multifeature confidence, divided by
    its documented maximum of 5.32 to land in [0, 1]. Essentia's own scale
    calls ~1.5 low and ~3.5 high, i.e. 0.28 and 0.66 here."""

    BEAT_INTERVAL_CONSISTENCY = "beat_interval_consistency"
    """Derived, not reported by the library: how regular the detected
    inter-beat intervals are. librosa's beat tracker returns no confidence,
    and a steady grid is the best available proxy for having found one."""

    NONE = "none"
    """No estimate was produced; ``confidence`` is 0."""


class WarningCode(StrEnum):
    """Machine-readable reasons a result is less than it could be."""

    SILENT_AUDIO = "silent_audio"
    NO_TONAL_CONTENT = "no_tonal_content"
    SHORT_AUDIO = "short_audio"
    LOW_KEY_CONFIDENCE = "low_key_confidence"
    LOW_TEMPO_CONFIDENCE = "low_tempo_confidence"
    KEY_ANALYSIS_FAILED = "key_analysis_failed"
    TEMPO_ANALYSIS_FAILED = "tempo_analysis_failed"
    TEMPO_UNSTABLE = "tempo_unstable"
    TEMPO_OUT_OF_DJ_RANGE = "tempo_out_of_dj_range"
    SEGMENTATION_SKIPPED = "segmentation_skipped"
    SEGMENTATION_FAILED = "segmentation_failed"
    LOUDNESS_UNAVAILABLE = "loudness_unavailable"
    DOWNBEATS_UNAVAILABLE = "downbeats_unavailable"
    RHYTHM_ANALYSIS_FAILED = "rhythm_analysis_failed"
    STRUCTURE_ANALYSIS_FAILED = "structure_analysis_failed"
    WARP_LARGE_STRETCH = "warp_requires_large_local_stretch"
    ENGINE_FALLBACK = "engine_fallback"
    ANALYSIS_TRUNCATED = "analysis_truncated"


class AnalysisWarning(_Model):
    code: WarningCode
    message: str
    stage: str | None = None


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


class TrackIdentity(_Model):
    filename: str
    sha256: str = Field(
        description="Content hash of the source file, for de-duplication and caching."
    )
    size_bytes: int


class AudioProperties(_Model):
    duration_seconds: float
    analysis_sample_rate: int = Field(
        description="Rate the signal was normalised to before analysis."
    )
    source_sample_rate: int | None = None
    source_channels: int | None = None
    codec: str | None = None
    container: str | None = None
    bit_rate_bps: int | None = None
    analysed_seconds: float = Field(
        description="How much of the file was analysed; less than duration if truncated."
    )


# --------------------------------------------------------------------------
# tonality (measurement)
# --------------------------------------------------------------------------


class KeyCandidate(_Model):
    """A runner-up. Kept because the gap to second place is often more
    informative than the winner's absolute score -- a track that scores 0.62
    for F minor and 0.61 for Ab major is genuinely ambiguous, and a single
    number cannot say so."""

    pitch_class: int = Field(ge=0, le=11)
    mode: Mode
    key: str = ""
    score: float

    @model_validator(mode="after")
    def _spell(self) -> Self:
        if not self.key:
            object.__setattr__(self, "key", canonical_key_name(self.pitch_class, self.mode))
        return self


class KeyEstimate(_Model):
    pitch_class: int | None = Field(default=None, ge=0, le=11)
    mode: Mode | None = None
    key: str | None = Field(
        default=None, description='Conventional spelling of the tonic, e.g. "Ab".'
    )
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_type: ConfidenceType
    reliable: bool
    alternatives: list[KeyCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def _spell(self) -> Self:
        if self.key is None and self.pitch_class is not None and self.mode is not None:
            object.__setattr__(self, "key", canonical_key_name(self.pitch_class, self.mode))
        return self

    @classmethod
    def unknown(cls, confidence: float = 0.0) -> KeyEstimate:
        return cls(
            confidence=confidence,
            confidence_type=ConfidenceType.NONE,
            reliable=False,
        )


# --------------------------------------------------------------------------
# tempo (measurement)
# --------------------------------------------------------------------------


class TempoRelation(StrEnum):
    PRIMARY = "primary"
    HALF_TIME = "half_time"
    DOUBLE_TIME = "double_time"
    THIRD = "one_third"
    TRIPLE = "triple"


class TempoCandidate(_Model):
    """
    A metrically equivalent reading of the same beat grid.

    Beat trackers cannot distinguish 70 from 140 BPM from the signal alone --
    both describe the same pulse train, and which one a listener hears is a
    genre convention. Rather than silently doubling, the reading the algorithm
    returned stays in ``TempoEstimate.bpm`` and the alternatives are listed
    here for the DJ layer to choose from.
    """

    bpm: float
    relation: TempoRelation
    in_dj_range: bool


class TempoEstimate(_Model):
    bpm: float | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_type: ConfidenceType
    reliable: bool
    stable: bool | None = Field(
        default=None,
        description="False when the beat grid drifts: live edits, rubato, tempo ramps.",
    )
    beat_interval_cv: float | None = Field(
        default=None,
        description="Coefficient of variation of inter-beat intervals; the basis for `stable`.",
    )
    candidates: list[TempoCandidate] = Field(default_factory=list)
    beat_count: int = 0

    @classmethod
    def unknown(cls, confidence: float = 0.0) -> TempoEstimate:
        return cls(
            confidence=confidence,
            confidence_type=ConfidenceType.NONE,
            reliable=False,
        )


# --------------------------------------------------------------------------
# structure and level (measurement)
# --------------------------------------------------------------------------


class TonalSegment(_Model):
    """A stretch of the track that reads as one key."""

    start_seconds: float
    end_seconds: float
    pitch_class: int | None = Field(default=None, ge=0, le=11)
    mode: Mode | None = None
    key: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reliable: bool

    @model_validator(mode="after")
    def _spell(self) -> Self:
        if self.key is None and self.pitch_class is not None and self.mode is not None:
            object.__setattr__(self, "key", canonical_key_name(self.pitch_class, self.mode))
        return self

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


class LoudnessMeasurement(_Model):
    """EBU R128 / ITU-R BS.1770 measurements, plus plain sample statistics."""

    integrated_lufs: float | None = None
    loudness_range_lu: float | None = None
    true_peak_dbtp: float | None = None
    sample_peak_dbfs: float | None = None
    rms_dbfs: float | None = None


# --------------------------------------------------------------------------
# DJ interpretation
# --------------------------------------------------------------------------


class CompatibleKey(_Model):
    camelot: str
    relationship: str
    key: str
    mode: Mode


class DJSegment(_Model):
    """A tonal segment expressed the way a DJ reads it, including how it
    relates to the track's overall key -- which is the part that matters when
    deciding where in the track to mix."""

    start_seconds: float
    end_seconds: float
    camelot: str | None = None
    key_label: str | None = None
    relationship_to_global: str | None = None
    reliable: bool


class DJInterpretation(_Model):
    camelot: str | None = None
    key_label: str | None = Field(default=None, description='e.g. "F minor".')
    compatible_keys: list[CompatibleKey] = Field(default_factory=list)
    mix_bpm: float | None = Field(
        default=None,
        description=(
            "The measured tempo folded into the usual DJ range. Equals "
            "`tempo.bpm` unless a half- or double-time reading was preferred."
        ),
    )
    mix_bpm_relation: TempoRelation | None = None
    segments: list[DJSegment] = Field(default_factory=list)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


class StageTiming(_Model):
    stage: str
    duration_ms: float


class EngineInfo(_Model):
    name: str = Field(description='Backend identifier, e.g. "essentia" or "chroma".')
    algorithm: str = Field(description='Specific algorithm, e.g. "essentia.KeyExtractor".')
    library_version: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class AnalysisMetadata(_Model):
    """
    Everything needed to decide whether a stored result is still current.

    ``analysis_version`` plus ``configuration_fingerprint`` answer "would
    re-running this today give the same answer?"; the engine blocks answer
    "which implementation produced it?".
    """

    analysis_version: str = ANALYSIS_VERSION
    schema_version: str = SCHEMA_VERSION
    package_version: str
    key_engine: EngineInfo | None = None
    tempo_engine: EngineInfo | None = None
    configuration_fingerprint: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float
    realtime_ratio: float | None = Field(
        default=None,
        description="processing time / audio duration. 0.01 means 100x faster than playback.",
    )
    stages: list[StageTiming] = Field(default_factory=list)


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------


class TrackAnalysis(_Model):
    schema_version: str = SCHEMA_VERSION
    track: TrackIdentity
    audio: AudioProperties
    tempo: TempoEstimate
    tonality: KeyEstimate
    loudness: LoudnessMeasurement = Field(default_factory=LoudnessMeasurement)
    tonal_segments: list[TonalSegment] = Field(default_factory=list)
    beats: list[float] = Field(
        default_factory=list,
        description=(
            "Beat positions in seconds. A flat convenience view of `rhythm.beats`, "
            "kept because it predates the rhythm block and callers depend on it."
        ),
    )
    downbeats: list[float] | None = Field(
        default=None,
        description=(
            "Bar-line positions in seconds, or null when no downbeat phase could be "
            "established. A flat view of `rhythm.downbeats`."
        ),
    )
    rhythm: RhythmAnalysis = Field(
        default_factory=RhythmAnalysis,
        description="The rhythmic timeline: indexed beats, bars, meter, local tempo, grid.",
    )
    structure: StructureAnalysis = Field(
        default_factory=StructureAnalysis,
        description="Where the track changes, and the phrase grid a DJ counts in.",
    )
    warp: WarpMap | None = Field(
        default=None,
        description=(
            "Present only for the `warp` analysis profile. Describes a "
            "correction; applying it is a separate, explicit step."
        ),
    )
    dj: DJInterpretation = Field(default_factory=DJInterpretation)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    analysis: AnalysisMetadata
