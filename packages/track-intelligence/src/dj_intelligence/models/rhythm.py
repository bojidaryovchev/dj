"""
The rhythmic timeline: beats, bars, meter, local tempo, and the grid.

These are all **measurements**. They describe where the pulses are and how the
tempo behaves; they say nothing about whether a DJ should do anything about
it. Warp recommendations and navigation live elsewhere.

Six concepts are kept apart on purpose, because collapsing them is how a beat
grid ends up quietly wrong:

``BeatObservation``
    Where a beat is in the *source* audio, and which musical beat it is.
    Detection and indexing are different questions and both are recorded.

``Downbeat``
    Which beats are beat one of a bar. Ordinary beat tracking does not
    establish this -- it needs a phase decision, made from musical evidence.

``Meter``
    How many beats are in a bar. Not assumed to be four.

``TempoCurvePoint``
    Local tempo through the track, so a drifting record is described rather
    than averaged away.

``TempoDrift``
    Whether the track can be treated as constant tempo, with the metrics
    behind that verdict exposed rather than hidden behind a label.

``BeatGrid``
    The assembled musical timeline, plus per-region confidence -- a beatless
    intro should not claim the same certainty as the groove after it.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BeatGrid",
    "BeatObservation",
    "Downbeat",
    "DriftClassification",
    "GridRegion",
    "Meter",
    "RhythmAnalysis",
    "TempoCurvePoint",
    "TempoDrift",
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BeatObservation(_Model):
    """
    One detected beat, with its position in musical time.

    Indexing conventions, fixed and relied on everywhere:

    * ``index`` is **zero-based** and counts detected beats from the start of
      the track.
    * ``bar`` is **zero-based** and counts bars from the first *complete* bar.
      Beats before the first downbeat get a negative bar index, because they
      genuinely are before bar zero and pretending otherwise would put a
      pickup on the downbeat.
    * ``beat_in_bar`` is **one-based**, because musicians count "one, two,
      three, four" and every DJ display does the same.

    ``bar`` and ``beat_in_bar`` are ``None`` when no downbeat phase could be
    established.
    """

    index: int
    time: float = Field(description="Position in the source audio, in seconds.")
    bar: int | None = None
    beat_in_bar: int | None = Field(default=None, ge=1)
    confidence: float = Field(ge=0.0, le=1.0)


class Downbeat(_Model):
    """Beat one of a bar."""

    bar: int = Field(description="Zero-based bar index.")
    beat_index: int = Field(description="Index of this beat in the beat list.")
    time: float
    confidence: float = Field(ge=0.0, le=1.0)


class Meter(_Model):
    """
    Beats per bar.

    ``None`` when the evidence does not support a choice. The DJ layer may
    assume 4/4 under an explicit fallback; the measurement layer does not.
    """

    beats_per_bar: int | None = Field(default=None, ge=2, le=16)
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: dict[str, float] = Field(
        default_factory=dict,
        description="Score per hypothesis, keyed by beats-per-bar, so a close call is visible.",
    )


class TempoCurvePoint(_Model):
    """Local tempo over one window of the track."""

    start_time: float
    end_time: float
    bpm: float
    beat_count: int = Field(description="Beats the estimate was fitted over.")


class DriftClassification(StrEnum):
    """
    How much the tempo moves, on a scale with documented cut-offs.

    Thresholds are on the *relative range* of local tempo, (max - min) /
    nominal, and they sit above the measurement noise floor: on a
    known-constant fixture the local estimates spread by well under 0.1%, so
    ``STABLE`` is set at 0.2% to avoid reporting drift that is really
    estimator noise.
    """

    STABLE = "stable"
    """< 0.2% -- a sequenced track. Warping would only add error."""

    MINOR_DRIFT = "minor_drift"
    """< 1.0% -- a tight live take, or a slightly loose master."""

    VARIABLE_TEMPO = "variable_tempo"
    """< 3.0% -- a real human performance, or a tempo automation."""

    HIGHLY_VARIABLE = "highly_variable"
    """>= 3.0% -- a live recording, a tempo transition, or bad tracking."""

    UNKNOWN = "unknown"
    """Too few beats to say."""


class TempoDrift(_Model):
    """The drift verdict, and every number behind it."""

    nominal_bpm: float | None = None
    local_bpm_min: float | None = None
    local_bpm_max: float | None = None
    max_absolute_bpm_delta: float | None = Field(
        default=None, description="Largest deviation of a local estimate from the nominal BPM."
    )
    relative_percent: float | None = Field(
        default=None, description="(max local - min local) / nominal, as a percentage."
    )
    classification: DriftClassification = DriftClassification.UNKNOWN
    tempo_stable: bool | None = None


class GridRegion(_Model):
    """
    A stretch of the track with its own grid confidence.

    A long ambient intro has no beats to be right or wrong about, and giving
    it the same confidence as the groove would be a claim we cannot support.
    """

    start: float
    end: float
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str | None = Field(
        default=None,
        description='Why the confidence is what it is, e.g. "beatless_intro".',
    )


class BeatGrid(_Model):
    """
    The musical timeline: which source instant is which musical beat.

    Everything downstream — the tempo map, warping, navigation — is built from
    this, so it carries its own quality assessment rather than leaving callers
    to infer one.
    """

    beats_per_bar: int | None = None
    first_downbeat_time: float | None = None
    first_downbeat_beat_index: int | None = Field(
        default=None,
        description="Index of the beat that starts bar 0. Also the phase of the bar grid.",
    )
    bar_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    regions: list[GridRegion] = Field(default_factory=list)


class RhythmAnalysis(_Model):
    """Everything measured about the rhythmic timeline."""

    beats: list[BeatObservation] = Field(default_factory=list)
    downbeats: list[Downbeat] = Field(default_factory=list)
    meter: Meter = Field(default_factory=lambda: Meter(confidence=0.0))
    tempo_curve: list[TempoCurvePoint] = Field(default_factory=list)
    drift: TempoDrift = Field(default_factory=TempoDrift)
    grid: BeatGrid = Field(default_factory=BeatGrid)
