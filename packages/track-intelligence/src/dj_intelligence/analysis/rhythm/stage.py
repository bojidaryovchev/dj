"""
The rhythm stage: everything between "here are some beats" and "here is a
musical timeline".

Kept out of the pipeline so that the pipeline stays an orchestrator. This
module owns the order the rhythmic steps have to happen in, which is not
arbitrary:

1. **Refine the grid's phase.** Every later step inherits the beat times, so
   correcting their systematic lag first means the bar lines, the tempo curve
   and the warp target are all built on corrected times rather than each
   compensating separately.
2. **Assign musical indices.** Ordinal position is not musical position when
   the tracker drops a beat.
3. **Find the bar phase.** Needs the beats; supplies the bars everything else
   counts in.
4. **Fit the local tempo curve**, then classify the drift.
5. **Assemble the grid and the tempo map**, which is the artifact the DJ layer
   and the warp system actually consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...audio.decoder import DecodedAudio
from ...config import Settings
from ...models import Meter, RhythmAnalysis, TempoEstimate
from ...timeline.tempo_map import TempoMap
from ..base import SupportsChromagram
from ..tempo.common import musical_beat_indices
from .downbeats import BeatSyncDownbeatAnalyzer
from .grid import build_grid
from .refine import estimate_grid_offset
from .tempo_curve import classify_drift, local_tempo_curve

__all__ = ["RhythmResult", "RhythmStage"]


@dataclass(slots=True)
class RhythmResult:
    """Everything the rhythm stage produces, plus the timeline it built."""

    analysis: RhythmAnalysis
    tempo_map: TempoMap | None = None
    beat_times: list[float] = field(default_factory=list)
    downbeat_times: list[float] | None = None
    grid_offset_ms: float = 0.0


class RhythmStage:
    """Turns beat times into a musical timeline."""

    def __init__(
        self,
        settings: Settings,
        *,
        chroma_source: SupportsChromagram | None = None,
    ) -> None:
        self._settings = settings
        self._downbeats = BeatSyncDownbeatAnalyzer(chroma_source=chroma_source)

    @property
    def downbeat_analyzer(self) -> BeatSyncDownbeatAnalyzer:
        return self._downbeats

    def run(
        self,
        audio: DecodedAudio,
        tempo: TempoEstimate,
        beats: list[float],
    ) -> RhythmResult:
        times = np.asarray(beats, dtype=np.float64)
        if times.size < 2:
            return RhythmResult(analysis=RhythmAnalysis())

        offset_ms = 0.0
        if self._settings.beat_offset_refinement:
            offset = estimate_grid_offset(audio, times)
            # Clamp at zero: shifting the grid earlier can push a first beat
            # that sits right at the start of the file to a negative time,
            # and there is no audio there to be on.
            times = np.maximum(times + offset.seconds, 0.0)
            offset_ms = offset.milliseconds

        indices = musical_beat_indices(times)

        estimate = (
            self._downbeats.analyze(audio, times) if self._settings.downbeats_enabled else None
        )
        beats_per_bar = estimate.beats_per_bar if estimate else None
        phase = estimate.phase if estimate else None
        phase_confidence = estimate.confidence if estimate else 0.0

        # An explicit opt-in fallback, never an assumption. The measurement
        # layer leaves the meter unknown; a DJ workflow may choose to assume
        # 4/4 and take the phase from the first beat.
        if beats_per_bar is None and self._settings.fallback_beats_per_bar is not None:
            beats_per_bar = self._settings.fallback_beats_per_bar
            phase = 0
            phase_confidence = 0.0

        curve = local_tempo_curve(
            times,
            indices,
            window_beats=self._settings.tempo_curve_window_beats,
            hop_beats=self._settings.tempo_curve_hop_beats,
        )
        drift = classify_drift(curve, tempo.bpm)

        grid, observations, downbeats, tempo_map = build_grid(
            times,
            indices,
            duration=audio.duration_seconds,
            beats_per_bar=beats_per_bar,
            downbeat_beat=phase,
            beat_confidence=tempo.confidence,
            phase_confidence=phase_confidence,
        )

        meter = (
            self._downbeats.to_meter(estimate)
            if estimate is not None
            else Meter(beats_per_bar=beats_per_bar, confidence=0.0)
        )

        return RhythmResult(
            analysis=RhythmAnalysis(
                beats=observations,
                downbeats=downbeats,
                meter=meter,
                tempo_curve=curve,
                drift=drift,
                grid=grid,
            ),
            tempo_map=tempo_map,
            beat_times=[round(float(time), 4) for time in times],
            downbeat_times=[entry.time for entry in downbeats] if downbeats else None,
            grid_offset_ms=round(offset_ms, 3),
        )
