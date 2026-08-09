"""
Tempo and beat tracking with librosa.

The chain is librosa's standard one: a spectral-flux onset strength envelope,
then dynamic-programming beat tracking (Ellis, 2007) which finds the pulse
train that best trades off "beats land on onsets" against "beats are evenly
spaced".

It is run at 22.05 kHz. Onset detection cares about broadband transients, not
about anything above 11 kHz, and halving the rate halves the STFT cost for
no measurable difference in the grid.

librosa reports no confidence, so this backend derives one from how regular
the beats it found are. See ``common.consistency_confidence`` for what that
does and does not mean.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...audio.decoder import DecodedAudio
from ...models import ConfidenceType, EngineInfo, TempoEstimate
from ..base import TempoAnalysis
from .common import bpm_from_beats, consistency_confidence, interval_stats, tempo_candidates

__all__ = ["LibrosaTempoAnalyzer"]

_ANALYSIS_RATE: Final = 22050
_HOP_LENGTH: Final = 512  # ~23 ms at 22.05 kHz: fine enough to place a beat


class LibrosaTempoAnalyzer:
    def __init__(
        self,
        *,
        sample_rate: int = _ANALYSIS_RATE,
        hop_length: int = _HOP_LENGTH,
        start_bpm: float = 126.0,
        dj_bpm_min: float = 70.0,
        dj_bpm_max: float = 180.0,
        stability_max_cv: float = 0.04,
        min_reliability: float = 0.35,
    ) -> None:
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        # The tracker's prior. 126 rather than librosa's 120 because that is
        # where this tool's material actually sits; it biases which of two
        # equally good metrical readings wins, nothing more.
        self._start_bpm = start_bpm
        self._dj_bpm_min = dj_bpm_min
        self._dj_bpm_max = dj_bpm_max
        self._stability_max_cv = stability_max_cv
        self._min_reliability = min_reliability

    @property
    def name(self) -> str:
        return "librosa"

    def describe(self) -> EngineInfo:
        import librosa

        return EngineInfo(
            name=self.name,
            algorithm="librosa.beat.beat_track (dynamic programming, Ellis 2007)",
            library_version=getattr(librosa, "__version__", None),
            parameters={
                "sample_rate": self._sample_rate,
                "hop_length": self._hop_length,
                "start_bpm": self._start_bpm,
            },
        )

    def analyze(self, audio: DecodedAudio) -> TempoAnalysis:
        import librosa

        signal = audio.resampled(self._sample_rate)
        if signal.size < self._hop_length * 8:
            return TempoAnalysis(estimate=TempoEstimate.unknown())

        onset_envelope = librosa.onset.onset_strength(
            y=signal, sr=self._sample_rate, hop_length=self._hop_length
        )
        if not np.any(onset_envelope):
            # Silence, or a drone with no transients at all.
            return TempoAnalysis(estimate=TempoEstimate.unknown())

        _reported_bpm, beat_times = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=self._sample_rate,
            hop_length=self._hop_length,
            start_bpm=self._start_bpm,
            trim=False,
            units="time",
        )
        beats = [float(t) for t in np.atleast_1d(beat_times)]

        bpm = bpm_from_beats(beats)
        if bpm is None:
            return TempoAnalysis(estimate=TempoEstimate.unknown(), beats=beats)

        _, cv = interval_stats(beats)
        confidence = consistency_confidence(cv)

        return TempoAnalysis(
            estimate=TempoEstimate(
                bpm=round(bpm, 2),
                confidence=round(confidence, 4),
                confidence_type=ConfidenceType.BEAT_INTERVAL_CONSISTENCY,
                reliable=confidence >= self._min_reliability,
                stable=None if cv is None else bool(cv <= self._stability_max_cv),
                beat_interval_cv=None if cv is None else round(cv, 5),
                candidates=tempo_candidates(bpm, dj_min=self._dj_bpm_min, dj_max=self._dj_bpm_max),
                beat_count=len(beats),
            ),
            beats=[round(t, 3) for t in beats],
            # librosa tracks beats, not bars. Guessing which beat is beat one
            # from a beat tracker's output is a separate algorithm, and a
            # wrong downbeat is worse than no downbeat.
            downbeats=None,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LibrosaTempoAnalyzer sr={self._sample_rate} hop={self._hop_length}>"
