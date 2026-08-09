"""
Tempo and beat tracking with Essentia's ``RhythmExtractor2013``.

``method="multifeature"`` runs several beat trackers over different onset
features and takes the consensus. It is slower than the single-feature
``degara`` alternative and it is what we use anyway, because the consensus is
where its confidence value comes from -- and a tempo number without a
confidence is not much use to a system that has to say "I am not sure".

**About that confidence.** Essentia documents it on a 0-5.32 scale, where
roughly 1.5 is low and 3.5 is high, and 0 means the trackers disagreed
completely. Dividing by 5.32 puts it in 0-1 for the schema; the division is
the only thing done to it. On that rescaled axis Essentia's own "low" is 0.28
and its "high" is 0.66, which is why the default reliability floor is 0.35
rather than something that sounds stricter.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from ...audio.decoder import DecodedAudio
from ...errors import BackendUnavailableError
from ...models import ConfidenceType, EngineInfo, TempoEstimate
from ..base import TempoAnalysis
from ..key.essentia import essentia_available, essentia_version
from .common import bpm_from_beats, interval_stats, tempo_candidates

__all__ = ["EssentiaTempoAnalyzer"]

# Documented maximum of RhythmExtractor2013's multifeature confidence.
_MAX_CONFIDENCE: Final = 5.32
_REQUIRED_SAMPLE_RATE: Final = 44100


class EssentiaTempoAnalyzer:
    def __init__(
        self,
        *,
        method: str = "multifeature",
        dj_bpm_min: float = 70.0,
        dj_bpm_max: float = 180.0,
        stability_max_cv: float = 0.04,
        min_reliability: float = 0.35,
    ) -> None:
        if not essentia_available():
            raise BackendUnavailableError(
                "Essentia is not installed; set DJTI_TEMPO_ENGINE=librosa-equivalent "
                "(`chroma` selects the portable pair) or use the Docker image."
            )
        self._method = method
        self._dj_bpm_min = dj_bpm_min
        self._dj_bpm_max = dj_bpm_max
        self._stability_max_cv = stability_max_cv
        self._min_reliability = min_reliability
        self._extractor: Any | None = None

    @property
    def name(self) -> str:
        return "essentia"

    def describe(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            algorithm="essentia.standard.RhythmExtractor2013",
            library_version=essentia_version(),
            parameters={"method": self._method, "confidence_scale_max": _MAX_CONFIDENCE},
        )

    def analyze(self, audio: DecodedAudio) -> TempoAnalysis:
        extractor = self._get_extractor()
        # RhythmExtractor2013's onset features are tuned for 44.1 kHz and the
        # algorithm does not take a sample rate parameter.
        signal = (
            audio.samples
            if audio.sample_rate == _REQUIRED_SAMPLE_RATE
            else audio.resampled(_REQUIRED_SAMPLE_RATE)
        )
        if signal.size < _REQUIRED_SAMPLE_RATE:
            return TempoAnalysis(estimate=TempoEstimate.unknown())

        reported_bpm, ticks, raw_confidence, _estimates, _intervals = extractor(signal)
        beats = [float(t) for t in np.asarray(ticks, dtype=np.float64)]

        # Prefer the grid's own tempo, falling back to the headline number
        # when there were too few beats to measure one.
        bpm = bpm_from_beats(beats) or (float(reported_bpm) if reported_bpm > 0 else None)
        if bpm is None:
            return TempoAnalysis(estimate=TempoEstimate.unknown(), beats=beats)

        _, cv = interval_stats(beats)
        confidence = float(np.clip(float(raw_confidence) / _MAX_CONFIDENCE, 0.0, 1.0))

        return TempoAnalysis(
            estimate=TempoEstimate(
                bpm=round(bpm, 2),
                confidence=round(confidence, 4),
                confidence_type=ConfidenceType.ESSENTIA_BEAT_CONFIDENCE,
                reliable=confidence >= self._min_reliability,
                stable=None if cv is None else bool(cv <= self._stability_max_cv),
                beat_interval_cv=None if cv is None else round(cv, 5),
                candidates=tempo_candidates(bpm, dj_min=self._dj_bpm_min, dj_max=self._dj_bpm_max),
                beat_count=len(beats),
            ),
            beats=[round(t, 3) for t in beats],
            downbeats=None,
        )

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            import essentia.standard as es

            self._extractor = es.RhythmExtractor2013(method=self._method)
        return self._extractor
