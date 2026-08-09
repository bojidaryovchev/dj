"""
Downbeat and meter detection: which beat is beat one?

Beat tracking finds the pulse. It says nothing about where bars start, and
that gap is why the previous version of this engine reported
``downbeats: null``. Bar phase is a *musical* question and it needs musical
evidence, so this module gathers evidence and decides.

**What it does not do.** Calling every fourth beat a downbeat is not downbeat
detection — it is a coin flip with four sides, and it will be wrong three
times out of four on any track whose first detected beat is not a bar line
(which is most of them, because trackers latch onto whatever transient starts
the intro).

**What it does.** Three beat-synchronous features, each of which tends to
peak on bar lines for different reasons:

*low-band energy* — the kick. In four-to-the-floor there is one on every
beat, but the one on the downbeat is usually the loudest, and in most other
genres the bar line is where the bass lands at all.

*onset strength* — bar lines are where new material enters, so the spectral
flux is larger there.

*harmonic change* — chords change on bar lines far more often than inside
them. This is Goto's chord-change heuristic, and it is the feature that
rescues tracks with a perfectly uniform kick pattern, where the first two
features have nothing to say.

Each feature is standardised across beats, then every (beats-per-bar, phase)
hypothesis is scored by how far above average the beats it calls downbeats
are. The winner takes it; the margin over the runner-up becomes the
confidence, so a genuinely ambiguous track reports low confidence instead of
a confident guess.

Meter is decided the same way, by comparing the best score for each candidate
beats-per-bar. Four wins on dance music, which is the point — but three is
evaluated on the same terms rather than being excluded by assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ...audio.decoder import DecodedAudio
from ...models import EngineInfo, Meter
from ..base import SupportsChromagram

__all__ = ["BeatSyncDownbeatAnalyzer", "DownbeatEstimate"]

_ANALYSIS_RATE: Final = 22050
_HOP: Final = 512
# Kick and bass fundamentals. Above ~150 Hz the snare and the low mids start
# voting, and they do not respect bar lines.
_LOW_BAND_HZ: Final = (20.0, 150.0)

# Weights over the three features. Harmonic change is weighted lowest: it is
# the most informative when it fires and the noisiest when it does not.
_WEIGHT_LOW_BAND: Final = 1.0
_WEIGHT_ONSET: Final = 0.8
_WEIGHT_HARMONIC: Final = 0.6

# Confidence saturates at this many standard errors of the margin between the
# best phase and the runner-up.
#
# The margin cannot be judged on its own, because how large a margin chance
# produces depends entirely on how many beats went into each phase average.
# Ten seconds of white noise gives six beats per phase, where the standard
# error of a mean is 0.58 -- so a "0.4 SD win" there is noise, while the same
# margin over 48 beats per phase is overwhelming evidence. Dividing by the
# standard error makes the two comparable, and is why noise reports no meter
# instead of a confident 4/4.
_Z_FOR_FULL_CONFIDENCE: Final = 3.0

_MIN_BARS: Final = 4


@dataclass(frozen=True, slots=True)
class DownbeatEstimate:
    """Where the bar lines are, and how sure we are."""

    beats_per_bar: int | None
    phase: int | None
    """Index into the beat list of the first downbeat. ``None`` if unknown."""

    confidence: float
    meter_confidence: float
    scores: dict[str, float]
    """Best score per beats-per-bar hypothesis, for inspection."""

    @property
    def known(self) -> bool:
        return self.beats_per_bar is not None and self.phase is not None


def _standardise(values: np.ndarray) -> np.ndarray:
    """Zero mean, unit variance, so features can be summed."""
    spread = float(np.std(values))
    if spread <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / spread


class BeatSyncDownbeatAnalyzer:
    """
    Downbeat and meter estimation from beat-synchronous features.

    Takes the beats as given — it decides phase, not pulse. Reuses the key
    analyser's chromagram when one is available, so the harmonic-change
    feature is usually free.
    """

    def __init__(
        self,
        *,
        candidate_meters: tuple[int, ...] = (4, 3),
        chroma_source: SupportsChromagram | None = None,
        min_confidence: float = 0.25,
    ) -> None:
        self._candidate_meters = candidate_meters
        self._chroma_source = chroma_source
        self._min_confidence = min_confidence

    @property
    def name(self) -> str:
        return "beat_sync_phase"

    def describe(self) -> EngineInfo:
        import librosa

        return EngineInfo(
            name=self.name,
            algorithm="beat-synchronous bar-phase detection (low band + onset + harmonic change)",
            library_version=getattr(librosa, "__version__", None),
            parameters={
                "candidate_meters": list(self._candidate_meters),
                "low_band_hz": list(_LOW_BAND_HZ),
                "weights": {
                    "low_band": _WEIGHT_LOW_BAND,
                    "onset": _WEIGHT_ONSET,
                    "harmonic_change": _WEIGHT_HARMONIC,
                },
                "z_for_full_confidence": _Z_FOR_FULL_CONFIDENCE,
                "min_confidence": self._min_confidence,
            },
        )

    # -- features ----------------------------------------------------------

    def _beat_features(
        self, audio: DecodedAudio, beat_times: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(low band energy, onset strength, harmonic change)`` per beat."""
        import librosa

        signal = audio.resampled(_ANALYSIS_RATE)
        frame_times_hop = _HOP / _ANALYSIS_RATE

        spectrum = np.abs(librosa.stft(y=signal, n_fft=2048, hop_length=_HOP, window="hann"))
        frequencies = librosa.fft_frequencies(sr=_ANALYSIS_RATE, n_fft=2048)
        low_band = (frequencies >= _LOW_BAND_HZ[0]) & (frequencies <= _LOW_BAND_HZ[1])
        low_energy = spectrum[low_band, :].sum(axis=0)

        onset_envelope = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(spectrum, ref=np.max), sr=_ANALYSIS_RATE, hop_length=_HOP
        )

        frame_count = spectrum.shape[1]
        beat_frames = np.clip(
            np.round(beat_times / frame_times_hop).astype(int), 0, frame_count - 1
        )

        # A beat's energy is the peak just after the onset, not the value at
        # the exact frame: a transient smears over two or three frames and the
        # beat time may sit a frame early.
        window = 3
        per_beat_low = np.array(
            [float(np.max(low_energy[frame : frame + window])) for frame in beat_frames]
        )
        per_beat_onset = np.array(
            [float(np.max(onset_envelope[frame : frame + window])) for frame in beat_frames]
        )

        return per_beat_low, per_beat_onset, self._harmonic_change(audio, beat_times)

    def _harmonic_change(self, audio: DecodedAudio, beat_times: np.ndarray) -> np.ndarray:
        """
        Cosine distance between each beat's chroma and the previous beat's.

        Chords change on bar lines. Where they do, this feature carries the
        phase on its own; where the harmony is static it contributes nothing
        rather than noise, because a flat feature standardises to zeros.
        """
        if self._chroma_source is None:
            return np.zeros(beat_times.size)

        try:
            chroma, frame_rate = self._chroma_source.chromagram(audio)
        except Exception:
            return np.zeros(beat_times.size)
        if chroma.size == 0:
            return np.zeros(beat_times.size)

        frames = chroma.shape[1]
        edges = np.clip(np.round(beat_times * frame_rate).astype(int), 0, frames)

        per_beat = np.zeros((beat_times.size, 12))
        for position in range(beat_times.size):
            start = edges[position]
            end = edges[position + 1] if position + 1 < edges.size else frames
            block = chroma[:, start:end] if end > start else chroma[:, start : start + 1]
            per_beat[position] = np.median(block, axis=1) if block.size else 0.0

        norms = np.linalg.norm(per_beat, axis=1, keepdims=True)
        normalised = per_beat / np.where(norms > 0, norms, 1.0)

        change = np.zeros(beat_times.size)
        change[1:] = 1.0 - np.sum(normalised[1:] * normalised[:-1], axis=1)
        change[0] = float(np.mean(change[1:])) if change.size > 1 else 0.0
        return change

    # -- decision ----------------------------------------------------------

    def analyze(
        self, audio: DecodedAudio, beat_times: list[float] | np.ndarray
    ) -> DownbeatEstimate:
        times = np.asarray(beat_times, dtype=np.float64)
        unknown = DownbeatEstimate(None, None, 0.0, 0.0, {})

        smallest_meter = min(self._candidate_meters)
        if times.size < smallest_meter * _MIN_BARS:
            return unknown

        try:
            low, onset, harmonic = self._beat_features(audio, times)
        except Exception:
            return unknown

        combined = (
            _WEIGHT_LOW_BAND * _standardise(low)
            + _WEIGHT_ONSET * _standardise(onset)
            + _WEIGHT_HARMONIC * _standardise(harmonic)
        )
        if not np.any(np.isfinite(combined)) or float(np.std(combined)) <= 1e-9:
            return unknown
        combined = _standardise(combined)

        best_per_meter: dict[int, tuple[float, int, float]] = {}
        for meter in self._candidate_meters:
            if times.size < meter * _MIN_BARS:
                continue
            groups = [combined[phase::meter] for phase in range(meter)]
            phase_scores = np.array([float(np.mean(group)) for group in groups])
            best_phase = int(np.argmax(phase_scores))
            ordered = np.sort(phase_scores)[::-1]
            margin = float(ordered[0] - ordered[1]) if ordered.size > 1 else 0.0

            # Standard error of a difference between two phase means, on
            # standardised (unit-variance) features: sqrt(2 / beats-per-phase).
            per_phase = min(group.size for group in groups)
            standard_error = float(np.sqrt(2.0 / max(per_phase, 1)))
            z_score = margin / standard_error if standard_error > 0 else 0.0
            best_per_meter[meter] = (float(phase_scores[best_phase]), best_phase, z_score)

        if not best_per_meter:
            return unknown

        meter = max(best_per_meter, key=lambda candidate: best_per_meter[candidate][0])
        _score, phase, z_score = best_per_meter[meter]

        confidence = float(np.clip(z_score / _Z_FOR_FULL_CONFIDENCE, 0.0, 1.0))
        meter_confidence = self._meter_confidence(best_per_meter, meter)

        if confidence < self._min_confidence:
            # A phase we do not believe is worse than no phase: it would put
            # every bar line, every phrase boundary and every 16-bar jump in
            # the wrong place while looking authoritative.
            return DownbeatEstimate(
                beats_per_bar=None,
                phase=None,
                confidence=confidence,
                meter_confidence=meter_confidence,
                scores={str(k): round(v[0], 4) for k, v in best_per_meter.items()},
            )

        return DownbeatEstimate(
            beats_per_bar=meter,
            phase=phase,
            confidence=confidence,
            meter_confidence=meter_confidence,
            scores={str(k): round(v[0], 4) for k, v in best_per_meter.items()},
        )

    @staticmethod
    def _meter_confidence(
        best_per_meter: dict[int, tuple[float, int, float]], winner: int
    ) -> float:
        """How clearly the winning meter beat the alternatives."""
        others = [score for meter, (score, _, _) in best_per_meter.items() if meter != winner]
        if not others:
            return 0.5  # nothing to compare against; do not claim certainty
        gap = best_per_meter[winner][0] - max(others)
        # These scores are already in standardised units; half a standard
        # deviation between meters is a clear win.
        return float(np.clip(gap / 0.5, 0.0, 1.0))

    def to_meter(self, estimate: DownbeatEstimate) -> Meter:
        return Meter(
            beats_per_bar=estimate.beats_per_bar,
            confidence=round(estimate.meter_confidence, 4),
            candidates=estimate.scores,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<BeatSyncDownbeatAnalyzer meters={self._candidate_meters}>"
