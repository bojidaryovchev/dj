"""
Key detection from a constant-Q chromagram. Backed by librosa.

The chain, which is the standard one and is described at greater length in
the README:

    audio -> harmonic component -> constant-Q transform -> chroma (12 bins)
          -> aggregate over time -> correlate against 24 key profiles -> key

Three decisions worth defending:

*Constant-Q rather than an FFT.* A CQT's bins are logarithmically spaced, so
one bin is one musical interval at every octave. Folding it to 12 pitch
classes is then a sum, not an interpolation.

*Aggregation over time is a median, not a mean.* A median ignores the
handful of frames where a crash cymbal or a bass drop dominates the
spectrum. Measured over 24 synthetic keys the two agree on the ``edma``
profile, but the mean drops to 50% accuracy on ``temperley`` while the median
holds at 100% -- so the median is the default and the mean is available for
comparison.

*Harmonic separation is available but off by default.* Splitting the
spectrogram into harmonic and percussive parts and keeping the harmonic one
is the textbook defence against a kick drum smearing energy across every
pitch class. Measured here it cost roughly 30x the runtime of the rest of
key detection and changed no answer: 24/24 synthetic keys either way, and the
same key with the same confidence on real mastered material. Paying 30x for a
benefit that cannot be demonstrated is not a default. Turn it on with
``DJTI_KEY_HARMONIC_SEPARATION=true`` and settle it properly against a
labelled library using ``scripts/evaluate_dataset.py``.

*Tuning estimated, not assumed.* Plenty of records are not at A=440 -- tape,
vinyl pitch, deliberate detuning, a producer who nudged the master. librosa
estimates the offset from the CQT bin distribution and the chroma is
computed against that, so a track a quarter-tone flat still lands on the
right pitch class rather than smearing across two.

This is the portable backend: it runs everywhere Python does, which is why it
is the default on Windows where Essentia has no wheels. It is also the
independent second opinion when Essentia *is* available -- see
``compare`` in the CLI.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from ...audio.decoder import DecodedAudio
from ...models import ConfidenceType, EngineInfo, KeyCandidate, KeyEstimate
from .profiles import KeyScores

__all__ = ["ChromaKeyAnalyzer"]

# C1. Below this is kick and rumble, above C8 is air; Essentia's key
# extractor uses a comparable 25 Hz - 3.5 kHz window.
_FMIN_HZ: Final = 32.703195662574829
_MAX_ALTERNATIVES: Final = 3

# Below this tonal salience, no key is claimed at all. See _tonal_salience.
#
# Measured: white noise 0.0001, a bare click track 0.0007, synthesised chord
# progressions 0.080-0.116, a real mastered track 0.097. The threshold sits
# in the empty two-orders-of-magnitude gap between "no tonal content" and
# "any tonal content" -- 15x above the loudest non-tonal fixture and 8x below
# the weakest tonal one.
_MIN_TONAL_SALIENCE: Final = 0.01


def _tonal_salience(vector: np.ndarray) -> float:
    """
    How far the chroma vector is from carrying no pitch information at all.

    ``1 - H(chroma) / log(12)``: zero when every pitch class holds equal
    energy, rising as energy concentrates on some of them.

    This exists because correlation cannot answer the question. Pearson
    correlation is invariant to scale and offset, so it measures the *shape*
    of a chroma vector while ignoring its magnitude -- and the shape of pure
    noise is still a shape. Left alone, the correlation is happy to report
    white noise as F# minor with a strength of 0.54, which is precisely the
    "it said a key because it failed" behaviour this system must not have.
    Flatness is the missing signal, and it is a property of the vector rather
    than of any template, so no key can score its way past it.
    """
    total = float(np.sum(vector))
    if total <= 0 or not np.isfinite(total):
        return 0.0
    distribution = np.clip(np.asarray(vector, dtype=np.float64) / total, 1e-12, None)
    entropy = float(-np.sum(distribution * np.log(distribution)))
    return max(0.0, 1.0 - entropy / float(np.log(12.0)))


class ChromaKeyAnalyzer:
    """
    Global key from CQT chroma correlated against key profiles.

    Also implements :class:`~dj_intelligence.analysis.base.SupportsChromagram`,
    so the segment analyser can reuse the one expensive front-end pass.
    """

    def __init__(
        self,
        *,
        profile: str = "edma",
        sample_rate: int = 22050,
        hop_length: int = 2048,
        bins_per_octave: int = 36,
        n_octaves: int = 7,
        harmonic_separation: bool = False,
        aggregate: str = "median",
        min_reliability: float = 0.35,
        min_tonal_salience: float = _MIN_TONAL_SALIENCE,
    ) -> None:
        if aggregate not in {"median", "mean"}:
            raise ValueError(f"aggregate must be 'median' or 'mean', not {aggregate!r}")
        self._profile = profile
        self._sample_rate = sample_rate
        self._hop_length = hop_length
        # 3 bins per semitone: enough resolution to survive a mistuned
        # record without paying for a full 60-bin analysis.
        self._bins_per_octave = bins_per_octave
        self._n_octaves = n_octaves
        self._harmonic_separation = harmonic_separation
        self._aggregate = aggregate
        self._min_reliability = min_reliability
        self._min_tonal_salience = min_tonal_salience
        self._scores = KeyScores(profile)

    @property
    def name(self) -> str:
        return "chroma"

    def describe(self) -> EngineInfo:
        import librosa

        return EngineInfo(
            name=self.name,
            algorithm="librosa.chroma_cqt + key-profile correlation",
            library_version=getattr(librosa, "__version__", None),
            parameters=self.parameters(),
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "profile": self._profile,
            "sample_rate": self._sample_rate,
            "hop_length": self._hop_length,
            "bins_per_octave": self._bins_per_octave,
            "n_octaves": self._n_octaves,
            "harmonic_separation": self._harmonic_separation,
            "aggregate": self._aggregate,
            "min_tonal_salience": self._min_tonal_salience,
        }

    # -- KeyAnalyzer -------------------------------------------------------

    def analyze(self, audio: DecodedAudio) -> KeyEstimate:
        chroma, _ = self.chromagram(audio)
        return self.estimate_from_chroma(chroma)

    # -- SupportsChromagram ------------------------------------------------

    def chromagram(self, audio: DecodedAudio) -> tuple[np.ndarray, float]:
        """
        Chromagram for the whole signal, memoised on the audio object.

        Returns ``(chroma, frame_rate)`` with chroma shaped (12, frames).
        """
        cache_key = f"chroma:{self._cache_fingerprint()}"
        if (cached := audio.features.get(cache_key)) is not None:
            return cached  # type: ignore[return-value]

        import librosa

        signal = audio.resampled(self._sample_rate)
        if self._harmonic_separation and signal.size >= self._hop_length * 4:
            signal = librosa.effects.harmonic(signal)

        # Estimated per track, then held fixed for every window, so segments
        # are compared against each other rather than against 23 different
        # tuning estimates.
        tuning = float(
            librosa.estimate_tuning(
                y=signal, sr=self._sample_rate, bins_per_octave=self._bins_per_octave
            )
        )

        chroma = librosa.feature.chroma_cqt(
            y=signal,
            sr=self._sample_rate,
            hop_length=self._hop_length,
            fmin=_FMIN_HZ,
            n_octaves=self._n_octaves,
            bins_per_octave=self._bins_per_octave,
            tuning=tuning,
        )
        frame_rate = self._sample_rate / self._hop_length
        result = (np.asarray(chroma, dtype=np.float64), frame_rate)
        audio.features[cache_key] = result
        return result

    def estimate_from_chroma(self, chroma: np.ndarray) -> KeyEstimate:
        """Score an aggregated chroma block against all 24 keys."""
        if chroma.size == 0:
            return KeyEstimate.unknown()

        vector = (
            np.median(chroma, axis=1) if self._aggregate == "median" else np.mean(chroma, axis=1)
        )

        # Before asking which key fits best, ask whether anything does.
        # Silence, white noise and percussion-only material all produce a
        # flat chroma that correlates respectably with some template purely
        # by accident.
        if _tonal_salience(vector) < self._min_tonal_salience:
            return KeyEstimate.unknown()

        correlations = self._scores.correlate(vector)

        best = int(np.argmax(correlations))
        best_score = float(correlations[best])
        if best_score <= 0.0:
            # Flat or noise-like chroma: no key fits better than chance.
            # Reporting the argmax here is exactly the "it said C major
            # because it failed" behaviour this system must not have.
            return KeyEstimate.unknown()

        ranked = np.argsort(correlations)[::-1]
        alternatives = [
            KeyCandidate(
                pitch_class=KeyScores.pitch_class_of(int(index)),
                mode=self._scores.mode_of(int(index)),
                score=round(float(correlations[index]), 4),
            )
            for index in ranked[1 : 1 + _MAX_ALTERNATIVES]
            if correlations[index] > 0.0
        ]

        confidence = float(np.clip(best_score, 0.0, 1.0))
        return KeyEstimate(
            pitch_class=KeyScores.pitch_class_of(best),
            mode=self._scores.mode_of(best),
            confidence=round(confidence, 4),
            confidence_type=ConfidenceType.KEY_PROFILE_CORRELATION,
            reliable=confidence >= self._min_reliability,
            alternatives=alternatives,
        )

    def _cache_fingerprint(self) -> str:
        # Profile is excluded: it changes the scoring, not the chromagram, so
        # two profiles can share one front-end pass.
        return (
            f"{self._sample_rate}:{self._hop_length}:{self._bins_per_octave}"
            f":{self._n_octaves}:{int(self._harmonic_separation)}"
        )
