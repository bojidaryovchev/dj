"""
Key detection with Essentia's ``KeyExtractor``.

Essentia is the reference implementation for this task: a maintained C++ MIR
library whose key extractor is the same code path a lot of published
evaluation uses, and whose EDM profiles come from the group that fitted them.
When it is installed, it is the default.

``KeyExtractor`` wraps the whole chain -- framing, spectral peaks, HPCP,
tuning correction, profile matching -- behind one call, which is deliberately
what we use rather than assembling the chain ourselves. Reproducing it by
hand would mean owning parameter choices that Essentia has already tuned and
tested, and quietly drifting from the algorithm everyone else is comparing
against.

It returns three values and we keep all three: the tonic, the scale, and the
correlation strength of the winning profile. It does not expose runners-up,
so ``alternatives`` is empty for this backend -- an honest gap rather than a
fabricated ranking.

**Availability.** Essentia publishes manylinux and macOS wheels only. There
is no Windows wheel, so on Windows this backend cannot be installed and the
chroma backend takes over. The Docker image installs it.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ...audio.decoder import DecodedAudio
from ...errors import BackendUnavailableError
from ...models import ConfidenceType, EngineInfo, KeyEstimate
from ...music.notes import InvalidKeyError, parse_mode, parse_pitch_class

__all__ = ["EssentiaKeyAnalyzer", "essentia_available", "essentia_version"]


def essentia_available() -> bool:
    """True if the Essentia extension module can be imported here."""
    try:
        return importlib.util.find_spec("essentia.standard") is not None
    except (ImportError, ValueError):
        return False


def essentia_version() -> str | None:
    try:
        import essentia

        return str(essentia.__version__)
    except Exception:
        return None


class EssentiaKeyAnalyzer:
    def __init__(
        self,
        *,
        profile: str = "edma",
        sample_rate: int = 44100,
        min_reliability: float = 0.35,
    ) -> None:
        if not essentia_available():
            raise BackendUnavailableError(
                "Essentia is not installed. It has no Windows wheel; install it on "
                "Linux/macOS with `pip install 'dj-track-intelligence[essentia]'`, "
                "or use the Docker image. Set DJTI_KEY_ENGINE=chroma to select the "
                "portable backend explicitly."
            )
        self._profile = profile
        self._sample_rate = sample_rate
        self._min_reliability = min_reliability
        self._extractor: Any | None = None

    @property
    def name(self) -> str:
        return "essentia"

    def describe(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            algorithm="essentia.standard.KeyExtractor",
            library_version=essentia_version(),
            parameters={"profileType": self._profile, "sampleRate": self._sample_rate},
        )

    def analyze(self, audio: DecodedAudio) -> KeyEstimate:
        extractor = self._get_extractor()
        # Essentia's Python bindings are strict: float32, C-contiguous, 1-D.
        samples = audio.samples
        if audio.sample_rate != self._sample_rate:
            samples = audio.resampled(self._sample_rate)

        key, scale, strength = extractor(samples)

        try:
            pitch_class = parse_pitch_class(key)
            mode = parse_mode(scale)
        except InvalidKeyError:
            # Essentia can return an empty key for material with no tonal
            # content at all. That is a real answer: "no key".
            return KeyEstimate.unknown(confidence=max(0.0, min(1.0, float(strength))))

        confidence = max(0.0, min(1.0, float(strength)))
        return KeyEstimate(
            pitch_class=pitch_class,
            mode=mode,
            confidence=round(confidence, 4),
            confidence_type=ConfidenceType.ESSENTIA_KEY_STRENGTH,
            reliable=confidence >= self._min_reliability,
        )

    def _get_extractor(self) -> Any:
        if self._extractor is None:
            import essentia.standard as es

            self._extractor = es.KeyExtractor(
                profileType=self._profile,
                sampleRate=self._sample_rate,
            )
        return self._extractor
