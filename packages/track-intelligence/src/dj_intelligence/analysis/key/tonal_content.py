"""
Is there any pitched content here at all?

A backend-independent guard, and it earns its place empirically. Run against
a bare click track and against white noise, Essentia's ``KeyExtractor``
returns *C major with strength 0.76* and *F major with 0.70*. Both are
confident, both are meaningless, and neither is a bug in Essentia -- key
estimators are built to answer "which key" and are not asked "is this even
tonal". Left alone, the default configuration would report a key for a drum
loop, which is the single behaviour this system is not allowed to have.

So the question is asked separately, before any backend is consulted, and it
is asked of the *signal* rather than of the estimator: how far is the
distribution of energy across the twelve pitch classes from being flat. No
key can score its way past a flat chroma, whichever library is doing the
scoring.

The chroma backend applies the same test internally on every window as well,
where it is free -- the chromagram is already in hand. This gate is what
extends that protection to Essentia and to whatever backend comes next.
"""

from __future__ import annotations

import numpy as np

from ...audio.decoder import DecodedAudio
from ..base import SupportsChromagram
from .chroma import _tonal_salience

__all__ = ["TonalContentGate"]


class TonalContentGate:
    """
    Measures tonal salience using any chromagram-capable analyser.

    Given the active key analyser it reuses that analyser's chromagram, which
    is memoised on the audio, so for the chroma backend this costs nothing at
    all. For Essentia -- which does not expose its internal HPCP -- it means
    one extra chroma pass, measured at well under 1% of real time and a small
    fraction of what Essentia itself spends.
    """

    def __init__(self, chroma_source: SupportsChromagram, *, min_salience: float = 0.01) -> None:
        self._chroma_source = chroma_source
        self._min_salience = min_salience

    @property
    def min_salience(self) -> float:
        return self._min_salience

    def salience(self, audio: DecodedAudio) -> float:
        """0 for a perfectly flat chroma, rising as energy concentrates."""
        chroma, _ = self._chroma_source.chromagram(audio)
        if chroma.size == 0:
            return 0.0
        return _tonal_salience(np.median(chroma, axis=1))

    def has_tonal_content(self, audio: DecodedAudio) -> tuple[bool, float]:
        """``(verdict, measured salience)``, so the caller can report both."""
        measured = self.salience(audio)
        return measured >= self._min_salience, measured
