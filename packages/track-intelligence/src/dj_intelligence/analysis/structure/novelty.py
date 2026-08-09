"""
Finding where the music changes.

The method is Foote's: describe every beat with a feature vector, compare
every beat with every other to get a self-similarity matrix, then slide a
checkerboard kernel down its diagonal. Where the kernel sits inside a
homogeneous section it sees one big similar block and scores low; where it
straddles a boundary it sees two similar blocks and two dissimilar ones, and
scores high. The peaks are the boundaries.

Two feature families, because sections differ in more than one way: MFCCs
catch timbre (the pads drop out, the drums come in) and chroma catches
harmony (the breakdown moves to the relative major). Either alone misses
whole classes of boundary.

**What this does not do is name the sections.** Novelty detection finds
*that* something changed, never *what*. Calling a peak "the drop" would need a
classifier trained on labelled music, and inventing the label without one puts
confident nonsense into a DJ's library — so every boundary is reported as
``structural_boundary`` and nothing more.

Boundaries are snapped to the nearest bar line when a bar grid exists, because
in this repertoire sections change on bar lines and the kernel's resolution is
coarser than a beat.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...audio.decoder import DecodedAudio
from ...models import BoundaryKind, EngineInfo, StructuralBoundary
from ...observability import get_logger
from ...timeline.tempo_map import TempoMap
from ..base import SupportsChromagram

__all__ = ["NoveltyStructureAnalyzer"]

log = get_logger(__name__)

_ANALYSIS_RATE: Final = 22050
_HOP: Final = 512
_MFCC_COUNT: Final = 13
# Kernel half-width in beats. 16 beats each side is four bars of 4/4: long
# enough to average over a phrase, short enough to resolve an 8-bar section.
_KERNEL_HALF_BEATS: Final = 16
_MIN_BEATS: Final = _KERNEL_HALF_BEATS * 3
# Boundaries closer than this are the same event seen twice.
_MIN_SPACING_BARS: Final = 4


def _checkerboard(half: int) -> np.ndarray:
    """Gaussian-tapered checkerboard. The taper stops distant, unrelated
    material from dominating the score at the kernel's corners."""
    size = 2 * half
    axis = np.arange(size) - half + 0.5
    grid_x, grid_y = np.meshgrid(axis, axis)
    sign = np.sign(grid_x) * np.sign(grid_y)
    taper = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * (half / 2.0) ** 2))
    return np.asarray(sign * taper, dtype=np.float64)


class NoveltyStructureAnalyzer:
    """Beat-synchronous self-similarity novelty over timbre and harmony."""

    def __init__(
        self,
        *,
        chroma_source: SupportsChromagram | None = None,
        min_spacing_bars: int = _MIN_SPACING_BARS,
        peak_threshold: float = 0.25,
    ) -> None:
        self._chroma_source = chroma_source
        self._min_spacing_bars = min_spacing_bars
        self._peak_threshold = peak_threshold

    @property
    def name(self) -> str:
        return "novelty"

    def describe(self) -> EngineInfo:
        import librosa

        return EngineInfo(
            name=self.name,
            algorithm="beat-synchronous self-similarity novelty (Foote checkerboard)",
            library_version=getattr(librosa, "__version__", None),
            parameters={
                "kernel_half_beats": _KERNEL_HALF_BEATS,
                "mfcc_count": _MFCC_COUNT,
                "min_spacing_bars": self._min_spacing_bars,
                "peak_threshold": self._peak_threshold,
            },
        )

    # -- features ----------------------------------------------------------

    def _beat_features(self, audio: DecodedAudio, beat_times: np.ndarray) -> np.ndarray:
        """One L2-normalised vector per beat: timbre stacked on harmony."""
        import librosa

        signal = audio.resampled(_ANALYSIS_RATE)
        mfcc = librosa.feature.mfcc(
            y=signal, sr=_ANALYSIS_RATE, n_mfcc=_MFCC_COUNT, hop_length=_HOP
        )
        # The first coefficient is loudness; a level change is not a section
        # change, and leaving it in makes every fade read as a boundary.
        mfcc = mfcc[1:]

        frames = librosa.time_to_frames(beat_times, sr=_ANALYSIS_RATE, hop_length=_HOP)
        frames = np.clip(frames, 0, mfcc.shape[1] - 1)
        timbre = librosa.util.sync(mfcc, frames, aggregate=np.mean)

        blocks = [timbre]
        if self._chroma_source is not None:
            try:
                chroma, frame_rate = self._chroma_source.chromagram(audio)
                chroma_frames = np.clip(
                    np.round(beat_times * frame_rate).astype(int), 0, chroma.shape[1] - 1
                )
                blocks.append(librosa.util.sync(chroma, chroma_frames, aggregate=np.mean))
            except Exception as exc:
                log.debug("structure.chroma_unavailable", extra={"error": str(exc)})

        width = min(block.shape[1] for block in blocks)
        stacked = np.vstack([block[:, :width] for block in blocks])

        # Standardise each dimension so MFCC magnitudes do not swamp chroma.
        centred = stacked - stacked.mean(axis=1, keepdims=True)
        scale = centred.std(axis=1, keepdims=True)
        centred /= np.where(scale > 0, scale, 1.0)

        norms = np.linalg.norm(centred, axis=0, keepdims=True)
        return np.asarray(centred / np.where(norms > 0, norms, 1.0), dtype=np.float64)

    # -- detection ---------------------------------------------------------

    def analyze(
        self,
        audio: DecodedAudio,
        beat_times: np.ndarray,
        tempo_map: TempoMap | None = None,
    ) -> list[StructuralBoundary]:
        times = np.asarray(beat_times, dtype=np.float64)
        if times.size < _MIN_BEATS:
            return []

        try:
            features = self._beat_features(audio, times)
        except Exception:
            return []
        if features.shape[1] < _MIN_BEATS:
            return []

        similarity = features.T @ features
        novelty = self._novelty(similarity)
        if novelty.size == 0 or float(np.max(novelty)) <= 0:
            return []
        novelty = novelty / float(np.max(novelty))

        peaks = self._pick_peaks(novelty, tempo_map, times)
        return self._to_boundaries(peaks, novelty, times, tempo_map)

    @staticmethod
    def _novelty(similarity: np.ndarray) -> np.ndarray:
        half = _KERNEL_HALF_BEATS
        kernel = _checkerboard(half)
        count = similarity.shape[0]
        curve = np.zeros(count)
        for centre in range(half, count - half):
            block = similarity[centre - half : centre + half, centre - half : centre + half]
            curve[centre] = float(np.sum(block * kernel))
        return np.maximum(curve, 0.0)

    def _pick_peaks(
        self, novelty: np.ndarray, tempo_map: TempoMap | None, times: np.ndarray
    ) -> list[int]:
        import librosa

        beats_per_bar = (tempo_map.beats_per_bar if tempo_map else None) or 4
        spacing = max(4, self._min_spacing_bars * beats_per_bar)
        picked = librosa.util.peak_pick(
            novelty,
            pre_max=spacing // 2,
            post_max=spacing // 2,
            pre_avg=spacing,
            post_avg=spacing,
            delta=self._peak_threshold,
            wait=spacing,
        )
        return [int(index) for index in picked if 0 <= index < times.size]

    def _to_boundaries(
        self,
        peaks: list[int],
        novelty: np.ndarray,
        times: np.ndarray,
        tempo_map: TempoMap | None,
    ) -> list[StructuralBoundary]:
        boundaries: list[StructuralBoundary] = []
        for index in peaks:
            time = float(times[index])
            bar: int | None = None
            beat_index: int | None = None
            snapped = False

            if tempo_map is not None and tempo_map.has_bars:
                # Sections change on bar lines, and the kernel cannot resolve
                # better than a few beats, so trust the grid over the peak.
                beats_per_bar = tempo_map.beats_per_bar
                downbeat = tempo_map.downbeat_beat
                assert beats_per_bar is not None and downbeat is not None  # has_bars
                bar_candidate = round((tempo_map.time_to_beat(time) - downbeat) / beats_per_bar)
                snapped_time = tempo_map.bar_to_time(bar_candidate)
                if abs(snapped_time - time) < 2.0:
                    time, bar, snapped = snapped_time, bar_candidate, True
                else:
                    bar = tempo_map.time_to_bar(time).bar
                beat_index = round(tempo_map.time_to_beat(time))

            boundaries.append(
                StructuralBoundary(
                    time=round(time, 3),
                    bar=bar,
                    beat_index=beat_index,
                    confidence=round(float(np.clip(novelty[index], 0.0, 1.0)), 4),
                    kind=BoundaryKind.STRUCTURAL,
                    snapped_to_downbeat=snapped,
                )
            )
        return boundaries
