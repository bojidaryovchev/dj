"""
Correcting the beat grid's phase.

Beat trackers are built to find tempo and pulse, and they are good at it. They
are not built to place a beat to the millisecond, and they do not: the onset
envelope a tracker works from is a spectral flux, which peaks *after* the
transient that caused it, so every reported beat comes out systematically
late. Measured on generated fixtures with a known grid, librosa's beats land
**+29.8 ms late with a standard deviation of only 7.1 ms**.

That shape is the important part. The spacing is good — the tracker's jitter
is 7 ms — and only the phase is wrong, by an amount that barely varies. So
the fix is a single global offset, not a per-beat snap:

* **Per-beat snapping made it worse.** Moving each beat individually to its
  local attack cut the bias but *raised* the jitter from 7 ms to 10 ms,
  because some beats got pulled onto the wrong transient. Net MAE 11.5 ms.
* **One offset, robustly estimated, keeps the tracker's regularity.** Each
  beat proposes a correction, the median of those proposals is applied to the
  whole grid, and the spacing is left exactly as the tracker found it. Net
  MAE 9.5 ms, jitter unchanged.

DJ software calls this the grid offset, and exposes it for the same reason.

The residual is around +9 ms and is at the resolution limit of the measurement
window: locating the start of a kick needs a window long enough to contain its
fundamental, and that window is itself ~6 ms. Shortening it makes things worse,
not better — a 64-sample window scored MAE 21 ms because it cannot see the low
end at all.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...audio.decoder import DecodedAudio

__all__ = ["GridOffset", "estimate_grid_offset"]

_ANALYSIS_RATE: Final = 22050
# 128 samples (5.8 ms) with a 32-sample hop (1.5 ms). Measured against 64/16
# and 256/64: this pair localises the attack best across constant, drifting,
# lead-in and beatless-intro fixtures.
_FRAME: Final = 128
_HOP: Final = 32
# Fraction of the peak-above-floor rise that counts as "the attack started".
_ATTACK_THRESHOLD: Final = 0.30
# How far around a beat to look, as a fraction of the beat period. Wider and
# a proposal can reach the neighbouring beat.
_SEARCH_FRACTION: Final = 0.20
_MAX_SHIFT_FRACTION: Final = 0.25
_MIN_BEATS: Final = 8


class GridOffset:
    """The measured phase error of a beat grid, and how it was obtained."""

    __slots__ = ("beats_used", "seconds", "spread_ms")

    def __init__(self, seconds: float, beats_used: int, spread_ms: float) -> None:
        self.seconds = seconds
        self.beats_used = beats_used
        self.spread_ms = spread_ms

    @property
    def milliseconds(self) -> float:
        return self.seconds * 1000.0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<GridOffset {self.milliseconds:+.2f}ms from {self.beats_used} beats>"


def estimate_grid_offset(
    audio: DecodedAudio,
    beat_times: list[float] | np.ndarray,
    *,
    max_shift_fraction: float = _MAX_SHIFT_FRACTION,
) -> GridOffset:
    """
    Estimate how late (or early) a whole beat grid sits.

    Add the returned ``seconds`` to every beat time. Returns zero when there
    is not enough evidence, which is the safe answer: an unshifted grid is
    merely late, a wrongly shifted one is wrong.
    """
    times = np.asarray(beat_times, dtype=np.float64)
    if times.size < _MIN_BEATS:
        return GridOffset(0.0, 0, 0.0)

    import librosa

    signal = audio.resampled(_ANALYSIS_RATE)
    if signal.size < _FRAME * 4:
        return GridOffset(0.0, 0, 0.0)

    envelope = librosa.feature.rms(y=signal, frame_length=_FRAME, hop_length=_HOP)[0]
    frame_seconds = _HOP / _ANALYSIS_RATE

    period = float(np.median(np.diff(times)))
    if period <= 0:
        return GridOffset(0.0, 0, 0.0)
    search = round(_SEARCH_FRACTION * period / frame_seconds)
    limit = max_shift_fraction * period

    proposals: list[float] = []
    for beat in times:
        centre = round(beat / frame_seconds)
        low = max(0, centre - search)
        high = min(envelope.size, centre + search + 1)
        if high - low < 4:
            continue

        window = envelope[low:high]
        peak_index = int(np.argmax(window))
        peak = float(window[peak_index])
        floor = float(np.min(window))
        if peak <= floor:
            continue

        # Walk back down the leading edge to where the attack began. Stop at
        # a local minimum so a decaying tail from the previous beat cannot
        # drag the estimate backwards.
        level = floor + _ATTACK_THRESHOLD * (peak - floor)
        index = peak_index
        while index > 0 and window[index - 1] > level and window[index - 1] <= window[index]:
            index -= 1

        proposal = (low + index) * frame_seconds - beat
        if abs(proposal) <= limit:
            proposals.append(proposal)

    if len(proposals) < _MIN_BEATS:
        return GridOffset(0.0, 0, 0.0)

    corrections = np.array(proposals, dtype=np.float64)
    # Median, not mean: a handful of beats will land on the wrong transient
    # and their proposals must not move the grid.
    offset = float(np.median(corrections))
    spread = float(np.percentile(np.abs(corrections - offset), 68)) * 1000.0
    return GridOffset(seconds=offset, beats_used=len(proposals), spread_ms=spread)
