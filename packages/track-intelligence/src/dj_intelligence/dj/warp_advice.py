"""
Should this track be warped?

**DJ interpretation, not measurement.** The analysis layer says how far each
beat is from a constant grid; whether that justifies stretching the audio is a
judgement about tolerances, risk and taste, and it belongs here.

The default answer is no, and that is deliberate. Most modern electronic music
is sequenced and already sits on a perfect grid; running it through a time
stretcher trades an imperceptible timing gain for real damage to transients.
Warping has to earn its place, so it is recommended only when:

* the grid is trustworthy enough to warp *to* — a wrong grid warped
  confidently is far worse than a right grid left alone;
* the error without warping actually exceeds the tolerance; and
* the correction it would take is not so violent that it implies the analysis
  is wrong rather than the recording being loose.

That last one is the important safety valve. A required stretch of 0.75x or
1.4x almost never means "this track speeds up by a third" — it means the beat
tracker picked a half-time reading, or the downbeat phase is wrong. Rendering
in that state would produce confidently mangled audio, so the map is returned
with a refusal and the metrics that explain it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from ..models import DriftClassification, WarpMap, WarpRecommendation, WarpSkipReason

__all__ = ["WarpAdviceRules", "recommend_warp"]

_MS: Final = 1000.0


@dataclass(frozen=True, slots=True)
class WarpAdviceRules:
    """
    Thresholds for the recommendation. Explicit, documented, configurable.

    ``tolerance_ms`` at 15 ms: two percussive sources drifting further apart
    than about 20 ms read as flam rather than as one hit, so a grid inside
    15 ms is not worth touching.

    ``min_grid_confidence`` at 0.5: below this the grid is not a reliable
    target, and correcting audio towards a grid we do not believe is the worst
    available outcome.
    """

    tolerance_ms: float = 15.0
    min_grid_confidence: float = 0.5
    min_safe_stretch_ratio: float = 0.9
    max_safe_stretch_ratio: float = 1.1


def recommend_warp(
    warp_map: WarpMap,
    *,
    grid_confidence: float,
    drift: DriftClassification,
    tempo_reliable: bool,
    target_bpm_requested: bool = False,
    rules: WarpAdviceRules | None = None,
) -> WarpRecommendation:
    """
    Decide whether to render, and say why either way.

    ``target_bpm_requested`` marks an explicit user request for a different
    tempo. That is not a correction and is not subject to the "already close
    enough" test — the user asked for 128 and 126 is not 128 however tidy its
    grid is.
    """
    resolved = rules or WarpAdviceRules()
    metrics = warp_map.metrics
    # The systematic error, not the raw maximum. A beat tracker's jitter puts
    # the odd beat 40 ms from an otherwise perfect grid, and warping to chase
    # that would stretch a sequenced track to fix noise -- the exact
    # over-correction this system is supposed to refuse.
    error_ms = metrics.systematic_error_ms

    def refuse(reason: str, skip: WarpSkipReason) -> WarpRecommendation:
        return WarpRecommendation(
            required=False,
            reason=reason,
            skip_reason=skip,
            source_grid_error_ms=error_ms,
            tolerance_ms=resolved.tolerance_ms,
        )

    if not warp_map.markers or grid_confidence < resolved.min_grid_confidence:
        return refuse(
            f"Grid confidence {grid_confidence:.2f} is below "
            f"{resolved.min_grid_confidence:.2f}; there is no trustworthy grid to warp to.",
            WarpSkipReason.NO_GRID,
        )

    if not tempo_reliable:
        return refuse(
            "Tempo interpretation is not reliable enough to warp against. A "
            "half- or double-time reading would stretch the track by 2x.",
            WarpSkipReason.TEMPO_UNRELIABLE,
        )

    unsafe = (
        metrics.min_stretch_ratio < resolved.min_safe_stretch_ratio
        or metrics.max_stretch_ratio > resolved.max_safe_stretch_ratio
    )
    if unsafe:
        return refuse(
            f"Correction would need a local stretch of "
            f"{metrics.min_stretch_ratio:.3f}-{metrics.max_stretch_ratio:.3f}, outside the "
            f"safe range {resolved.min_safe_stretch_ratio:g}-{resolved.max_safe_stretch_ratio:g}. "
            f"This usually means the beat grid or the metrical level is wrong, "
            f"not that the track really moves that much.",
            WarpSkipReason.UNSAFE_STRETCH,
        )

    if not target_bpm_requested and error_ms <= resolved.tolerance_ms:
        return refuse(
            f"Source grid is already within {resolved.tolerance_ms:g} ms "
            f"(systematic error {error_ms:.1f} ms; worst single beat "
            f"{metrics.max_correction_ms:.1f} ms, which is detector jitter). "
            f"Warping would cost transient quality for no audible gain.",
            WarpSkipReason.ALREADY_ALIGNED,
        )

    why = (
        f"Requested target tempo {warp_map.target_bpm:g} BPM."
        if target_bpm_requested
        else f"Grid drifts {error_ms:.1f} ms from constant tempo "
        f"({drift.value}), over the {resolved.tolerance_ms:g} ms tolerance."
    )
    return WarpRecommendation(
        required=True,
        reason=why,
        source_grid_error_ms=error_ms,
        tolerance_ms=resolved.tolerance_ms,
    )
