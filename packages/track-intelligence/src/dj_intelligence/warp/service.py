"""
The warp workflow, end to end.

    analyse -> plan -> (decide) -> render -> re-analyse -> report

One place owns that sequence so the CLI and the API cannot implement it
differently, exactly as ``engine.analyze`` does for analysis. In particular
the refusal logic lives here rather than in either caller: a track whose grid
is already good must not be stretched just because one of the two front ends
forgot to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import AnalysisProfile, Settings, get_settings
from ..engine import get_pipeline
from ..errors import DJIntelligenceError
from ..models import TrackAnalysis, WarpProvenance, WarpRenderReport
from ..observability import get_logger
from .renderer import WARP_ALGORITHM_VERSION, WarpRenderer
from .verify import verify_render

__all__ = ["WarpOutcome", "warp_track"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WarpOutcome:
    """What happened, including the case where nothing did."""

    analysis: TrackAnalysis
    rendered: bool
    report: WarpRenderReport | None = None
    skipped_reason: str | None = None


def warp_track(
    source: Path | str,
    *,
    output: Path | None = None,
    target_bpm: float | None = None,
    force: bool = False,
    verify: bool = True,
    settings: Settings | None = None,
) -> WarpOutcome:
    """
    Analyse a track, and correct its grid if that is actually warranted.

    ``force`` overrides the recommendation — a user who wants the render can
    have it — but the recommendation is still computed and reported, so the
    override is a decision rather than an accident.
    """
    source_path = Path(source)
    resolved = (settings or get_settings()).with_overrides(profile=AnalysisProfile.WARP)
    pipeline = get_pipeline(resolved)

    analysis = pipeline.analyze(source_path, target_bpm=target_bpm)
    warp_map = analysis.warp
    if warp_map is None or not warp_map.markers:
        return WarpOutcome(
            analysis=analysis,
            rendered=False,
            skipped_reason="No beat grid was found, so there is nothing to warp to.",
        )

    recommendation = warp_map.recommendation
    if not recommendation.required and not force:
        return WarpOutcome(analysis=analysis, rendered=False, skipped_reason=recommendation.reason)

    destination = output or source_path.with_suffix(".warped.wav")
    if destination.resolve() == source_path.resolve():
        # Never in place. The source is the only copy of the original.
        raise DJIntelligenceError("warp output would overwrite the source file")

    renderer = WarpRenderer(
        ffmpeg_path=resolved.ffmpeg_path,
        crossfade_ms=resolved.warp_crossfade_ms,
        sample_rate=analysis.audio.source_sample_rate or resolved.sample_rate,
    )
    report = renderer.render(
        source_path,
        warp_map,
        destination,
        duration=analysis.audio.duration_seconds,
        channels=analysis.audio.source_channels or 2,
        sample_rate=analysis.audio.source_sample_rate or resolved.sample_rate,
    )

    if verify:

        def beats_of(path: Path) -> list[float]:
            return pipeline.analyze(path).beats

        report = report.model_copy(
            update={
                "verification": verify_render(
                    destination,
                    warp_map,
                    analyze_beats=beats_of,
                    source_beats=analysis.beats,
                    threshold_ms=resolved.warp_verification_threshold_ms,
                )
            }
        )

    provenance = WarpProvenance(
        algorithm_version=WARP_ALGORITHM_VERSION,
        target_bpm=warp_map.target_bpm,
        renderer=report.renderer,
        renderer_version=report.renderer_version,
        configuration={
            "max_grid_error_ms": resolved.warp_max_grid_error_ms,
            "max_marker_distance_bars": resolved.warp_max_marker_distance_bars,
            "min_marker_distance_beats": resolved.warp_min_marker_distance_beats,
            "crossfade_ms": resolved.warp_crossfade_ms,
            "forced": force,
        },
    )
    analysis = analysis.model_copy(
        update={"warp": warp_map.model_copy(update={"provenance": provenance})}
    )

    log.info(
        "warp.rendered",
        extra={
            "output": str(destination),
            "target_bpm": report.target_bpm,
            "markers": report.marker_count,
            "verified": report.verification.passed if report.verification else None,
            "mean_error_ms": report.verification.mean_grid_error_ms
            if report.verification
            else None,
        },
    )
    return WarpOutcome(analysis=analysis, rendered=True, report=report)
