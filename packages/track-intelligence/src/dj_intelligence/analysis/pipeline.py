"""
The analysis pipeline.

    file -> hash -> decode/normalise -> [tempo | key | segments | loudness]
         -> aggregate -> DJ interpretation -> TrackAnalysis

The pipeline owns orchestration and nothing else. It does not know how a key
is estimated, only that something satisfying ``KeyAnalyzer`` will estimate
one; adding a feature extractor means adding a stage here and an
implementation there.

It is also the only place allowed to import the ``dj`` package, in its role
as the result aggregator -- the final step is asking the interpretation layer
to translate the measurements it has collected.

**Every stage is individually recoverable.** A backend that throws does not
fail the request: the corresponding measurement comes back as "unknown", a
warning records what happened, and the rest of the analysis is still
returned. A track whose key detection crashed still has a usable BPM.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..audio.decoder import DecodedAudio, FFmpegDecoder
from ..audio.hashing import sha256_file
from ..config import Settings, get_settings
from ..dj.interpret import interpret
from ..dj.warp_advice import WarpAdviceRules, recommend_warp
from ..models import (
    AnalysisMetadata,
    AnalysisWarning,
    AudioProperties,
    KeyEstimate,
    LoudnessMeasurement,
    PhraseGridEntry,
    RhythmAnalysis,
    StageTiming,
    StructureAnalysis,
    TempoEstimate,
    TonalSegment,
    TrackAnalysis,
    TrackIdentity,
    WarningCode,
    WarpMap,
)
from ..observability import get_logger, stage_timer
from ..timeline.navigation import Navigator
from ..timeline.warp_map import WarpParameters, build_warp_map
from ..version import package_version
from .base import (
    KeyAnalyzer,
    LoudnessAnalyzer,
    SegmentKeyAnalyzer,
    SupportsChromagram,
    TempoAnalysis,
    TempoAnalyzer,
)
from .registry import (
    build_key_analyzer,
    build_loudness_analyzer,
    build_segment_analyzer,
    build_tempo_analyzer,
    build_tonal_content_gate,
)
from .rhythm.stage import RhythmResult, RhythmStage
from .structure.novelty import NoveltyStructureAnalyzer

__all__ = ["AnalysisPipeline"]

log = get_logger(__name__)


class AnalysisPipeline:
    """
    Analyses one file at a time. Cheap to construct, safe to reuse.

    Analysers are built once and held, because some of them (Essentia's
    algorithms in particular) pay a real setup cost that should not be
    repeated per track in a bulk run. Nothing about a previous file is
    retained between calls.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        decoder: FFmpegDecoder | None = None,
        key_analyzer: KeyAnalyzer | None = None,
        tempo_analyzer: TempoAnalyzer | None = None,
        segment_analyzer: SegmentKeyAnalyzer | None = None,
        loudness_analyzer: LoudnessAnalyzer | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.decoder = decoder or FFmpegDecoder(
            ffmpeg_path=self.settings.ffmpeg_path,
            ffprobe_path=self.settings.ffprobe_path,
            sample_rate=self.settings.sample_rate,
            min_duration_seconds=self.settings.min_duration_seconds,
        )
        self.key_analyzer = key_analyzer or build_key_analyzer(self.settings)
        self.tempo_analyzer = tempo_analyzer or build_tempo_analyzer(self.settings)
        self.segment_analyzer = segment_analyzer or build_segment_analyzer(
            self.key_analyzer, self.settings
        )
        self.loudness_analyzer = (
            loudness_analyzer
            if loudness_analyzer is not None
            else build_loudness_analyzer(self.settings)
        )
        self.tonal_content_gate = build_tonal_content_gate(self.key_analyzer, self.settings)

        # Both reuse the key analyser's chromagram when it has one, so
        # harmonic evidence for bar phase and for structure is effectively
        # free. See DecodedAudio.features.
        chroma_source = (
            self.key_analyzer if isinstance(self.key_analyzer, SupportsChromagram) else None
        )
        self.rhythm_stage = RhythmStage(self.settings, chroma_source=chroma_source)
        self.structure_analyzer = NoveltyStructureAnalyzer(
            chroma_source=chroma_source,
            min_spacing_bars=self.settings.structure_min_spacing_bars,
        )

    # -- public API --------------------------------------------------------

    def analyze(
        self,
        path: Path | str,
        *,
        display_name: str | None = None,
        target_bpm: float | None = None,
    ) -> TrackAnalysis:
        """
        Analyse one audio file.

        ``display_name`` overrides the filename in the result, which matters
        for uploads: the temporary file is called something like
        ``tmp8f3a.mp3`` and the caller wants their own name back.

        Raises :class:`~dj_intelligence.errors.AudioIngestError` subclasses if
        the file cannot be decoded at all. Every other failure is reported in
        the result.
        """
        source = Path(path)
        started = time.perf_counter()
        warnings: list[AnalysisWarning] = []
        stages: list[StageTiming] = []

        log.info("analysis.started", extra={"file": display_name or source.name})

        with stage_timer(log, "hash") as fields:
            digest = sha256_file(source)
            size_bytes = source.stat().st_size
            fields["sha256"] = digest[:12]
        stages.append(StageTiming(stage="hash", duration_ms=fields["duration_ms"]))

        with stage_timer(log, "decode") as fields:
            audio = self.decoder.decode(
                source, max_seconds=self.settings.max_analysis_seconds or None
            )
            fields["duration_seconds"] = round(audio.duration_seconds, 2)
            fields["codec"] = audio.source.codec
        stages.append(StageTiming(stage="decode", duration_ms=fields["duration_ms"]))

        if audio.truncated:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.ANALYSIS_TRUNCATED,
                    message=(
                        f"Only the first {self.settings.max_analysis_seconds:g}s were analysed "
                        f"(DJTI_MAX_ANALYSIS_SECONDS)."
                    ),
                    stage="decode",
                )
            )

        silent = audio.rms_dbfs < self.settings.silence_rms_dbfs
        if silent:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.SILENT_AUDIO,
                    message=(
                        f"Signal level {audio.rms_dbfs:.1f} dBFS RMS is below the silence "
                        f"threshold; no tonal or rhythmic claim is made."
                    ),
                    stage="decode",
                )
            )

        tempo_result = self._run_tempo(audio, silent, warnings, stages)

        # Asked before any key backend runs, and of the signal rather than of
        # the estimator: a key extractor will name a key for a drum loop, and
        # Essentia measurably does (C major, strength 0.76, for a bare click
        # track). Skipping the key stage entirely for material with no pitched
        # content is both the correct answer and the cheaper one.
        tonal = not silent and self._has_tonal_content(audio, warnings, stages)
        key_estimate = self._run_key(audio, silent or not tonal, warnings, stages)
        segments = self._run_segments(audio, silent or not tonal, warnings, stages)
        loudness = self._run_loudness(audio, warnings, stages)

        rhythm = self._run_rhythm(audio, tempo_result, warnings, stages)
        structure = self._run_structure(audio, rhythm, warnings, stages)
        warp = self._run_warp(rhythm, tempo_result, warnings, stages, target_bpm=target_bpm)

        self._flag_confidence(key_estimate, tempo_result.estimate, warnings)

        dj_view = interpret(
            key=key_estimate,
            tempo=tempo_result.estimate,
            segments=segments,
            dj_bpm_min=self.settings.dj_bpm_min,
            dj_bpm_max=self.settings.dj_bpm_max,
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        analysed_seconds = audio.duration_seconds
        result = TrackAnalysis(
            track=TrackIdentity(
                filename=display_name or source.name,
                sha256=digest,
                size_bytes=size_bytes,
            ),
            audio=AudioProperties(
                duration_seconds=round(audio.source.duration_seconds or analysed_seconds, 3),
                analysis_sample_rate=audio.sample_rate,
                source_sample_rate=audio.source.sample_rate,
                source_channels=audio.source.channels,
                codec=audio.source.codec,
                container=audio.source.container,
                bit_rate_bps=audio.source.bit_rate_bps,
                analysed_seconds=round(analysed_seconds, 3),
            ),
            tempo=tempo_result.estimate,
            tonality=key_estimate,
            loudness=loudness,
            tonal_segments=segments,
            beats=rhythm.beat_times or tempo_result.beats,
            downbeats=rhythm.downbeat_times,
            rhythm=rhythm.analysis,
            structure=structure,
            warp=warp,
            dj=dj_view,
            warnings=warnings,
            analysis=AnalysisMetadata(
                package_version=package_version(),
                key_engine=self.key_analyzer.describe(),
                tempo_engine=self.tempo_analyzer.describe(),
                configuration_fingerprint=self.settings.analysis_fingerprint,
                parameters=self.settings.analysis_parameters(),
                processing_time_ms=round(elapsed_ms, 1),
                realtime_ratio=(
                    round(elapsed_ms / 1000.0 / analysed_seconds, 5) if analysed_seconds else None
                ),
                stages=stages,
            ),
        )

        log.info(
            "analysis.completed",
            extra={
                "file": result.track.filename,
                "camelot": dj_view.camelot,
                "bpm": tempo_result.estimate.bpm,
                "bars": rhythm.analysis.grid.bar_count,
                "grid_confidence": rhythm.analysis.grid.confidence,
                "drift": rhythm.analysis.drift.classification.value,
                "warp_required": warp.recommendation.required if warp else None,
                "key_confidence": key_estimate.confidence,
                "tempo_confidence": tempo_result.estimate.confidence,
                "duration_ms": round(elapsed_ms, 1),
                "realtime_ratio": result.analysis.realtime_ratio,
                "warnings": len(warnings),
            },
        )
        return result

    # -- stages ------------------------------------------------------------

    def _run_tempo(
        self,
        audio: DecodedAudio,
        silent: bool,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> TempoAnalysis:
        if silent:
            return TempoAnalysis(estimate=TempoEstimate.unknown())
        try:
            with stage_timer(log, "tempo", engine=self.tempo_analyzer.name) as fields:
                result = self.tempo_analyzer.analyze(audio)
                fields["bpm"] = result.estimate.bpm
                fields["confidence"] = result.estimate.confidence
            stages.append(StageTiming(stage="tempo", duration_ms=fields["duration_ms"]))
            return result
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.TEMPO_ANALYSIS_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="tempo",
                )
            )
            return TempoAnalysis(estimate=TempoEstimate.unknown())

    def _has_tonal_content(
        self,
        audio: DecodedAudio,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> bool:
        try:
            with stage_timer(log, "tonal_content") as fields:
                tonal, salience = self.tonal_content_gate.has_tonal_content(audio)
                fields["salience"] = round(salience, 5)
                fields["tonal"] = tonal
            stages.append(StageTiming(stage="tonal_content", duration_ms=fields["duration_ms"]))
        except Exception as exc:
            log.warning("tonal_content.failed", extra={"error": str(exc)})
            return True

        if not tonal:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.NO_TONAL_CONTENT,
                    message=(
                        f"Pitch-class energy is essentially flat (salience {salience:.4f} < "
                        f"{self.tonal_content_gate.min_salience:g}); this is percussion, noise "
                        f"or atonal material, so no key is claimed."
                    ),
                    stage="tonal_content",
                )
            )
        return tonal

    def _run_key(
        self,
        audio: DecodedAudio,
        silent: bool,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> KeyEstimate:
        if silent:
            return KeyEstimate.unknown()
        try:
            with stage_timer(log, "key", engine=self.key_analyzer.name) as fields:
                estimate = self.key_analyzer.analyze(audio)
                fields["key"] = estimate.key
                fields["mode"] = estimate.mode.value if estimate.mode else None
                fields["confidence"] = estimate.confidence
            stages.append(StageTiming(stage="key", duration_ms=fields["duration_ms"]))
            return estimate
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.KEY_ANALYSIS_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="key",
                )
            )
            return KeyEstimate.unknown()

    def _run_segments(
        self,
        audio: DecodedAudio,
        silent: bool,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> list[TonalSegment]:
        if not self.settings.segments_enabled or silent:
            return []
        if audio.duration_seconds < self.settings.segment_window_seconds:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.SEGMENTATION_SKIPPED,
                    message=(
                        f"Track is shorter than the {self.settings.segment_window_seconds:g}s "
                        f"analysis window."
                    ),
                    stage="segments",
                )
            )
            return []
        try:
            with stage_timer(log, "segments") as fields:
                segments = self.segment_analyzer.analyze(audio)
                fields["count"] = len(segments)
            stages.append(StageTiming(stage="segments", duration_ms=fields["duration_ms"]))
            return segments
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.SEGMENTATION_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="segments",
                )
            )
            return []

    def _run_loudness(
        self,
        audio: DecodedAudio,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> LoudnessMeasurement:
        if self.loudness_analyzer is None:
            return LoudnessMeasurement()
        try:
            with stage_timer(log, "loudness") as fields:
                loudness = self.loudness_analyzer.analyze(audio)
                fields["integrated_lufs"] = loudness.integrated_lufs
            stages.append(StageTiming(stage="loudness", duration_ms=fields["duration_ms"]))
            if loudness.integrated_lufs is None:
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.LOUDNESS_UNAVAILABLE,
                        message="ffmpeg did not report an integrated loudness for this file.",
                        stage="loudness",
                    )
                )
            return loudness
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.LOUDNESS_UNAVAILABLE,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="loudness",
                )
            )
            return LoudnessMeasurement()

    # -- rhythm, structure and warp ----------------------------------------

    def _run_rhythm(
        self,
        audio: DecodedAudio,
        tempo_result: TempoAnalysis,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> RhythmResult:
        """Bars, meter, local tempo and the grid. Needs beats to work from."""
        if not self.settings.profile.includes_rhythm or len(tempo_result.beats) < 2:
            return RhythmResult(analysis=RhythmAnalysis())
        try:
            with stage_timer(log, "rhythm") as fields:
                result = self.rhythm_stage.run(audio, tempo_result.estimate, tempo_result.beats)
                fields["bars"] = result.analysis.grid.bar_count
                fields["beats_per_bar"] = result.analysis.meter.beats_per_bar
                fields["grid_offset_ms"] = result.grid_offset_ms
                fields["drift"] = result.analysis.drift.classification.value
            stages.append(StageTiming(stage="rhythm", duration_ms=fields["duration_ms"]))
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.RHYTHM_ANALYSIS_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="rhythm",
                )
            )
            return RhythmResult(analysis=RhythmAnalysis())

        if result.analysis.meter.beats_per_bar is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.DOWNBEATS_UNAVAILABLE,
                    message=(
                        "No bar phase could be established, so bars, phrases and bar-level "
                        "navigation are unavailable. Set DJTI_FALLBACK_BEATS_PER_BAR to "
                        "assume a meter anyway."
                    ),
                    stage="rhythm",
                )
            )
        return result

    def _run_structure(
        self,
        audio: DecodedAudio,
        rhythm: RhythmResult,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
    ) -> StructureAnalysis:
        """Boundaries where the music changes, plus the phrase grid."""
        if not (self.settings.profile.includes_structure and self.settings.structure_enabled):
            return StructureAnalysis()
        if rhythm.tempo_map is None or not rhythm.beat_times:
            return StructureAnalysis()

        boundaries = []
        try:
            with stage_timer(log, "structure") as fields:
                boundaries = self.structure_analyzer.analyze(
                    audio, np.asarray(rhythm.beat_times, dtype=np.float64), rhythm.tempo_map
                )
                fields["boundaries"] = len(boundaries)
            stages.append(StageTiming(stage="structure", duration_ms=fields["duration_ms"]))
        except Exception as exc:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.STRUCTURE_ANALYSIS_FAILED,
                    message=f"{type(exc).__name__}: {exc}",
                    stage="structure",
                )
            )

        # The phrase grid is arithmetic, not detection: it always exists when
        # there are bars, and it never depends on whether novelty found
        # anything.
        phrase_grid = []
        if rhythm.tempo_map.has_bars:
            navigator = Navigator(rhythm.tempo_map, phrase_bars=self.settings.phrase_bars)
            phrase_grid = [
                PhraseGridEntry(
                    index=window.index,
                    start_bar=window.start_bar,
                    bars=window.bars,
                    start_time=window.start_time,
                    end_time=window.end_time,
                )
                for window in navigator.phrase_grid(audio.duration_seconds)
            ]

        return StructureAnalysis(
            boundaries=boundaries,
            phrase_grid=phrase_grid,
            phrase_bars=self.settings.phrase_bars if phrase_grid else None,
        )

    def _run_warp(
        self,
        rhythm: RhythmResult,
        tempo_result: TempoAnalysis,
        warnings: list[AnalysisWarning],
        stages: list[StageTiming],
        *,
        target_bpm: float | None,
    ) -> WarpMap | None:
        """Plan a correction. Never applies one -- see ``warp.renderer``."""
        if not self.settings.profile.includes_warp:
            return None
        if rhythm.tempo_map is None:
            return None

        with stage_timer(log, "warp") as fields:
            warp_map = build_warp_map(
                rhythm.tempo_map,
                parameters=WarpParameters(
                    target_bpm=target_bpm,
                    max_grid_error_ms=self.settings.warp_max_grid_error_ms,
                    max_marker_distance_bars=self.settings.warp_max_marker_distance_bars,
                    min_marker_distance_beats=self.settings.warp_min_marker_distance_beats,
                    min_safe_stretch_ratio=self.settings.warp_min_safe_stretch_ratio,
                    max_safe_stretch_ratio=self.settings.warp_max_safe_stretch_ratio,
                    tolerance_ms=self.settings.warp_tolerance_ms,
                ),
            )
            recommendation = recommend_warp(
                warp_map,
                grid_confidence=rhythm.analysis.grid.confidence,
                drift=rhythm.analysis.drift.classification,
                tempo_reliable=tempo_result.estimate.reliable,
                target_bpm_requested=target_bpm is not None,
                rules=WarpAdviceRules(
                    tolerance_ms=self.settings.warp_tolerance_ms,
                    min_grid_confidence=self.settings.warp_min_grid_confidence,
                    min_safe_stretch_ratio=self.settings.warp_min_safe_stretch_ratio,
                    max_safe_stretch_ratio=self.settings.warp_max_safe_stretch_ratio,
                ),
            )
            warp_map = warp_map.model_copy(update={"recommendation": recommendation})
            fields["markers"] = warp_map.metrics.marker_count
            fields["required"] = recommendation.required
        stages.append(StageTiming(stage="warp", duration_ms=fields["duration_ms"]))

        for warning in warp_map.warnings:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.WARP_LARGE_STRETCH,
                    message=(
                        f"Correction implies a local stretch of "
                        f"{warp_map.metrics.min_stretch_ratio:.3f}-"
                        f"{warp_map.metrics.max_stretch_ratio:.3f}."
                    ),
                    stage="warp",
                )
                if warning == "warp_requires_large_local_stretch"
                else AnalysisWarning(
                    code=WarningCode.WARP_LARGE_STRETCH, message=warning, stage="warp"
                )
            )
        return warp_map

    # -- post-conditions ---------------------------------------------------

    def _flag_confidence(
        self,
        key: KeyEstimate,
        tempo: TempoEstimate,
        warnings: list[AnalysisWarning],
    ) -> None:
        """
        Say out loud what the numbers already imply.

        A caller reading ``reliable`` learns the same thing, but a low-
        confidence result that carries no warning reads like a confident one
        to anyone skimming, and skimming is what people do.
        """
        if key.pitch_class is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.LOW_KEY_CONFIDENCE,
                    message="No key could be determined for this material.",
                    stage="key",
                )
            )
        elif not key.reliable:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.LOW_KEY_CONFIDENCE,
                    message=(
                        f"Key confidence {key.confidence:.2f} ({key.confidence_type.value}) is "
                        f"below the {self.settings.key_min_reliability:.2f} threshold; treat "
                        f"the key as a suggestion."
                    ),
                    stage="key",
                )
            )

        if tempo.bpm is None:
            warnings.append(
                AnalysisWarning(
                    code=WarningCode.LOW_TEMPO_CONFIDENCE,
                    message="No tempo could be determined for this material.",
                    stage="tempo",
                )
            )
        else:
            if not tempo.reliable:
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.LOW_TEMPO_CONFIDENCE,
                        message=(
                            f"Tempo confidence {tempo.confidence:.2f} "
                            f"({tempo.confidence_type.value}) is below the "
                            f"{self.settings.tempo_min_reliability:.2f} threshold."
                        ),
                        stage="tempo",
                    )
                )
            if tempo.stable is False:
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.TEMPO_UNSTABLE,
                        message=(
                            f"Beat spacing varies by {tempo.beat_interval_cv:.1%}; the track "
                            f"may be live, hand-edited or tempo-automated."
                        ),
                        stage="tempo",
                    )
                )
            if not (self.settings.dj_bpm_min <= tempo.bpm <= self.settings.dj_bpm_max):
                warnings.append(
                    AnalysisWarning(
                        code=WarningCode.TEMPO_OUT_OF_DJ_RANGE,
                        message=(
                            f"{tempo.bpm:.2f} BPM is outside {self.settings.dj_bpm_min:g}-"
                            f"{self.settings.dj_bpm_max:g}; see tempo.candidates for the "
                            f"half/double readings and dj.mix_bpm for the one chosen."
                        ),
                        stage="tempo",
                    )
                )
