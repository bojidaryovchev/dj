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

from ..audio.decoder import DecodedAudio, FFmpegDecoder
from ..audio.hashing import sha256_file
from ..config import Settings, get_settings
from ..dj.interpret import interpret
from ..models import (
    AnalysisMetadata,
    AnalysisWarning,
    AudioProperties,
    KeyEstimate,
    LoudnessMeasurement,
    StageTiming,
    TempoEstimate,
    TonalSegment,
    TrackAnalysis,
    TrackIdentity,
    WarningCode,
)
from ..observability import get_logger, stage_timer
from ..version import package_version
from .base import KeyAnalyzer, LoudnessAnalyzer, SegmentKeyAnalyzer, TempoAnalysis, TempoAnalyzer
from .registry import (
    build_key_analyzer,
    build_loudness_analyzer,
    build_segment_analyzer,
    build_tempo_analyzer,
    build_tonal_content_gate,
)

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

    # -- public API --------------------------------------------------------

    def analyze(self, path: Path | str, *, display_name: str | None = None) -> TrackAnalysis:
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
            beats=tempo_result.beats,
            downbeats=tempo_result.downbeats,
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
