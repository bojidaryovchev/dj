"""
The pipeline end to end, against generated audio.

What these can and cannot prove is worth stating plainly. They prove the
chain is wired up, that the profiles point the right way, that uncertainty is
represented rather than papered over, and that a backend crashing does not
take the request with it. They do **not** prove accuracy on real records --
synthetic audio has no mastering, no reverb, no vocals and a far cleaner
spectrum than anything a producer prints. ``scripts/evaluate_dataset.py``
exists for that, and the README says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dj_intelligence.analysis.base import TempoAnalysis
from dj_intelligence.analysis.pipeline import AnalysisPipeline
from dj_intelligence.config import Settings
from dj_intelligence.models import TempoEstimate, WarningCode
from dj_intelligence.music.notes import Mode

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR"))


# -- the headline claim -----------------------------------------------------


@pytest.mark.parametrize(
    ("fixture_name", "tonic", "mode", "camelot"),
    [
        ("f_minor_wav", "F", Mode.MINOR, "4A"),
        ("a_minor_wav", "A", Mode.MINOR, "8A"),
        ("c_major_wav", "C", Mode.MAJOR, "8B"),
    ],
)
def test_key_mode_and_camelot(
    pipeline: AnalysisPipeline,
    request: pytest.FixtureRequest,
    fixture_name: str,
    tonic: str,
    mode: Mode,
    camelot: str,
) -> None:
    result = pipeline.analyze(request.getfixturevalue(fixture_name))
    assert result.tonality.key == tonic
    assert result.tonality.mode is mode
    assert result.tonality.reliable
    assert result.dj.camelot == camelot


def test_tempo_is_recovered(pipeline: AnalysisPipeline, f_minor_wav: Path) -> None:
    result = pipeline.analyze(f_minor_wav)
    assert result.tempo.bpm == pytest.approx(126.0, abs=1.0)
    assert result.tempo.reliable
    assert result.tempo.stable is True
    assert result.tempo.beat_count > 50
    assert len(result.beats) == result.tempo.beat_count


def test_the_result_carries_its_provenance(pipeline: AnalysisPipeline, f_minor_wav: Path) -> None:
    result = pipeline.analyze(f_minor_wav)
    meta = result.analysis
    assert meta.analysis_version == "1.0.0"
    assert meta.key_engine is not None and meta.key_engine.name == "chroma"
    assert meta.tempo_engine is not None
    assert meta.configuration_fingerprint
    assert meta.processing_time_ms > 0
    assert meta.realtime_ratio is not None
    assert {stage.stage for stage in meta.stages} >= {"hash", "decode", "key", "tempo"}


def test_track_identity_is_recorded(pipeline: AnalysisPipeline, f_minor_wav: Path) -> None:
    result = pipeline.analyze(f_minor_wav)
    assert len(result.track.sha256) == 64
    assert result.track.size_bytes > 0
    assert result.track.filename == f_minor_wav.name


def test_display_name_overrides_the_temp_filename(
    pipeline: AnalysisPipeline, f_minor_wav: Path
) -> None:
    result = pipeline.analyze(f_minor_wav, display_name="Artist - Title.mp3")
    assert result.track.filename == "Artist - Title.mp3"


def test_analysis_is_deterministic(pipeline: AnalysisPipeline, f_minor_wav: Path) -> None:
    """Same file, same engine, same configuration -> same measurements."""
    first = pipeline.analyze(f_minor_wav)
    second = pipeline.analyze(f_minor_wav)
    assert first.tonality == second.tonality
    assert first.tempo == second.tempo
    assert first.tonal_segments == second.tonal_segments
    assert first.track.sha256 == second.track.sha256


def test_dj_layer_is_derived_from_the_measurement(
    pipeline: AnalysisPipeline, f_minor_wav: Path
) -> None:
    result = pipeline.analyze(f_minor_wav)
    assert [k.camelot for k in result.dj.compatible_keys] == ["4A", "3A", "5A", "4B"]
    assert result.dj.key_label == "F minor"
    assert result.dj.mix_bpm == result.tempo.bpm


def test_loudness_is_measured(pipeline: AnalysisPipeline, f_minor_wav: Path) -> None:
    result = pipeline.analyze(f_minor_wav)
    assert result.loudness.integrated_lufs is not None
    assert -40.0 < result.loudness.integrated_lufs < 0.0
    assert result.loudness.sample_peak_dbfs is not None


# -- uncertainty ------------------------------------------------------------


def test_silence_claims_nothing(pipeline: AnalysisPipeline, silence_wav: Path) -> None:
    """The behaviour the whole confidence model exists to protect."""
    result = pipeline.analyze(silence_wav)
    assert result.tonality.key is None
    assert result.tonality.mode is None
    assert result.tonality.reliable is False
    assert result.dj.camelot is None
    assert result.tempo.bpm is None
    assert WarningCode.SILENT_AUDIO in {w.code for w in result.warnings}


def test_noise_is_assigned_no_key_at_all(pipeline: AnalysisPipeline, noise_wav: Path) -> None:
    """
    White noise has no key and must be reported as having none.

    Correlation alone will not do this: it is scale- and offset-invariant, so
    it once scored flat noise at 0.54 for F# minor. The tonal-salience gate
    is what makes this test pass, and this test is why it exists.
    """
    result = pipeline.analyze(noise_wav)
    assert result.tonality.key is None
    assert result.tonality.reliable is False
    assert result.dj.camelot is None


def test_percussion_only_gives_tempo_but_no_key(
    pipeline: AnalysisPipeline, click_wav: Path
) -> None:
    """A click track has a tempo and no tonality. Both must be reported
    correctly, and the tempo must not be withheld because the key failed."""
    result = pipeline.analyze(click_wav)
    assert result.tempo.bpm == pytest.approx(128.0, abs=2.0)
    assert result.tempo.reliable
    assert result.tonality.key is None


def test_low_confidence_is_warned_about(pipeline: AnalysisPipeline, noise_wav: Path) -> None:
    result = pipeline.analyze(noise_wav)
    codes = {w.code for w in result.warnings}
    assert codes & {WarningCode.LOW_KEY_CONFIDENCE, WarningCode.SILENT_AUDIO}


def test_truncation_is_reported(f_minor_wav: Path) -> None:
    settings = Settings(
        key_engine="chroma", tempo_engine="chroma", max_analysis_seconds=10.0, log_level="ERROR"
    )
    result = AnalysisPipeline(settings).analyze(f_minor_wav)
    assert result.audio.analysed_seconds == pytest.approx(10.0, abs=0.1)
    assert result.audio.duration_seconds == pytest.approx(40.0, abs=0.1)
    assert WarningCode.ANALYSIS_TRUNCATED in {w.code for w in result.warnings}


# -- resilience -------------------------------------------------------------


class _ExplodingTempoAnalyzer:
    """A backend that fails the way a real one might: mid-analysis."""

    name = "exploding"

    def describe(self):  # type: ignore[no-untyped-def]
        from dj_intelligence.models import EngineInfo

        return EngineInfo(name="exploding", algorithm="raises")

    def analyze(self, audio):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend fell over")


def test_a_failing_backend_degrades_instead_of_failing_the_request(
    f_minor_wav: Path,
) -> None:
    settings = Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")
    pipeline = AnalysisPipeline(settings, tempo_analyzer=_ExplodingTempoAnalyzer())

    result = pipeline.analyze(f_minor_wav)

    assert result.tempo.bpm is None
    assert WarningCode.TEMPO_ANALYSIS_FAILED in {w.code for w in result.warnings}
    # ...and the rest of the analysis still arrived.
    assert result.tonality.key == "F"
    assert result.dj.camelot == "4A"


class _OverconfidentKeyAnalyzer:
    """
    A backend that names a key for anything, and exposes no chromagram.

    Not a straw man: this is measured Essentia behaviour. Its ``KeyExtractor``
    returns C major with strength 0.76 for a bare click track and F major with
    0.70 for white noise, and it offers no way to ask whether the material was
    tonal in the first place.
    """

    name = "overconfident"

    def describe(self):  # type: ignore[no-untyped-def]
        from dj_intelligence.models import EngineInfo

        return EngineInfo(name="overconfident", algorithm="always C major")

    def analyze(self, audio):  # type: ignore[no-untyped-def]
        from dj_intelligence.models import ConfidenceType, KeyEstimate

        return KeyEstimate(
            pitch_class=0,
            mode=Mode.MAJOR,
            confidence=0.76,
            confidence_type=ConfidenceType.ESSENTIA_KEY_STRENGTH,
            reliable=True,
        )


def test_a_backend_that_never_says_no_is_still_gated(noise_wav: Path) -> None:
    """
    The guard has to work for backends that have none of their own.

    The chroma backend refuses noise by itself. Essentia does not, and it is
    the default inside the Docker image -- so the tonal-content check runs
    ahead of whichever backend is configured, and vetoes it.
    """
    settings = Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")
    pipeline = AnalysisPipeline(settings, key_analyzer=_OverconfidentKeyAnalyzer())

    result = pipeline.analyze(noise_wav)

    assert result.tonality.key is None
    assert result.dj.camelot is None
    assert WarningCode.NO_TONAL_CONTENT in {w.code for w in result.warnings}


def test_the_gate_does_not_veto_real_tonal_material(f_minor_wav: Path) -> None:
    """The guard must be a floor, not a filter -- ordinary music passes."""
    settings = Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")
    pipeline = AnalysisPipeline(settings, key_analyzer=_OverconfidentKeyAnalyzer())

    result = pipeline.analyze(f_minor_wav)

    assert result.tonality.key == "C"  # what the stub claims, not vetoed
    assert WarningCode.NO_TONAL_CONTENT not in {w.code for w in result.warnings}


class _HalfTimeTempoAnalyzer:
    name = "half-time"

    def describe(self):  # type: ignore[no-untyped-def]
        from dj_intelligence.models import EngineInfo

        return EngineInfo(name="half-time", algorithm="fixed")

    def analyze(self, audio):  # type: ignore[no-untyped-def]
        from dj_intelligence.analysis.tempo.common import tempo_candidates
        from dj_intelligence.models import ConfidenceType

        return TempoAnalysis(
            estimate=TempoEstimate(
                bpm=63.0,
                confidence=0.9,
                confidence_type=ConfidenceType.BEAT_INTERVAL_CONSISTENCY,
                reliable=True,
                candidates=tempo_candidates(63.0, dj_min=70.0, dj_max=180.0),
            )
        )


def test_half_time_readings_are_interpreted_not_overwritten(f_minor_wav: Path) -> None:
    settings = Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")
    pipeline = AnalysisPipeline(settings, tempo_analyzer=_HalfTimeTempoAnalyzer())

    result = pipeline.analyze(f_minor_wav)

    assert result.tempo.bpm == 63.0  # the measurement, untouched
    assert result.dj.mix_bpm == 126.0  # the interpretation
    assert WarningCode.TEMPO_OUT_OF_DJ_RANGE in {w.code for w in result.warnings}
