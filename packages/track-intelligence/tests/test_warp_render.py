"""
Warping audio for real: render a drifting track and measure the result.

This is the most important test in this phase. Everything else checks that a
plan is sensible; this checks that applying the plan to actual samples with an
actual time stretcher actually puts the beats where they were supposed to go —
measured by analysing the rendered file from scratch, not by trusting the
renderer's own arithmetic.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from dj_intelligence.analysis.pipeline import AnalysisPipeline
from dj_intelligence.config import AnalysisProfile, Settings
from dj_intelligence.errors import DJIntelligenceError
from dj_intelligence.synth import Groove, render_groove, write_wav
from dj_intelligence.warp import WarpRenderer, grid_error_ms, warp_track

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings(key_engine="chroma", tempo_engine="chroma", log_level="ERROR")


@pytest.fixture(scope="module")
def renderer_available() -> bool:
    return WarpRenderer().available()


@pytest.fixture
def drifting_track(tmp_path: Path) -> Path:
    """125 -> 127 BPM: enough drift that a DJ would notice by the end."""
    return write_wav(
        tmp_path / "drifting.wav", render_groove(Groove(bpm=125.0, bpm_end=127.0, bars=48)).samples
    )


@pytest.fixture
def steady_track(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "steady.wav", render_groove(Groove(bpm=126.0, bars=48)).samples)


# -- the headline claim -----------------------------------------------------


def test_warping_a_drifting_track_fixes_its_grid(
    drifting_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    """
    Analyse, plan, render, re-analyse, and require a real improvement.

    The assertion is deliberately relative as well as absolute: an absolute
    error threshold could be met by a renderer that did nothing to a track
    that was already fine, so the rendered grid also has to be several times
    better than the input's.
    """
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")

    output = tmp_path / "corrected.wav"
    outcome = warp_track(drifting_track, output=output, settings=settings)

    assert outcome.rendered is True
    assert output.exists() and output.stat().st_size > 0
    report = outcome.report
    assert report is not None

    verification = report.verification
    assert verification is not None
    assert verification.source_mean_grid_error_ms is not None
    assert verification.mean_grid_error_ms < verification.source_mean_grid_error_ms / 4
    assert verification.mean_grid_error_ms < 15.0
    assert verification.passed is True


def test_the_source_is_never_modified(
    drifting_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")
    before = drifting_track.read_bytes()
    warp_track(drifting_track, output=tmp_path / "out.wav", settings=settings)
    assert drifting_track.read_bytes() == before


def test_writing_over_the_source_is_refused(drifting_track: Path, settings: Settings) -> None:
    with pytest.raises(DJIntelligenceError, match="overwrite the source"):
        warp_track(drifting_track, output=drifting_track, settings=settings)


def test_duration_is_preserved(
    drifting_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    """Correcting drift redistributes time; it does not add or remove it."""
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")
    outcome = warp_track(drifting_track, output=tmp_path / "same-length.wav", settings=settings)
    report = outcome.report
    assert report is not None
    assert report.output_duration_seconds == pytest.approx(report.source_duration_seconds, rel=0.01)


def test_pitch_is_not_shifted(
    drifting_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    """
    Time changes, pitch does not.

    Checked musically rather than by trusting the filter: the rendered file is
    analysed again and must come back in the same key. Resampling — the naive
    way to change tempo — would move the key by several semitones.
    """
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")

    pipeline = AnalysisPipeline(settings)
    before = pipeline.analyze(drifting_track)

    output = tmp_path / "pitch.wav"
    outcome = warp_track(drifting_track, output=output, settings=settings)
    assert outcome.rendered
    after = pipeline.analyze(output)

    assert after.tonality.key == before.tonality.key
    assert after.tonality.mode == before.tonality.mode
    assert outcome.report is not None
    assert outcome.report.pitch_shift_cents == 0.0


def test_rendered_tempo_matches_the_target(
    drifting_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")
    output = tmp_path / "tempo.wav"
    outcome = warp_track(drifting_track, output=output, target_bpm=126.0, settings=settings)
    assert outcome.rendered

    after = AnalysisPipeline(settings).analyze(output)
    assert after.tempo.bpm == pytest.approx(126.0, abs=0.6)


# -- refusing to over-correct -----------------------------------------------


def test_a_good_track_is_not_touched(
    steady_track: Path, tmp_path: Path, settings: Settings
) -> None:
    """The rule that protects most of a real library."""
    output = tmp_path / "should-not-exist.wav"
    outcome = warp_track(steady_track, output=output, settings=settings)

    assert outcome.rendered is False
    assert output.exists() is False
    assert outcome.skipped_reason and "within" in outcome.skipped_reason


def test_force_overrides_the_refusal(
    steady_track: Path, tmp_path: Path, settings: Settings, renderer_available: bool
) -> None:
    if not renderer_available:
        pytest.skip("this ffmpeg has no librubberband")
    output = tmp_path / "forced.wav"
    outcome = warp_track(steady_track, output=output, force=True, settings=settings)
    assert outcome.rendered is True
    assert output.exists()


# -- segment planning -------------------------------------------------------


def test_segments_tile_the_target_timeline(drifting_track: Path, settings: Settings) -> None:
    """No gaps and no overlaps, or the output would stutter at every marker."""
    pipeline = AnalysisPipeline(settings.with_overrides(profile=AnalysisProfile.WARP))
    analysis = pipeline.analyze(drifting_track)
    assert analysis.warp is not None

    spans = WarpRenderer().segments(analysis.warp, analysis.audio.duration_seconds)
    assert spans
    for earlier, later in itertools.pairwise(spans):
        assert later.target_start == pytest.approx(earlier.target_end, abs=1e-6)
        assert later.source_start == pytest.approx(earlier.source_end, abs=1e-6)
    assert all(span.ratio > 0 for span in spans)


def test_segment_ratios_stay_gentle(drifting_track: Path, settings: Settings) -> None:
    pipeline = AnalysisPipeline(settings.with_overrides(profile=AnalysisProfile.WARP))
    analysis = pipeline.analyze(drifting_track)
    assert analysis.warp is not None
    spans = WarpRenderer().segments(analysis.warp, analysis.audio.duration_seconds)
    assert all(0.95 < span.ratio < 1.05 for span in spans)


# -- verification maths -----------------------------------------------------


def test_grid_error_is_zero_on_a_perfect_grid() -> None:
    beats = np.arange(200) * (60.0 / 126.0) + 1.5
    errors = grid_error_ms(beats, target_bpm=126.0, anchor_time=1.5)
    assert float(np.max(errors)) < 1e-6


def test_grid_error_measures_distance_to_the_nearest_gridline() -> None:
    period = 60.0 / 126.0
    beats = np.arange(50) * period + 1.5 + 0.010  # 10 ms late, every beat
    errors = grid_error_ms(beats, target_bpm=126.0, anchor_time=1.5)
    assert float(np.mean(errors)) == pytest.approx(10.0, abs=0.1)


def test_grid_error_of_nothing_is_empty() -> None:
    assert grid_error_ms([], target_bpm=126.0, anchor_time=0.0).size == 0
