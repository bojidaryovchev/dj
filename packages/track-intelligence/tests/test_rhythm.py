"""
Rhythmic analysis against fixtures whose grid is known exactly.

``render_groove`` returns the beat times it synthesised, so these are real
accuracy measurements rather than one estimator agreeing with another. Every
way a grid can go wrong has a fixture: tempo that ramps, tempo that steps, a
first downbeat that is not at zero, an intro with no beats in it, missing
kicks, and off-beat percussion designed to fool a detector that just follows
transients.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from dj_intelligence.analysis.pipeline import AnalysisPipeline
from dj_intelligence.analysis.rhythm import classify_drift, local_tempo_curve
from dj_intelligence.analysis.tempo.common import musical_beat_indices
from dj_intelligence.config import AnalysisProfile, Settings
from dj_intelligence.models import DriftClassification
from dj_intelligence.synth import Groove, GrooveRender, render_groove, write_wav

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline() -> AnalysisPipeline:
    return AnalysisPipeline(
        Settings(
            key_engine="chroma",
            tempo_engine="chroma",
            profile=AnalysisProfile.WARP,
            log_level="ERROR",
        )
    )


def analysed(
    pipeline: AnalysisPipeline, groove: Groove, tmp_path: Path, name: str
) -> tuple[GrooveRender, object]:
    render = render_groove(groove)
    path = write_wav(tmp_path / f"{name}.wav", render.samples)
    return render, pipeline.analyze(path)


def beat_errors_ms(detected: list[float], truth: np.ndarray) -> np.ndarray:
    """Distance from each true beat to the nearest detected one."""
    found = np.asarray(detected, dtype=np.float64)
    if found.size == 0:
        return np.array([np.inf])
    return np.array([np.min(np.abs(found - moment)) for moment in truth]) * 1000.0


# -- beat timing ------------------------------------------------------------


def test_beat_timing_accuracy(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    """
    The grid must land on the beats, not near them.

    Raw librosa beats are ~30 ms late systematically; the grid-offset stage
    exists to remove that, and this is the test that would catch its removal.
    """
    render, result = analysed(pipeline, Groove(bpm=126.0, bars=48), tmp_path, "timing")
    errors = beat_errors_ms(result.beats, render.beat_times)
    assert float(np.mean(errors)) < 15.0
    assert float(np.percentile(errors, 95)) < 30.0


def test_offset_correction_can_be_disabled(tmp_path: Path) -> None:
    """Without it the grid is measurably, systematically late."""
    render = render_groove(Groove(bpm=126.0, bars=32))
    path = write_wav(tmp_path / "raw.wav", render.samples)
    settings = Settings(
        key_engine="chroma",
        tempo_engine="chroma",
        beat_offset_refinement=False,
        log_level="ERROR",
    )
    result = AnalysisPipeline(settings).analyze(path)
    errors = beat_errors_ms(result.beats, render.beat_times)
    assert float(np.mean(errors)) > 15.0


# -- downbeats and meter ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "groove"),
    [
        ("plain", Groove(bpm=126.0, bars=48)),
        ("lead_in", Groove(bpm=126.0, bars=48, lead_in_seconds=1.9)),
        ("missing_kicks", Groove(bpm=126.0, bars=48, drop_kick_beats=(37, 38, 100))),
        ("syncopated", Groove(bpm=126.0, bars=48, syncopated_percussion=True)),
        ("drifting", Groove(bpm=125.0, bpm_end=127.0, bars=48)),
    ],
)
def test_downbeat_phase_is_correct(
    pipeline: AnalysisPipeline, tmp_path: Path, name: str, groove: Groove
) -> None:
    """
    Bar one must be bar one.

    A phase error is invisible in the BPM and ruins every bar jump, phrase
    boundary and 16-bar loop in the track, so it is checked against the
    synthesised downbeats directly.
    """
    render, result = analysed(pipeline, groove, tmp_path, name)
    assert result.rhythm.meter.beats_per_bar == 4
    assert result.rhythm.downbeats, "no downbeats detected"

    detected = np.array([downbeat.time for downbeat in result.rhythm.downbeats])
    distances = np.array([np.min(np.abs(detected - moment)) for moment in render.downbeat_times])
    assert float(np.median(distances)) < 0.05


def test_downbeats_reach_the_flat_field(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    """The field that used to be permanently null."""
    _, result = analysed(pipeline, Groove(bpm=126.0, bars=32), tmp_path, "flat")
    assert result.downbeats is not None
    assert len(result.downbeats) > 5


def test_beats_carry_bar_positions(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    _, result = analysed(pipeline, Groove(bpm=126.0, bars=32), tmp_path, "positions")
    beats = result.rhythm.beats
    assert all(beat.beat_in_bar in (1, 2, 3, 4) for beat in beats)
    # Indices are consecutive musical positions and bars advance every four.
    first_downbeat = next(beat for beat in beats if beat.beat_in_bar == 1 and beat.bar == 0)
    following = [beat for beat in beats if beat.index == first_downbeat.index + 4]
    assert following and following[0].bar == 1


def test_meter_is_reported_with_its_confidence(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    _, result = analysed(pipeline, Groove(bpm=126.0, bars=32), tmp_path, "meter")
    assert result.rhythm.meter.beats_per_bar == 4
    assert 0.0 <= result.rhythm.meter.confidence <= 1.0
    assert "3" in result.rhythm.meter.candidates  # 3/4 was evaluated, not assumed away


def test_noise_gets_no_bar_phase(pipeline: AnalysisPipeline, noise_wav: Path) -> None:
    """No meter, rather than a confident 4/4 over white noise."""
    result = pipeline.analyze(noise_wav)
    assert result.rhythm.meter.beats_per_bar is None or result.rhythm.meter.confidence < 0.5


# -- tempo curve and drift --------------------------------------------------


@pytest.mark.parametrize(
    ("name", "groove", "expected"),
    [
        ("steady", Groove(bpm=126.0, bars=48), DriftClassification.STABLE),
        (
            "ramp",
            Groove(bpm=125.0, bpm_end=127.0, bars=48),
            DriftClassification.VARIABLE_TEMPO,
        ),
        (
            "step",
            Groove(bpm=124.0, step_at_bar=24, step_bpm=128.0, bars=48),
            DriftClassification.HIGHLY_VARIABLE,
        ),
    ],
)
def test_drift_classification(
    pipeline: AnalysisPipeline,
    tmp_path: Path,
    name: str,
    groove: Groove,
    expected: DriftClassification,
) -> None:
    _, result = analysed(pipeline, groove, tmp_path, name)
    assert result.rhythm.drift.classification is expected
    assert result.rhythm.drift.relative_percent is not None
    assert result.rhythm.drift.local_bpm_min is not None


def test_tempo_curve_follows_a_ramp(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    _, result = analysed(
        pipeline, Groove(bpm=124.0, bpm_end=130.0, bars=64), tmp_path, "ramp_curve"
    )
    curve = result.rhythm.tempo_curve
    assert len(curve) >= 3
    assert curve[-1].bpm > curve[0].bpm + 2.0
    assert all(
        later.start_time >= earlier.start_time for earlier, later in itertools.pairwise(curve)
    )


def test_the_curve_is_quiet_on_a_steady_track() -> None:
    """A constant grid must not produce fake drift from estimator noise."""
    times = np.arange(400) * (60.0 / 126.0)
    curve = local_tempo_curve(times, musical_beat_indices(times))
    drift = classify_drift(curve, 126.0)
    assert drift.classification is DriftClassification.STABLE
    assert drift.relative_percent is not None and drift.relative_percent < 0.05


def test_drift_of_nothing_is_unknown() -> None:
    assert classify_drift([], 126.0).classification is DriftClassification.UNKNOWN


# -- grid confidence --------------------------------------------------------


def test_a_beatless_intro_is_reported_as_low_evidence(
    pipeline: AnalysisPipeline, tmp_path: Path
) -> None:
    render = render_groove(Groove(bpm=126.0, bars=40, lead_in_seconds=6.0))
    path = write_wav(tmp_path / "leadin.wav", render.samples)
    result = pipeline.analyze(path)

    weak = [region for region in result.rhythm.grid.regions if region.reason]
    assert weak, "a six-second silent lead-in should not claim full grid confidence"
    assert weak[0].confidence < result.rhythm.grid.confidence


def test_grid_regions_cover_the_track(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    _, result = analysed(pipeline, Groove(bpm=126.0, bars=32), tmp_path, "regions")
    regions = result.rhythm.grid.regions
    assert regions
    assert regions[0].start == pytest.approx(0.0, abs=0.5)
    for earlier, later in itertools.pairwise(regions):
        assert later.start >= earlier.start


# -- profiles ---------------------------------------------------------------


def test_the_basic_profile_skips_rhythm(tmp_path: Path) -> None:
    """A caller that only wants key and BPM should not pay for the rest."""
    render = render_groove(Groove(bpm=126.0, bars=32))
    path = write_wav(tmp_path / "basic.wav", render.samples)
    settings = Settings(
        key_engine="chroma",
        tempo_engine="chroma",
        profile=AnalysisProfile.BASIC,
        log_level="ERROR",
    )
    result = AnalysisPipeline(settings).analyze(path)

    assert result.tempo.bpm is not None
    assert result.rhythm.beats == []
    assert result.structure.boundaries == []
    assert result.warp is None


def test_the_warp_profile_adds_a_map(pipeline: AnalysisPipeline, tmp_path: Path) -> None:
    _, result = analysed(pipeline, Groove(bpm=126.0, bars=32), tmp_path, "warp_profile")
    assert result.warp is not None
    assert result.warp.recommendation.reason
