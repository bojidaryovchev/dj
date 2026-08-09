"""Time-windowed key analysis."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from dj_intelligence.analysis.key.chroma import ChromaKeyAnalyzer
from dj_intelligence.analysis.key.segmentation import SlidingWindowKeyAnalyzer
from dj_intelligence.analysis.pipeline import AnalysisPipeline
from dj_intelligence.audio.decoder import FFmpegDecoder
from dj_intelligence.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def decoder() -> FFmpegDecoder:
    return FFmpegDecoder()


def test_a_modulation_is_found(decoder: FFmpegDecoder, modulating_wav: Path) -> None:
    """45 s of A minor then 45 s of F minor must not read as one key."""
    analyzer = SlidingWindowKeyAnalyzer(
        ChromaKeyAnalyzer(), window_seconds=20.0, hop_seconds=10.0, min_confidence=0.3
    )
    segments = analyzer.analyze(decoder.decode(modulating_wav))

    labelled = [s for s in segments if s.reliable]
    assert len(labelled) >= 2
    assert labelled[0].key == "A"
    assert labelled[-1].key == "F"
    # The change is found near where it actually is, not at either end.
    boundary = labelled[0].end_seconds
    assert 35.0 < boundary < 55.0


def test_segments_tile_the_timeline(decoder: FFmpegDecoder, modulating_wav: Path) -> None:
    """No gaps, no overlaps: consecutive segments meet exactly."""
    audio = decoder.decode(modulating_wav)
    analyzer = SlidingWindowKeyAnalyzer(ChromaKeyAnalyzer(), window_seconds=20.0, hop_seconds=10.0)
    segments = analyzer.analyze(audio)

    assert segments[0].start_seconds == 0.0
    assert segments[-1].end_seconds == pytest.approx(audio.duration_seconds, abs=0.05)
    for earlier, later in itertools.pairwise(segments):
        assert later.start_seconds == pytest.approx(earlier.end_seconds, abs=1e-6)
        assert later.end_seconds > later.start_seconds


def test_a_single_key_track_is_one_segment(decoder: FFmpegDecoder, f_minor_wav: Path) -> None:
    analyzer = SlidingWindowKeyAnalyzer(ChromaKeyAnalyzer(), window_seconds=15.0, hop_seconds=7.5)
    segments = analyzer.analyze(decoder.decode(f_minor_wav))
    assert len(segments) == 1
    assert segments[0].key == "F"


def test_a_track_shorter_than_the_window_yields_nothing(
    decoder: FFmpegDecoder, f_minor_wav: Path
) -> None:
    analyzer = SlidingWindowKeyAnalyzer(ChromaKeyAnalyzer(), window_seconds=300.0)
    assert analyzer.analyze(decoder.decode(f_minor_wav)) == []


def test_the_chroma_fast_path_agrees_with_slicing(
    decoder: FFmpegDecoder, modulating_wav: Path
) -> None:
    """
    The optimisation must not change the answer.

    Segmentation reuses one chromagram for the whole track when the backend
    exposes it, and slices audio otherwise. Both routes have to agree, or the
    fast path is a silent accuracy regression.
    """
    audio = decoder.decode(modulating_wav)
    analyzer = ChromaKeyAnalyzer()
    segmenter = SlidingWindowKeyAnalyzer(analyzer, window_seconds=20.0, hop_seconds=10.0)

    fast = segmenter._windows_via_chroma(audio, audio.duration_seconds)
    audio.features.clear()
    slow = segmenter._windows_via_slices(audio, audio.duration_seconds)

    assert [w.label for w in fast] == [w.label for w in slow]


def test_the_chromagram_is_computed_once_per_track(
    decoder: FFmpegDecoder, f_minor_wav: Path
) -> None:
    audio = decoder.decode(f_minor_wav)
    analyzer = ChromaKeyAnalyzer()
    first, rate = analyzer.chromagram(audio)
    again, rate_again = analyzer.chromagram(audio)
    assert first is again and rate == rate_again


def test_segments_reach_the_dj_layer(modulating_wav: Path) -> None:
    settings = Settings(
        key_engine="chroma",
        tempo_engine="chroma",
        segment_window_seconds=20.0,
        segment_hop_seconds=10.0,
        log_level="ERROR",
    )
    result = AnalysisPipeline(settings).analyze(modulating_wav)

    assert len(result.tonal_segments) >= 2
    assert len(result.dj.segments) == len(result.tonal_segments)
    labelled = [s for s in result.dj.segments if s.reliable]
    assert {s.camelot for s in labelled} == {"8A", "4A"}
    # Every labelled segment says how it sits against the global key.
    assert all(s.relationship_to_global for s in labelled)


def test_segmentation_can_be_disabled(f_minor_wav: Path) -> None:
    settings = Settings(
        key_engine="chroma", tempo_engine="chroma", segments_enabled=False, log_level="ERROR"
    )
    result = AnalysisPipeline(settings).analyze(f_minor_wav)
    assert result.tonal_segments == []


def test_window_and_hop_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        SlidingWindowKeyAnalyzer(ChromaKeyAnalyzer(), window_seconds=0.0)
