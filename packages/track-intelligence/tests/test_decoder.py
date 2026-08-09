"""
Ingest: decoding, normalisation, and refusing bad input.

These need ffmpeg on PATH. That is not a mock-able boundary in any useful
sense -- the whole value of the decoder is that ffmpeg accepts what we claim
it accepts, and a fake subprocess would test our beliefs rather than reality.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from dj_intelligence.audio.decoder import FFmpegDecoder
from dj_intelligence.audio.ffmpeg import redact_path, resolve_tool
from dj_intelligence.audio.hashing import sha256_file
from dj_intelligence.audio.probe import probe
from dj_intelligence.errors import (
    AudioDecodeError,
    AudioTooShortError,
    EmptyAudioError,
    ToolNotFoundError,
    UnsupportedFormatError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def decoder() -> FFmpegDecoder:
    return FFmpegDecoder()


def test_decodes_to_mono_float32(decoder: FFmpegDecoder, f_minor_wav: Path) -> None:
    audio = decoder.decode(f_minor_wav)
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.sample_rate == 44100
    assert audio.duration_seconds == pytest.approx(40.0, abs=0.05)
    assert np.all(np.isfinite(audio.samples))
    assert audio.peak <= 1.0


def test_records_what_the_source_was(decoder: FFmpegDecoder, f_minor_wav: Path) -> None:
    audio = decoder.decode(f_minor_wav)
    assert audio.source.codec == "pcm_s16le"
    assert audio.source.sample_rate == 44100
    assert audio.source.duration_seconds == pytest.approx(40.0, abs=0.05)


def test_codec_does_not_survive_ingest(
    decoder: FFmpegDecoder, f_minor_wav: Path, tmp_path: Path
) -> None:
    """The whole point of the decoder: analysis sees the same thing whatever
    the container was. Transcode to FLAC and MP3 and compare."""
    flac = tmp_path / "same.flac"
    mp3 = tmp_path / "same.mp3"
    ffmpeg = resolve_tool("ffmpeg")
    for target, args in ((flac, []), (mp3, ["-b:a", "320k"])):
        subprocess.run(
            [ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(f_minor_wav), *args, str(target)],
            check=True,
            stdin=subprocess.DEVNULL,
        )

    original = decoder.decode(f_minor_wav)
    lossless = decoder.decode(flac)
    lossy = decoder.decode(mp3)

    assert lossless.sample_rate == original.sample_rate == lossy.sample_rate
    # FLAC is lossless, so it must round-trip to the same samples.
    assert np.allclose(lossless.samples, original.samples, atol=1e-6)
    # MP3 is not, so only the shape and level are comparable.
    assert lossy.duration_seconds == pytest.approx(original.duration_seconds, abs=0.1)
    assert lossy.rms_dbfs == pytest.approx(original.rms_dbfs, abs=1.0)
    assert lossy.source.codec == "mp3"


def test_resampling_is_cached_and_correct(decoder: FFmpegDecoder, f_minor_wav: Path) -> None:
    audio = decoder.decode(f_minor_wav)
    half = audio.resampled(22050)
    assert half.dtype == np.float32
    assert len(half) == pytest.approx(len(audio.samples) / 2, rel=0.01)
    assert audio.resampled(22050) is half  # cached, not recomputed
    assert audio.resampled(audio.sample_rate) is audio.samples


def test_truncation_is_flagged(f_minor_wav: Path) -> None:
    decoder = FFmpegDecoder()
    audio = decoder.decode(f_minor_wav, max_seconds=5.0)
    assert audio.duration_seconds == pytest.approx(5.0, abs=0.05)
    assert audio.truncated is True
    assert audio.source.duration_seconds == pytest.approx(40.0, abs=0.05)


def test_windowing_is_a_view(decoder: FFmpegDecoder, f_minor_wav: Path) -> None:
    audio = decoder.decode(f_minor_wav)
    window = audio.window(10.0, 20.0)
    assert window.duration_seconds == pytest.approx(10.0, abs=0.01)
    assert window.samples.base is not None  # a view, not a copy
    assert window.features is not audio.features


# -- failure modes ----------------------------------------------------------


def test_corrupt_file_is_rejected(decoder: FFmpegDecoder, corrupt_file: Path) -> None:
    with pytest.raises(UnsupportedFormatError):
        decoder.decode(corrupt_file)


def test_error_text_does_not_leak_the_local_path(
    decoder: FFmpegDecoder, corrupt_file: Path
) -> None:
    """Uploads land on a server-side temp path; clients must not see it."""
    with pytest.raises(UnsupportedFormatError) as caught:
        decoder.decode(corrupt_file)
    assert str(corrupt_file.parent) not in str(caught.value)
    assert corrupt_file.name in str(caught.value)


def test_missing_file_is_rejected(decoder: FFmpegDecoder, tmp_path: Path) -> None:
    with pytest.raises(AudioDecodeError):
        decoder.decode(tmp_path / "nope.mp3")


def test_zero_length_audio_is_rejected(decoder: FFmpegDecoder, empty_wav: Path) -> None:
    with pytest.raises((EmptyAudioError, UnsupportedFormatError)):
        decoder.decode(empty_wav)


def test_extremely_short_audio_is_rejected(decoder: FFmpegDecoder, tiny_wav: Path) -> None:
    with pytest.raises(AudioTooShortError):
        decoder.decode(tiny_wav)


def test_a_file_with_no_audio_stream_is_rejected(decoder: FFmpegDecoder, tmp_path: Path) -> None:
    """Extension is not evidence: a text file named .mp3 is not an MP3."""
    liar = tmp_path / "text.mp3"
    liar.write_text("just some text, definitely not audio\n" * 100)
    with pytest.raises(UnsupportedFormatError):
        probe(liar)


def test_missing_tool_is_named_clearly() -> None:
    with pytest.raises(ToolNotFoundError, match="not found on PATH"):
        resolve_tool("definitely-not-a-real-tool-xyz")


def test_silence_decodes_but_measures_as_silent(decoder: FFmpegDecoder, silence_wav: Path) -> None:
    audio = decoder.decode(silence_wav)
    assert audio.samples.size > 0
    assert audio.rms_dbfs == float("-inf")
    assert audio.peak_dbfs == float("-inf")


# -- hashing ----------------------------------------------------------------


def test_hash_is_stable_and_content_addressed(f_minor_wav: Path, tmp_path: Path) -> None:
    first = sha256_file(f_minor_wav)
    assert first == sha256_file(f_minor_wav)
    assert len(first) == 64

    copy = tmp_path / "renamed.wav"
    copy.write_bytes(f_minor_wav.read_bytes())
    assert sha256_file(copy) == first  # the name is not part of the identity


def test_hash_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 999 + b"y")
    assert sha256_file(a) != sha256_file(b)


def test_redaction_removes_directories_but_keeps_the_name() -> None:
    path = Path("/var/folders/tmp/djti-abc123.mp3")
    message = f"Error opening {path}: Invalid data"
    redacted = redact_path(message, path)
    assert "/var/folders" not in redacted
    assert "djti-abc123.mp3" in redacted
