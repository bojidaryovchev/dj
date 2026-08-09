"""
Decoding and normalisation.

Everything upstream of this module knows about codecs; nothing downstream
does. The decoder is the seam: it takes any container ffmpeg can open and
produces one representation --

    mono, float32, 44.1 kHz by default

-- and every analyser is written against that. Adding FLAC support is an
ffmpeg question, not an analysis question.

Why these choices:

*mono*
    Key and tempo are properties of the arrangement, not the stereo image,
    and every MIR algorithm here wants one channel. Downmixing early halves
    the memory and the work.

*float32*
    Enough precision for analysis, half the footprint of float64, and the
    native input type for both backends.

*44.1 kHz*
    Keeps the whole audible band. Analysers that only need a fraction of it
    (chroma cares about nothing above ~5 kHz) resample down themselves via
    :meth:`DecodedAudio.resampled`, so the decision to throw information away
    is made by the code that knows it can afford to.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import AudioDecodeError, AudioTooShortError, EmptyAudioError
from .ffmpeg import redact_path, resolve_tool, run_tool
from .probe import SourceInfo, probe

__all__ = ["DecodedAudio", "FFmpegDecoder"]

_BYTES_PER_SAMPLE = 4  # float32
_MIN_DECODE_TIMEOUT_SECONDS = 120.0


@dataclass(slots=True)
class DecodedAudio:
    """
    Normalised audio plus a memory of where it came from.

    Mutable only so the resample cache can be filled; the sample buffer
    itself is never rewritten in place.
    """

    samples: np.ndarray
    """float32, mono, shape (n,), nominally in [-1, 1]."""

    sample_rate: int
    source: SourceInfo
    path: Path
    truncated: bool = False
    """True when ``max_seconds`` cut the file short of its real duration."""

    _resampled: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    features: dict[str, object] = field(default_factory=dict, repr=False)
    """
    Scratch space for derived representations, keyed by the analyser.

    A chromagram costs far more than the key estimate computed from it, and
    the global key analyser and the segment analyser want the same one. Rather
    than have the pipeline know that -- and so know what a chromagram is --
    analysers memoise here under a key that includes their parameters. The
    cache lives and dies with the decoded audio, so there is no stale-state
    problem across files.
    """

    @property
    def duration_seconds(self) -> float:
        """Duration of the material actually decoded."""
        return len(self.samples) / self.sample_rate

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples, dtype=np.float64))))

    @property
    def rms_dbfs(self) -> float:
        rms = self.rms
        return -np.inf if rms <= 0 else float(20.0 * np.log10(rms))

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if self.samples.size else 0.0

    @property
    def peak_dbfs(self) -> float:
        peak = self.peak
        return -np.inf if peak <= 0 else float(20.0 * np.log10(peak))

    def resampled(self, sample_rate: int) -> np.ndarray:
        """
        The signal at another rate, computed once and cached.

        Chroma extraction runs perfectly well at 22.05 kHz and costs roughly
        half as much there, so this is the difference between a fast analyser
        and a wasteful one. The cache matters because the segment analyser
        asks for the same rate the global analyser just used.
        """
        if sample_rate == self.sample_rate:
            return self.samples
        # `is not None`, not truthiness: an ndarray in a boolean context
        # raises once it has more than one element.
        if (cached := self._resampled.get(sample_rate)) is not None:
            return cached

        import soxr  # local: keeps the import cost off `--help`

        converted = np.asarray(
            soxr.resample(self.samples, self.sample_rate, sample_rate, quality="HQ"),
            dtype=np.float32,
        )
        self._resampled[sample_rate] = converted
        return converted

    def slice_seconds(self, start: float, end: float) -> np.ndarray:
        """Samples between two timestamps, clamped to the signal."""
        first = max(0, round(start * self.sample_rate))
        last = min(len(self.samples), round(end * self.sample_rate))
        return self.samples[first:last] if last > first else self.samples[:0]

    def window(self, start: float, end: float) -> DecodedAudio:
        """
        A stretch of this audio as its own :class:`DecodedAudio`.

        The samples are a numpy view, so windowing a track is free; only the
        feature cache is fresh, which is what we want -- a window's chromagram
        is not the track's. Lets any analyser be run over part of a track
        without a second interface for "analyse this slice".
        """
        return DecodedAudio(
            samples=self.slice_seconds(start, end),
            sample_rate=self.sample_rate,
            source=self.source,
            path=self.path,
            truncated=self.truncated,
        )


class FFmpegDecoder:
    """
    Decoder backed by an ffmpeg subprocess.

    An ffmpeg pipe rather than a Python decoding library because it is the
    only realistic way to accept everything a DJ's folder contains --
    MP3, WAV, FLAC, M4A/AAC, OGG, Opus, AIFF, WMA -- without collecting a
    library per codec, and because the resampler is better than most.
    """

    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        sample_rate: int = 44100,
        min_duration_seconds: float = 1.0,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._ffprobe_path = ffprobe_path
        self.sample_rate = sample_rate
        self._min_duration_seconds = min_duration_seconds

    def decode(self, path: Path | str, *, max_seconds: float | None = None) -> DecodedAudio:
        """
        Decode a file to mono float32.

        Raises :class:`AudioIngestError` subclasses for anything that means
        "there is nothing here to analyse": unreadable container, no audio
        stream, zero samples, or too short to be worth it.
        """
        source_path = Path(path)
        if not source_path.is_file():
            raise AudioDecodeError(f"not a file: {source_path}")

        info = probe(source_path, self._ffprobe_path)
        executable = resolve_tool(self._ffmpeg_path)

        args = ["-v", "error", "-i", str(source_path)]
        if max_seconds and max_seconds > 0:
            # Before -i would seek the input; after it, it limits output
            # duration, which is what "analyse the first N seconds" means.
            args += ["-t", f"{max_seconds:.6f}"]
        args += [
            "-map",
            "0:a:0",  # first audio stream only; ignore cover art
            "-vn",
            "-ac",
            "1",  # downmix
            "-ar",
            str(self.sample_rate),
            "-f",
            "f32le",  # raw little-endian float32 on stdout
            "-acodec",
            "pcm_f32le",
            "-",
        ]

        try:
            completed = run_tool(
                executable,
                args,
                timeout=self._decode_timeout(info.duration_seconds),
                capture_stdout=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioDecodeError(f"ffmpeg timed out decoding {source_path.name}") from exc
        except OSError as exc:
            raise AudioDecodeError(f"could not run ffmpeg: {exc}") from exc

        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
            reason = (
                redact_path(detail[-1], source_path) if detail else f"exit {completed.returncode}"
            )
            raise AudioDecodeError(f"ffmpeg failed on {source_path.name}: {reason}")

        raw = completed.stdout
        usable = len(raw) - (len(raw) % _BYTES_PER_SAMPLE)
        samples = np.frombuffer(raw[:usable], dtype="<f4")

        if samples.size == 0:
            raise EmptyAudioError(f"{source_path.name} decoded to zero samples")

        # A decoder that emits NaN or +/-inf (some broken MP3s do) would
        # poison every downstream statistic silently. Clean it here, once.
        if not np.all(np.isfinite(samples)):
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

        duration = samples.size / self.sample_rate
        if duration < self._min_duration_seconds:
            raise AudioTooShortError(
                f"{source_path.name} is {duration:.3f}s; "
                f"at least {self._min_duration_seconds:g}s is needed to analyse"
            )

        truncated = bool(
            max_seconds and info.duration_seconds and info.duration_seconds > max_seconds + 0.5
        )
        return DecodedAudio(
            samples=np.ascontiguousarray(samples, dtype=np.float32),
            sample_rate=self.sample_rate,
            source=info,
            path=source_path,
            truncated=truncated,
        )

    @staticmethod
    def _decode_timeout(duration_seconds: float | None) -> float:
        """Generous but finite. A corrupt file that makes ffmpeg spin should
        fail the request, not pin a worker forever."""
        if not duration_seconds:
            return 600.0
        return max(_MIN_DECODE_TIMEOUT_SECONDS, duration_seconds * 2.0)
