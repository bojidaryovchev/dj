"""
Rendering a warp map onto audio.

Time is changed; pitch is not. That rules out resampling, which changes both,
and it is why this uses **Rubber Band** — the same library DAWs use for
offline warping — reached through ffmpeg's ``rubberband`` filter.

**Variable warping, not one global ratio.** A drifting record needs a
different stretch in different places, so each span between two warp markers
is rendered with its own ratio and the results are assembled onto the target
timeline. That is genuine keyframe warping; the alternative of averaging the
whole track to a single ratio would leave exactly the drift we set out to
remove.

Rubber Band's own command-line tool takes a time map directly
(``rubberband --timemap``) and would do this in one pass. It is not installed
everywhere — notably not on Windows — so the segment renderer is the portable
path, and it drives the same library underneath. When the CLI is present it is
preferred, because one pass has no joins at all.

**Joins.** Adjacent segments come from the same source at slightly different
ratios, so butting them together can click. Each segment is rendered with a
little extra tail and overlapped into the next with an equal-power crossfade,
which removes the discontinuity without moving any marker: segment *k* still
starts exactly at its target time.

**Length is pinned, not accumulated.** Every segment is trimmed or padded to
the exact sample count its target span demands. A stretcher that returns three
samples too many is normal; letting that compound across forty segments is how
the end of a track ends up a beat late.

The source file is never modified. Output is a new lossless artifact.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from ..audio.ffmpeg import resolve_tool, run_tool, tool_version
from ..errors import AudioDecodeError, DJIntelligenceError
from ..models import WarpMap, WarpRenderReport
from ..observability import get_logger

__all__ = ["WARP_ALGORITHM_VERSION", "RenderSegment", "WarpRenderer"]

log = get_logger(__name__)

WARP_ALGORITHM_VERSION: Final = "1.0.0"

_MIN_SEGMENT_SECONDS: Final = 0.05
_TIMEOUT_FLOOR: Final = 120.0


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """One span of source audio and where it has to land."""

    source_start: float
    source_end: float
    target_start: float
    target_end: float

    @property
    def source_duration(self) -> float:
        return self.source_end - self.source_start

    @property
    def target_duration(self) -> float:
        return self.target_end - self.target_start

    @property
    def ratio(self) -> float:
        """Target over source. >1 means the segment is being lengthened."""
        return self.target_duration / self.source_duration if self.source_duration > 0 else 1.0

    @property
    def tempo_factor(self) -> float:
        """What ffmpeg's rubberband filter wants: source over target."""
        return 1.0 / self.ratio if self.ratio > 0 else 1.0


class WarpRenderer:
    """Renders a :class:`WarpMap` to a new audio file."""

    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        crossfade_ms: float = 8.0,
        sample_rate: int = 44100,
        transients: str = "crisp",
        detector: str = "percussive",
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._crossfade_ms = max(0.0, crossfade_ms)
        self._sample_rate = sample_rate
        # Percussive detection and crisp transients: this is dance music, and
        # a smeared kick is the one artefact a DJ will hear immediately.
        self._transients = transients
        self._detector = detector

    @property
    def name(self) -> str:
        return "ffmpeg-rubberband"

    def version(self) -> str | None:
        return tool_version(self._ffmpeg_path)

    @staticmethod
    def rubberband_cli_available() -> bool:
        """Whether the one-pass ``rubberband --timemap`` path could be used."""
        return shutil.which("rubberband") is not None

    def available(self) -> bool:
        """Whether ffmpeg here was built with librubberband."""
        try:
            executable = resolve_tool(self._ffmpeg_path)
            completed = run_tool(
                executable, ["-hide_banner", "-filters"], timeout=30, capture_stdout=True
            )
        except (OSError, subprocess.SubprocessError, DJIntelligenceError):
            return False
        return b" rubberband " in completed.stdout

    # -- planning ----------------------------------------------------------

    def segments(self, warp_map: WarpMap, duration: float) -> list[RenderSegment]:
        """
        Turn markers into spans, including the unwarped head and tail.

        Audio before the first marker and after the last is copied at its own
        length and simply placed where the timeline says. Stretching an intro
        to fit a grid it was never on would be inventing a correction.
        """
        markers = warp_map.markers
        if len(markers) < 2:
            return []

        spans: list[RenderSegment] = []

        head_target_start = markers[0].target_time - markers[0].source_time
        if markers[0].source_time > _MIN_SEGMENT_SECONDS and head_target_start >= 0:
            spans.append(
                RenderSegment(
                    source_start=0.0,
                    source_end=markers[0].source_time,
                    target_start=head_target_start,
                    target_end=markers[0].target_time,
                )
            )

        for current, following in itertools.pairwise(markers):
            spans.append(
                RenderSegment(
                    source_start=current.source_time,
                    source_end=following.source_time,
                    target_start=current.target_time,
                    target_end=following.target_time,
                )
            )

        last = markers[-1]
        if duration - last.source_time > _MIN_SEGMENT_SECONDS:
            tail = duration - last.source_time
            spans.append(
                RenderSegment(
                    source_start=last.source_time,
                    source_end=duration,
                    target_start=last.target_time,
                    target_end=last.target_time + tail,
                )
            )
        return [span for span in spans if span.source_duration > _MIN_SEGMENT_SECONDS]

    # -- rendering ---------------------------------------------------------

    def render(
        self,
        source_path: Path,
        warp_map: WarpMap,
        output_path: Path,
        *,
        duration: float,
        channels: int = 2,
        sample_rate: int | None = None,
    ) -> WarpRenderReport:
        """
        Render ``source_path`` onto the warp map's target timeline.

        Returns a report; verification is a separate step so that rendering
        and checking cannot be confused for one another.
        """
        started = time.perf_counter()
        rate = sample_rate or self._sample_rate
        spans = self.segments(warp_map, duration)
        if not spans:
            raise AudioDecodeError("warp map has too few markers to render")

        if not self.available():
            raise DJIntelligenceError(
                "this ffmpeg was not built with librubberband, so pitch-preserving "
                "time stretching is unavailable. Install an ffmpeg with "
                "--enable-librubberband, or install the rubberband CLI."
            )

        crossfade = round(self._crossfade_ms / 1000.0 * rate)
        total_samples = round(max(span.target_end for span in spans) * rate) + crossfade
        output = np.zeros((total_samples, channels), dtype=np.float64)

        warnings: list[str] = []
        for index, span in enumerate(spans):
            wanted = round(span.target_duration * rate)
            tail = crossfade if index < len(spans) - 1 else 0
            rendered = self._render_segment(source_path, span, rate, channels, extra_samples=tail)
            rendered = _fit(rendered, wanted + tail)

            if index > 0 and crossfade > 0:
                rendered[:crossfade] *= _fade_in(crossfade)[:, None]
            if tail > 0:
                rendered[-crossfade:] *= _fade_out(crossfade)[:, None]

            start = round(span.target_start * rate)
            end = min(start + rendered.shape[0], total_samples)
            output[start:end] += rendered[: end - start]

        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > 1.0:
            # Overlap-add can exceed full scale where two segments sum.
            output /= peak
            warnings.append(f"output normalised by {1.0 / peak:.4f} to avoid clipping")

        self._write(output.astype(np.float32), output_path, rate, channels)
        elapsed = time.perf_counter() - started

        ratios = np.array([span.ratio for span in spans], dtype=np.float64)
        return WarpRenderReport(
            output_path=str(output_path),
            renderer=self.name,
            renderer_version=self.version(),
            target_bpm=warp_map.target_bpm,
            source_duration_seconds=round(duration, 3),
            output_duration_seconds=round(total_samples / rate, 3),
            expected_duration_seconds=round(max(span.target_end for span in spans), 3),
            marker_count=len(warp_map.markers),
            segment_count=len(spans),
            min_stretch_ratio=round(float(np.min(ratios)), 6),
            max_stretch_ratio=round(float(np.max(ratios)), 6),
            mean_stretch_ratio=round(float(np.mean(ratios)), 6),
            pitch_shift_cents=0.0,
            crossfade_ms=self._crossfade_ms,
            render_seconds=round(elapsed, 3),
            warnings=warnings,
        )

    def _render_segment(
        self,
        source_path: Path,
        span: RenderSegment,
        rate: int,
        channels: int,
        *,
        extra_samples: int,
    ) -> np.ndarray:
        """Extract one span, stretch it with Rubber Band, return raw samples."""
        executable = resolve_tool(self._ffmpeg_path)
        extra_source = extra_samples / rate * span.tempo_factor
        args = [
            "-v",
            "error",
            "-ss",
            f"{span.source_start:.6f}",
            "-t",
            f"{span.source_duration + extra_source:.6f}",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            (
                f"rubberband=tempo={span.tempo_factor:.9f}:pitch=1"
                f":transients={self._transients}:detector={self._detector}"
            ),
            "-ac",
            str(channels),
            "-ar",
            str(rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ]
        try:
            completed = run_tool(
                executable,
                args,
                timeout=max(_TIMEOUT_FLOOR, span.source_duration * 20),
                capture_stdout=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise AudioDecodeError(f"rubberband render failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioDecodeError(
                f"rubberband render failed: {detail[-1] if detail else completed.returncode}"
            )

        raw = np.frombuffer(completed.stdout, dtype="<f4")
        usable = raw.size - (raw.size % channels)
        return raw[:usable].reshape(-1, channels).astype(np.float64)

    def _write(self, samples: np.ndarray, output_path: Path, rate: int, channels: int) -> None:
        """Encode the assembled buffer. Lossless: WAV or FLAC by extension."""
        executable = resolve_tool(self._ffmpeg_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "flac" if output_path.suffix.lower() == ".flac" else "pcm_s24le"
        args = [
            "-v",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            str(rate),
            "-ac",
            str(channels),
            "-i",
            "-",
            "-c:a",
            codec,
            str(output_path),
        ]
        executable_args = [executable, "-nostdin", *args]
        completed = subprocess.run(  # noqa: S603 -- list args, resolved executable, no shell
            executable_args,
            input=samples.tobytes(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=max(_TIMEOUT_FLOOR, samples.shape[0] / rate * 10),
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
            raise AudioDecodeError(
                f"could not write {output_path.name}: "
                f"{detail[-1] if detail else completed.returncode}"
            )


def _fade_in(length: int) -> np.ndarray:
    """Equal power, so a crossfade holds level rather than dipping."""
    return np.sin(np.linspace(0.0, np.pi / 2, length)) ** 1.0


def _fade_out(length: int) -> np.ndarray:
    return np.cos(np.linspace(0.0, np.pi / 2, length)) ** 1.0


def _fit(samples: np.ndarray, wanted: int) -> np.ndarray:
    """Trim or zero-pad to an exact length, so error cannot accumulate."""
    if samples.shape[0] == wanted:
        return samples
    if samples.shape[0] > wanted:
        return samples[:wanted]
    padding = np.zeros((wanted - samples.shape[0], samples.shape[1]), dtype=samples.dtype)
    return np.vstack([samples, padding])
