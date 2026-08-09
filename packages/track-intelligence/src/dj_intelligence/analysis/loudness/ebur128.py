"""
Loudness, measured to EBU R128 by ffmpeg.

Why shell out again rather than compute it from the samples we already have:
R128 is a *channel-weighted* standard. Integrated loudness sums K-weighted
channel powers with defined per-channel gains, and true peak is measured on
the oversampled signal. The decoder has already downmixed to mono and thrown
both of those away, so computing LUFS from its output would produce a number
that looks like LUFS, is close to LUFS, and is not LUFS. A second pass over
the original file gives the real measurement.

The cheap statistics -- sample peak and RMS -- do come from the decoded
signal, because they are exactly what they claim to be at any channel count.

Loudness is not used to decide anything yet. It is here because it is the
input every future "energy" feature will need, and because measuring it is
free next to the analysis that surrounds it.
"""

from __future__ import annotations

import re
import subprocess
from typing import Final

from ...audio.decoder import DecodedAudio
from ...audio.ffmpeg import resolve_tool, run_tool, tool_version
from ...models import EngineInfo, LoudnessMeasurement

__all__ = ["EbuR128LoudnessAnalyzer"]

# ffmpeg prints the summary to stderr at the end of the ebur128 filter run.
_INTEGRATED_RE: Final = re.compile(r"^\s*I:\s*(-?[\d.]+|-inf)\s*LUFS", re.MULTILINE)
_RANGE_RE: Final = re.compile(r"^\s*LRA:\s*(-?[\d.]+)\s*LU", re.MULTILINE)
_TRUE_PEAK_RE: Final = re.compile(r"^\s*Peak:\s*(-?[\d.]+|-inf)\s*dBFS", re.MULTILINE)

_TIMEOUT_FLOOR_SECONDS: Final = 120.0


def _first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(1)
    if value == "-inf":
        return None  # digital silence; None says that better than -infinity
    try:
        return float(value)
    except ValueError:
        return None


class EbuR128LoudnessAnalyzer:
    def __init__(
        self,
        *,
        ffmpeg_path: str = "ffmpeg",
        max_seconds: float | None = None,
    ) -> None:
        self._ffmpeg_path = ffmpeg_path
        self._max_seconds = max_seconds

    @property
    def name(self) -> str:
        return "ffmpeg"

    def describe(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            algorithm="ffmpeg ebur128 (EBU R128 / ITU-R BS.1770)",
            library_version=tool_version(self._ffmpeg_path),
            parameters={"peak": "true"},
        )

    def analyze(self, audio: DecodedAudio) -> LoudnessMeasurement:
        """
        Measure the source file. Returns whatever ffmpeg managed to produce;
        a failure here degrades to nulls rather than failing the analysis --
        nobody's track is unusable because a loudness number is missing.
        """
        sample_peak = audio.peak_dbfs
        rms = audio.rms_dbfs
        measurement = LoudnessMeasurement(
            sample_peak_dbfs=None if sample_peak == float("-inf") else round(sample_peak, 2),
            rms_dbfs=None if rms == float("-inf") else round(rms, 2),
        )

        try:
            executable = resolve_tool(self._ffmpeg_path)
        except Exception:
            return measurement

        args = ["-v", "info", "-i", str(audio.path)]
        if self._max_seconds and self._max_seconds > 0:
            args += ["-t", f"{self._max_seconds:.6f}"]
        args += ["-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"]

        try:
            completed = run_tool(
                executable,
                args,
                timeout=max(_TIMEOUT_FLOOR_SECONDS, audio.duration_seconds * 2.0),
            )
        except (OSError, subprocess.SubprocessError):
            return measurement

        report = completed.stderr.decode("utf-8", "replace")
        return measurement.model_copy(
            update={
                "integrated_lufs": _first_float(_INTEGRATED_RE, report),
                "loudness_range_lu": _first_float(_RANGE_RE, report),
                "true_peak_dbtp": _first_float(_TRUE_PEAK_RE, report),
            }
        )
