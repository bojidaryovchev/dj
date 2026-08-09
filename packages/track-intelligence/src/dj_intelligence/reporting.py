"""
Terminal presentation.

Split out of ``cli.py`` once the command surface grew past analysis: the CLI
now has an analysis report, a rhythmic grid report and a warp report, and
mixing three page layouts in with argument parsing made both harder to read.
Nothing here computes anything — every number arrives already measured.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table
from rich.text import Text

from .models import TrackAnalysis, WarpRenderReport
from .version import ANALYSIS_VERSION

__all__ = [
    "ACCENT",
    "MUTED",
    "Glyph",
    "clock",
    "console",
    "duration",
    "field",
    "render_analysis",
    "render_beats",
    "render_grid",
    "render_warp",
    "rule",
    "stderr_console",
]

console = Console()
stderr_console = Console(stderr=True)

ACCENT = "bright_cyan"
MUTED = "grey62"


def _encodable(text: str) -> bool:
    encoding = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


class Glyph:
    """
    Box-drawing characters, downgraded when the console cannot encode them.

    A Windows console still defaults to cp1252, where writing U+2500 raises
    ``UnicodeEncodeError`` and takes the whole command down. Printing a report
    is not worth crashing over, so the decoration adapts to the terminal
    rather than assuming UTF-8.
    """

    _RICH = _encodable("─·→—✓✗")

    RULE = "─" if _RICH else "-"
    DOT = "·" if _RICH else "*"
    ARROW = "→" if _RICH else "->"
    DASH = "—" if _RICH else "-"
    RANGE = "–" if _RICH else "-"  # noqa: RUF001 -- an en dash is the right glyph for a range
    YES = "✓" if _RICH else "+"
    NO = "✗" if _RICH else "x"


def rule() -> str:
    return Glyph.RULE * 41


def duration(seconds: float) -> str:
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}:{remainder:02d}"


def clock(seconds: float) -> str:
    """``mm:ss.mmm`` -- grid positions need better than whole seconds."""
    minutes, remainder = divmod(seconds, 60.0)
    return f"{int(minutes):02d}:{remainder:06.3f}"


def field(label: str, value: object, *, note: str | None = None) -> None:
    console.print(Text(label, style=MUTED))
    body = Text(str(value))
    if note:
        body.append(f"  {note}", style=MUTED)
    console.print(body)
    console.print()


def _heading(title: str) -> None:
    console.print(Text(rule(), style=ACCENT))
    console.print(Text(f" {title}", style=f"bold {ACCENT}"))
    console.print(Text(rule(), style=ACCENT))
    console.print()


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


def render_analysis(result: TrackAnalysis) -> None:
    _heading("DJ TRACK ANALYSIS")

    field("File", result.track.filename)
    field(
        "Duration",
        duration(result.audio.duration_seconds),
        note=(
            f"analysed {duration(result.audio.analysed_seconds)}"
            if result.audio.analysed_seconds < result.audio.duration_seconds - 0.5
            else None
        ),
    )

    tempo = result.tempo
    if tempo.bpm is None:
        field("Tempo", "not determined")
    else:
        note = None
        if result.dj.mix_bpm and abs(result.dj.mix_bpm - tempo.bpm) > 0.01:
            note = f"{Glyph.ARROW} mix at {result.dj.mix_bpm:.2f} ({result.dj.mix_bpm_relation})"
        elif tempo.stable is False:
            note = "unstable grid"
        field("Tempo", f"{tempo.bpm:.2f} BPM", note=note)

    if result.tonality.key is None:
        field("Key", "not determined")
    else:
        mode = result.tonality.mode.value if result.tonality.mode else ""
        field(
            "Key",
            f"{result.tonality.key} {mode}",
            note=None if result.tonality.reliable else "low confidence",
        )
        field("Camelot", result.dj.camelot or Glyph.DASH)

    confidence = Table.grid(padding=(0, 2))
    confidence.add_column(style=MUTED)
    confidence.add_column()
    confidence.add_row(
        "tonal", f"{result.tonality.confidence:.2f}  ({result.tonality.confidence_type})"
    )
    confidence.add_row("tempo", f"{tempo.confidence:.2f}  ({tempo.confidence_type})")
    console.print(Text("Confidence", style=MUTED))
    console.print(confidence)
    console.print()

    _render_rhythm_summary(result)

    if result.dj.compatible_keys:
        neighbours = Table.grid(padding=(0, 2))
        neighbours.add_column()
        neighbours.add_column(style=MUTED)
        for entry in result.dj.compatible_keys:
            neighbours.add_row(
                entry.camelot, f"{entry.key} {entry.mode}  {Glyph.DOT}  {entry.relationship}"
            )
        console.print(Text("Recommended harmonic neighbours", style=MUTED))
        console.print(neighbours)
        console.print()

    reliable_segments = [segment for segment in result.dj.segments if segment.reliable]
    if len(reliable_segments) > 1:
        segments = Table.grid(padding=(0, 2))
        segments.add_column(style=MUTED)
        segments.add_column()
        segments.add_column(style=MUTED)
        for segment in result.dj.segments:
            segments.add_row(
                f"{duration(segment.start_seconds)}{Glyph.RANGE}{duration(segment.end_seconds)}",
                segment.camelot or Glyph.DASH,
                segment.relationship_to_global or ("unclear" if not segment.reliable else ""),
            )
        console.print(Text("Tonal segments", style=MUTED))
        console.print(segments)
        console.print()

    if result.loudness.integrated_lufs is not None:
        field("Loudness", f"{result.loudness.integrated_lufs:.1f} LUFS integrated")

    engine = result.analysis.key_engine
    field(
        "Analysis engine",
        f"{engine.name if engine else 'unknown'}  {Glyph.DOT}  v{ANALYSIS_VERSION}",
    )
    field(
        "Processing time",
        f"{result.analysis.processing_time_ms / 1000:.2f} s",
        note=(
            f"{result.analysis.realtime_ratio:.4f}x real-time"
            if result.analysis.realtime_ratio
            else None
        ),
    )

    _render_warnings(result)
    console.print(Text(rule(), style=ACCENT))


def _render_rhythm_summary(result: TrackAnalysis) -> None:
    rhythm = result.rhythm
    if not rhythm.grid.bar_count:
        return

    meter = rhythm.meter.beats_per_bar
    field(
        "Bars",
        str(rhythm.grid.bar_count),
        note=(
            f"{meter}/4  {Glyph.DOT}  grid {rhythm.grid.confidence:.0%}"
            if meter
            else "meter not determined"
        ),
    )
    if rhythm.grid.first_downbeat_time is not None:
        field("First downbeat", clock(rhythm.grid.first_downbeat_time))
    if rhythm.drift.classification.value != "unknown":
        field(
            "Tempo drift",
            rhythm.drift.classification.value.replace("_", " "),
            note=(
                f"{rhythm.drift.local_bpm_min:.2f}{Glyph.RANGE}{rhythm.drift.local_bpm_max:.2f} BPM"
                if rhythm.drift.local_bpm_min is not None
                else None
            ),
        )
    if result.warp is not None:
        field(
            "Warp recommended",
            "YES" if result.warp.recommendation.required else "no",
            note=(
                f"{result.warp.metrics.marker_count} markers"
                if result.warp.recommendation.required
                else None
            ),
        )


def _render_warnings(result: TrackAnalysis) -> None:
    if not result.warnings:
        return
    console.print(Text("Warnings", style="yellow"))
    for warning in result.warnings:
        console.print(Text(f"  {warning.code}: {warning.message}", style=MUTED))
    console.print()


# --------------------------------------------------------------------------
# rhythmic grid
# --------------------------------------------------------------------------


def render_grid(result: TrackAnalysis) -> None:
    rhythm = result.rhythm
    _heading("RHYTHMIC GRID")

    field("File", result.track.filename)
    field("Nominal BPM", f"{result.tempo.bpm:.2f}" if result.tempo.bpm else "not determined")

    drift = rhythm.drift
    if drift.classification.value != "unknown":
        field(
            "Tempo",
            drift.classification.value.replace("_", " ").capitalize(),
            note=f"range {drift.relative_percent:.3f}%"
            if drift.relative_percent is not None
            else None,
        )
        if drift.local_bpm_min is not None and drift.local_bpm_max is not None:
            field(
                "Tempo range", f"{drift.local_bpm_min:.2f} {Glyph.RANGE} {drift.local_bpm_max:.2f}"
            )

    field("Beats", str(len(rhythm.beats)))
    field("Bars", str(rhythm.grid.bar_count))
    field(
        "Meter",
        f"{rhythm.meter.beats_per_bar}/4" if rhythm.meter.beats_per_bar else "not determined",
        note=f"confidence {rhythm.meter.confidence:.0%}" if rhythm.meter.beats_per_bar else None,
    )
    if rhythm.grid.first_downbeat_time is not None:
        field("First reliable downbeat", clock(rhythm.grid.first_downbeat_time))
    field("Grid confidence", f"{rhythm.grid.confidence:.0%}")

    weak = [region for region in rhythm.grid.regions if region.reason]
    if weak:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED)
        table.add_column()
        for region in weak:
            table.add_row(
                f"{clock(region.start)}{Glyph.RANGE}{clock(region.end)}",
                f"{region.reason}  ({region.confidence:.0%})",
            )
        console.print(Text("Low-evidence regions", style=MUTED))
        console.print(table)
        console.print()

    if result.structure.boundaries:
        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED)
        table.add_column()
        for boundary in result.structure.boundaries:
            table.add_row(
                clock(boundary.time),
                f"bar {boundary.bar}" if boundary.bar is not None else Glyph.DASH,
            )
        console.print(
            Text(f"Structural boundaries ({len(result.structure.boundaries)})", style=MUTED)
        )
        console.print(table)
        console.print()

    if result.structure.phrase_grid:
        field(
            "Phrase grid",
            f"{len(result.structure.phrase_grid)} x {result.structure.phrase_bars} bars",
        )

    if result.warp is not None:
        field("Warp recommended", "YES" if result.warp.recommendation.required else "no")
        field("Estimated markers", str(result.warp.metrics.marker_count))
        if result.warp.recommendation.reason:
            console.print(Text(f"  {result.warp.recommendation.reason}", style=MUTED))
            console.print()

    console.print(Text(rule(), style=ACCENT))


def render_beats(result: TrackAnalysis) -> None:
    table = Table.grid(padding=(0, 2))
    for _ in range(4):
        table.add_column(style=MUTED)
    for beat in result.rhythm.beats:
        table.add_row(
            str(beat.index),
            clock(beat.time),
            f"bar {beat.bar}" if beat.bar is not None else Glyph.DASH,
            f"beat {beat.beat_in_bar}" if beat.beat_in_bar is not None else Glyph.DASH,
        )
    console.print(table)


# --------------------------------------------------------------------------
# warp
# --------------------------------------------------------------------------


def render_warp(
    result: TrackAnalysis,
    *,
    rendered: bool,
    report: WarpRenderReport | None,
    skipped_reason: str | None,
) -> None:
    _heading("WARP")

    field("File", result.track.filename)
    field("Input BPM", f"{result.tempo.bpm:.2f}" if result.tempo.bpm else "unknown")

    if result.warp is not None:
        field("Target BPM", f"{result.warp.target_bpm:.2f}")
        field(
            "Tempo drift",
            f"{result.rhythm.drift.relative_percent:.2f}%"
            if result.rhythm.drift.relative_percent is not None
            else "unknown",
            note=result.rhythm.drift.classification.value.replace("_", " "),
        )
        field("Warp markers", str(result.warp.metrics.marker_count))
        field("Max correction", f"{result.warp.metrics.systematic_error_ms:.0f} ms")

    if not rendered or report is None:
        console.print(Text("Not rendered", style="yellow"))
        if skipped_reason:
            console.print(Text(f"  {skipped_reason}", style=MUTED))
        console.print()
        console.print(Text(rule(), style=ACCENT))
        return

    field(
        "Average stretch ratio",
        f"{report.mean_stretch_ratio:.5f}",
        note=f"{report.min_stretch_ratio:.5f}{Glyph.RANGE}{report.max_stretch_ratio:.5f}",
    )
    field("Pitch shift", f"{report.pitch_shift_cents:.0f} cents")
    field("Renderer", f"{report.renderer}  {Glyph.DOT}  {report.segment_count} segments")

    if report.verification is not None:
        verification = report.verification
        table = Table.grid(padding=(0, 2))
        table.add_column(style=MUTED)
        table.add_column()
        if verification.source_mean_grid_error_ms is not None:
            table.add_row("before", f"{verification.source_mean_grid_error_ms:.1f} ms mean")
        table.add_row("after", f"{verification.mean_grid_error_ms:.1f} ms mean")
        table.add_row("p95", f"{verification.p95_grid_error_ms:.1f} ms")
        table.add_row("max", f"{verification.max_grid_error_ms:.1f} ms")
        if verification.improvement_factor:
            table.add_row("improvement", f"{verification.improvement_factor:.1f}x")
        table.add_row(
            "result",
            Text(f"{Glyph.YES} PASSED", style="green")
            if verification.passed
            else Text(f"{Glyph.NO} FAILED", style="red"),
        )
        console.print(Text("Verification", style=MUTED))
        console.print(table)
        console.print()

    field("Output", report.output_path)
    for warning in report.warnings:
        console.print(Text(f"  {warning}", style="yellow"))
    console.print(Text(rule(), style=ACCENT))
