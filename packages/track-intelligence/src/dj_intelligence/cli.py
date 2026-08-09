"""
The ``dj-analyze`` command.

A presentation layer and nothing more. Every number it prints comes from
:func:`dj_intelligence.engine.analyze`, which is the same call the HTTP API
makes -- there is no second analysis path to drift out of step.

    dj-analyze track.mp3            human-readable report
    dj-analyze track.mp3 --json     the canonical document, for piping
    dj-analyze keys 4A              harmonic neighbours of a Camelot key
    dj-analyze compat 4A 126 5A 126.5
    dj-analyze engines              which backends this install can run
    dj-analyze serve                start the HTTP API
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config import EngineChoice, KeyProfile, Settings, get_settings
from .errors import (
    AudioIngestError,
    BackendUnavailableError,
    DJIntelligenceError,
    ToolNotFoundError,
)
from .models import TrackAnalysis
from .models.compatibility import TrackReference
from .music.camelot import CamelotKey
from .music.notes import InvalidKeyError
from .observability import configure_logging
from .version import ANALYSIS_VERSION

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Analyse audio for DJ-relevant musical properties.",
)

_COMMANDS = {"analyze", "keys", "compat", "engines", "serve"}

console = Console()
stderr_console = Console(stderr=True)

_ACCENT = "bright_cyan"
_MUTED = "grey62"


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


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _duration(seconds: float) -> str:
    minutes, remainder = divmod(round(seconds), 60)
    return f"{minutes}:{remainder:02d}"


def _field(label: str, value: object, *, note: str | None = None) -> None:
    console.print(Text(label, style=_MUTED))
    body = Text(str(value))
    if note:
        body.append(f"  {note}", style=_MUTED)
    console.print(body)
    console.print()


def _render(result: TrackAnalysis) -> None:
    rule = Glyph.RULE * 41
    console.print(Text(rule, style=_ACCENT))
    console.print(Text(" DJ TRACK ANALYSIS", style=f"bold {_ACCENT}"))
    console.print(Text(rule, style=_ACCENT))
    console.print()

    _field("File", result.track.filename)
    _field(
        "Duration",
        _duration(result.audio.duration_seconds),
        note=(
            f"analysed {_duration(result.audio.analysed_seconds)}"
            if result.audio.analysed_seconds < result.audio.duration_seconds - 0.5
            else None
        ),
    )

    tempo = result.tempo
    if tempo.bpm is None:
        _field("Tempo", "not determined")
    else:
        note = None
        if result.dj.mix_bpm and abs(result.dj.mix_bpm - tempo.bpm) > 0.01:
            note = f"{Glyph.ARROW} mix at {result.dj.mix_bpm:.2f} ({result.dj.mix_bpm_relation})"
        elif tempo.stable is False:
            note = "unstable grid"
        _field("Tempo", f"{tempo.bpm:.2f} BPM", note=note)

    if result.tonality.key is None:
        _field("Key", "not determined")
    else:
        mode = result.tonality.mode.value if result.tonality.mode else ""
        _field(
            "Key",
            f"{result.tonality.key} {mode}",
            note=None if result.tonality.reliable else "low confidence",
        )
        _field("Camelot", result.dj.camelot or Glyph.DASH)

    confidence = Table.grid(padding=(0, 2))
    confidence.add_column(style=_MUTED)
    confidence.add_column()
    confidence.add_row(
        "tonal", f"{result.tonality.confidence:.2f}  ({result.tonality.confidence_type})"
    )
    confidence.add_row("tempo", f"{tempo.confidence:.2f}  ({tempo.confidence_type})")
    console.print(Text("Confidence", style=_MUTED))
    console.print(confidence)
    console.print()

    if result.dj.compatible_keys:
        neighbours = Table.grid(padding=(0, 2))
        neighbours.add_column()
        neighbours.add_column(style=_MUTED)
        for entry in result.dj.compatible_keys:
            neighbours.add_row(
                entry.camelot, f"{entry.key} {entry.mode}  {Glyph.DOT}  {entry.relationship}"
            )
        console.print(Text("Recommended harmonic neighbours", style=_MUTED))
        console.print(neighbours)
        console.print()

    reliable_segments = [s for s in result.dj.segments if s.reliable]
    if len(reliable_segments) > 1:
        segments = Table.grid(padding=(0, 2))
        segments.add_column(style=_MUTED)
        segments.add_column()
        segments.add_column(style=_MUTED)
        for segment in result.dj.segments:
            segments.add_row(
                f"{_duration(segment.start_seconds)}{Glyph.RANGE}{_duration(segment.end_seconds)}",
                segment.camelot or Glyph.DASH,
                segment.relationship_to_global or ("unclear" if not segment.reliable else ""),
            )
        console.print(Text("Tonal segments", style=_MUTED))
        console.print(segments)
        console.print()

    if result.loudness.integrated_lufs is not None:
        _field("Loudness", f"{result.loudness.integrated_lufs:.1f} LUFS integrated")

    engine = result.analysis.key_engine
    _field(
        "Analysis engine",
        f"{engine.name if engine else 'unknown'}  {Glyph.DOT}  v{ANALYSIS_VERSION}",
    )
    _field(
        "Processing time",
        f"{result.analysis.processing_time_ms / 1000:.2f} s",
        note=(
            f"{result.analysis.realtime_ratio:.4f}x real-time"
            if result.analysis.realtime_ratio
            else None
        ),
    )

    if result.warnings:
        console.print(Text("Warnings", style="yellow"))
        for warning in result.warnings:
            console.print(Text(f"  {warning.code}: {warning.message}", style=_MUTED))
        console.print()

    console.print(Text(rule, style=_ACCENT))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _settings_from_options(
    engine: EngineChoice | None,
    profile: KeyProfile | None,
    segments: bool,
    max_seconds: float | None,
) -> Settings:
    """CLI flags override the environment, without mutating global settings."""
    base = get_settings()
    overrides: dict[str, object] = {}
    if engine is not None:
        overrides["key_engine"] = engine
        overrides["tempo_engine"] = engine
    if profile is not None:
        overrides["key_profile"] = profile
    if not segments:
        overrides["segments_enabled"] = False
    if max_seconds is not None:
        overrides["max_analysis_seconds"] = max_seconds
    return base.model_copy(update=overrides) if overrides else base


@app.command()
def analyze(
    file: Annotated[Path, typer.Argument(help="Audio file to analyse.", show_default=False)],
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the canonical result document.")
    ] = False,
    engine: Annotated[
        EngineChoice | None, typer.Option("--engine", help="Force an analysis backend.")
    ] = None,
    profile: Annotated[
        KeyProfile | None, typer.Option("--profile", help="Key profile to score against.")
    ] = None,
    segments: Annotated[
        bool, typer.Option("--segments/--no-segments", help="Time-windowed key analysis.")
    ] = True,
    max_seconds: Annotated[
        float | None, typer.Option("--max-seconds", help="Analyse only the first N seconds.")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log every stage to stderr.")
    ] = False,
) -> None:
    """Analyse one audio file."""
    settings = _settings_from_options(engine, profile, segments, max_seconds)
    # Logs go to stderr so that `--json` output stays pipeable. Without
    # --verbose the console stays quiet unless something is wrong.
    configure_logging("DEBUG" if verbose else "WARNING", settings.log_format)

    from .engine import analyze as run_analysis  # deferred: librosa import is slow

    try:
        result = run_analysis(file, settings=settings)
    except (AudioIngestError, ToolNotFoundError, BackendUnavailableError) as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=2) from exc
    except DJIntelligenceError as exc:  # pragma: no cover - defensive
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=1) from exc

    if as_json:
        # print(), not console.print(): rich would wrap and style it.
        print(result.model_dump_json(indent=2))
    else:
        _render(result)


@app.command()
def keys(
    camelot: Annotated[str, typer.Argument(help='Camelot key, e.g. "4A".')],
    extended: Annotated[
        bool, typer.Option("--extended", help="Include energy-boost and diagonal moves.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show the keys that mix with a Camelot key."""
    from .music.harmony import compatible_keys

    try:
        source = CamelotKey.parse(camelot)
    except InvalidKeyError as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=2) from exc

    relations = compatible_keys(source, extended=extended)
    if as_json:
        print(
            json.dumps(
                [
                    {
                        "camelot": relation.camelot.notation,
                        "relationship": relation.relationship.value,
                        "key": relation.camelot.tonic,
                        "mode": relation.camelot.mode.value,
                    }
                    for relation in relations
                ],
                indent=2,
            )
        )
        return

    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column(style=_MUTED)
    for relation in relations:
        table.add_row(
            relation.camelot.notation,
            f"{relation.camelot.key_name}  {Glyph.DOT}  {relation.relationship.value}",
        )
    console.print(Text(f"{source.notation}  ({source.key_name})", style=f"bold {_ACCENT}"))
    console.print(table)


@app.command()
def compat(
    camelot_a: Annotated[str, typer.Argument(help='Camelot of the outgoing track, or "-".')],
    bpm_a: Annotated[float, typer.Argument(help="BPM of the outgoing track.")],
    camelot_b: Annotated[str, typer.Argument(help='Camelot of the incoming track, or "-".')],
    bpm_b: Annotated[float, typer.Argument(help="BPM of the incoming track.")],
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Score mixing one track into another."""
    from .dj.compatibility import score_pair

    try:
        result = score_pair(
            TrackReference(camelot=None if camelot_a == "-" else camelot_a, bpm=bpm_a),
            TrackReference(camelot=None if camelot_b == "-" else camelot_b, bpm=bpm_b),
        )
    except (ValueError, InvalidKeyError) as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=2) from exc

    if as_json:
        print(result.model_dump_json(indent=2))
        return

    console.print(Text(f"score  {result.score:.2f}", style=f"bold {_ACCENT}"))
    table = Table.grid(padding=(0, 2))
    table.add_column(style=_MUTED)
    table.add_column()
    table.add_row(
        "harmonic",
        Glyph.DASH if result.components.harmonic is None else f"{result.components.harmonic:.2f}",
    )
    table.add_row(
        "tempo", Glyph.DASH if result.components.tempo is None else f"{result.components.tempo:.2f}"
    )
    console.print(table)
    console.print()
    for reason in result.reasons:
        console.print(Text(f"  {reason}", style=_MUTED))


@app.command()
def engines() -> None:
    """Report which analysis backends this installation can run."""
    from .analysis.registry import available_engines

    settings = get_settings()
    table = Table.grid(padding=(0, 2))
    table.add_column()
    table.add_column()
    for name, ready in available_engines().items():
        table.add_row(
            Text(Glyph.YES if ready else Glyph.NO, style="green" if ready else "red"),
            Text(name, style="" if ready else _MUTED),
        )
    console.print(Text("Backends", style=f"bold {_ACCENT}"))
    console.print(table)
    console.print()
    console.print(Text(f"configured key engine:   {settings.key_engine.value}", style=_MUTED))
    console.print(Text(f"configured tempo engine: {settings.tempo_engine.value}", style=_MUTED))
    console.print(Text(f"analysis version:        {ANALYSIS_VERSION}", style=_MUTED))
    console.print(Text(f"config fingerprint:      {settings.analysis_fingerprint}", style=_MUTED))


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    reload: Annotated[bool, typer.Option("--reload", help="Restart on code changes.")] = False,
) -> None:
    """Start the HTTP API."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "dj_intelligence.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_config=None,  # our own structured logging owns the output
    )


def main() -> None:
    """
    Console-script entry point.

    ``dj-analyze track.mp3`` has to work, and so does ``dj-analyze keys 4A``.
    Click cannot have both a bare argument and subcommands on the same group
    without the first token being ambiguous, so the dispatch is explicit: if
    the first token is not a known command and not an option, it is a file
    and the ``analyze`` command is implied.
    """
    argv = sys.argv[1:]
    if argv and argv[0] not in _COMMANDS and not argv[0].startswith("-"):
        sys.argv.insert(1, "analyze")
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
