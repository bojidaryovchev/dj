"""
The ``dj-analyze`` command.

A presentation layer and nothing more. Every number it prints comes from
:func:`dj_intelligence.engine.analyze` or :func:`dj_intelligence.warp.warp_track`,
which is what the HTTP API calls too -- there is no second analysis path to
drift out of step. Rendering lives in :mod:`dj_intelligence.reporting`.

    dj-analyze track.mp3            human-readable report
    dj-analyze track.mp3 --json     the canonical document, for piping
    dj-analyze grid track.mp3       bars, meter, drift, warp advice
    dj-analyze warp track.mp3 --target-bpm 126 -o corrected.wav
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
from rich.table import Table
from rich.text import Text

from .config import AnalysisProfile, EngineChoice, KeyProfile, Settings, get_settings
from .errors import (
    AudioIngestError,
    BackendUnavailableError,
    DJIntelligenceError,
    ToolNotFoundError,
)
from .models.compatibility import TrackReference
from .music.camelot import CamelotKey
from .music.notes import InvalidKeyError
from .observability import configure_logging
from .reporting import (
    ACCENT,
    MUTED,
    Glyph,
    console,
    render_analysis,
    render_beats,
    render_grid,
    render_warp,
    stderr_console,
)
from .version import ANALYSIS_VERSION

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Analyse audio for DJ-relevant musical properties.",
)

_COMMANDS = {"analyze", "grid", "warp", "keys", "compat", "engines", "serve"}

_ACCENT = ACCENT
_MUTED = MUTED


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def _settings_from_options(
    engine: EngineChoice | None,
    profile: KeyProfile | None,
    segments: bool,
    max_seconds: float | None,
    depth: AnalysisProfile | None = None,
) -> Settings:
    """CLI flags override the environment, without mutating global settings."""
    base = get_settings()
    overrides: dict[str, object] = {}
    if depth is not None:
        overrides["profile"] = depth
    if engine is not None:
        overrides["key_engine"] = engine
        overrides["tempo_engine"] = engine
    if profile is not None:
        overrides["key_profile"] = profile
    if not segments:
        overrides["segments_enabled"] = False
    if max_seconds is not None:
        overrides["max_analysis_seconds"] = max_seconds
    return base.with_overrides(**overrides) if overrides else base


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
    analysis_profile: Annotated[
        AnalysisProfile | None,
        typer.Option("--depth", help="How much of the pipeline to run: basic, full or warp."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log every stage to stderr.")
    ] = False,
) -> None:
    """Analyse one audio file."""
    settings = _settings_from_options(engine, profile, segments, max_seconds, analysis_profile)
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
        render_analysis(result)


@app.command()
def grid(
    file: Annotated[Path, typer.Argument(help="Audio file to analyse.", show_default=False)],
    as_json: Annotated[bool, typer.Option("--json", help="Emit the rhythm block as JSON.")] = False,
    beats: Annotated[
        bool, typer.Option("--beats", help="List every beat with its bar position.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Show the rhythmic grid: bars, meter, tempo drift and warp advice."""
    settings = get_settings().with_overrides(profile=AnalysisProfile.WARP)
    configure_logging("DEBUG" if verbose else "WARNING", settings.log_format)

    from .engine import analyze as run_analysis

    try:
        result = run_analysis(file, settings=settings)
    except (AudioIngestError, ToolNotFoundError, BackendUnavailableError) as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=2) from exc

    if as_json:
        print(
            json.dumps(
                {
                    "rhythm": result.rhythm.model_dump(mode="json"),
                    "structure": result.structure.model_dump(mode="json"),
                    "warp": result.warp.model_dump(mode="json") if result.warp else None,
                },
                indent=2,
            )
        )
        return

    render_grid(result)
    if beats:
        render_beats(result)


@app.command()
def warp(
    file: Annotated[Path, typer.Argument(help="Audio file to correct.", show_default=False)],
    target_bpm: Annotated[
        float | None, typer.Option("--target-bpm", help="Constant tempo to warp to.")
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write. Default: <name>.warped.wav"),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Render even when it is not recommended.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Plan and report; write nothing.")
    ] = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip re-analysing the rendered file.")
    ] = False,
    as_json: Annotated[bool, typer.Option("--json")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Correct a track's beat grid by time-stretching it. Pitch is preserved.

    Does nothing by default when the grid is already within tolerance, because
    stretching a well-sequenced track costs transient quality for no gain.
    Use --force to override, or --dry-run to see the plan.
    """
    settings = get_settings()
    configure_logging("DEBUG" if verbose else "WARNING", settings.log_format)

    try:
        if dry_run:
            from .engine import analyze as run_analysis

            analysis = run_analysis(
                file,
                settings=settings.with_overrides(profile=AnalysisProfile.WARP),
                target_bpm=target_bpm,
            )
            rendered, report = False, None
            skipped = (
                analysis.warp.recommendation.reason if analysis.warp else "No beat grid was found."
            )
            skipped = f"[dry run] {skipped}"
        else:
            from .warp import warp_track

            outcome = warp_track(
                file,
                output=output,
                target_bpm=target_bpm,
                force=force,
                verify=not no_verify,
                settings=settings,
            )
            analysis = outcome.analysis
            rendered = outcome.rendered
            report = outcome.report
            skipped = outcome.skipped_reason
    except (AudioIngestError, ToolNotFoundError, BackendUnavailableError) as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=2) from exc
    except DJIntelligenceError as exc:
        stderr_console.print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(code=1) from exc

    if as_json:
        print(
            json.dumps(
                {
                    "rendered": rendered,
                    "skipped_reason": skipped,
                    "warp": analysis.warp.model_dump(mode="json") if analysis.warp else None,
                    "report": report.model_dump(mode="json") if report else None,
                },
                indent=2,
            )
        )
        return

    render_warp(analysis, rendered=rendered, report=report, skipped_reason=skipped)


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
