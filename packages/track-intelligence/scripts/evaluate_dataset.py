#!/usr/bin/env python
"""
Measure the analyser against a labelled library of real music.

The synthetic fixtures in the test suite prove the pipeline is wired up. They
cannot prove accuracy, because they have no mastering, no reverb, no vocals
and a cleaner spectrum than any record. This script is where accuracy is
actually established, and where two engines get compared on the same
material.

    python scripts/evaluate_dataset.py dataset.csv
    python scripts/evaluate_dataset.py dataset.csv --engine essentia
    python scripts/evaluate_dataset.py dataset.csv --json results.json

The CSV needs a ``file`` column and at least one label column:

    file,expected_key,expected_mode,expected_camelot,expected_bpm,expected_first_downbeat
    track1.mp3,F,minor,4A,126,1.426
    track2.mp3,A,minor,8A,128,

Relative paths resolve against the CSV's own directory, so a dataset file can
sit next to the audio. ``expected_camelot`` is optional -- it is derived from
key and mode when absent, and cross-checked against them when present.

Metrics reported, and why each one:

*exact key*
    Tonic and mode both right. The headline number.

*camelot exact / camelot compatible*
    Compatible means the detected key is one of the four keys that mix with
    the true one. A DJ tool that is "wrong" but harmonically adjacent has
    failed much less badly than one that is a tritone out, and the gap
    between these two numbers is the most useful diagnostic here.

*relative confusion*
    How much of the error is F minor vs Ab major -- the same notes, the wrong
    tonic. It is the dominant failure mode of every chroma-based method, so
    it is broken out rather than buried in "wrong".

*BPM within 1% / 2%*
    Percentages, not absolute BPM: 2 BPM at 174 is a smaller error than 2 BPM
    at 90.

*BPM octave-tolerant*
    Counts a half- or double-time reading as correct. The difference between
    this and the strict figure is how often the engine picked the wrong
    metrical level rather than the wrong tempo.

*downbeat phase*
    Whether the detected bar lines land on the labelled first downbeat, within
    half a beat. This is the metric that catches a grid which is perfectly
    spaced and one beat out of phase -- invisible in every tempo figure above,
    and ruinous for bar jumps and phrase alignment.

*grid error*
    Where no ground truth exists, a track can still be scored against the
    constant grid its own tempo implies. ``--verify-grid`` reports the mean
    and p95 distance from each detected beat to the nearest gridline, which
    is the measurement the warp verifier uses and needs no labels at all.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Run from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dj_intelligence.analysis.pipeline import AnalysisPipeline
from dj_intelligence.config import Settings
from dj_intelligence.errors import DJIntelligenceError
from dj_intelligence.music.camelot import CamelotKey
from dj_intelligence.music.harmony import compatible_keys
from dj_intelligence.music.notes import (
    InvalidKeyError,
    parse_mode,
    parse_pitch_class,
)
from dj_intelligence.observability import configure_logging
from dj_intelligence.warp import grid_error_ms


@dataclass
class Expectation:
    path: Path
    camelot: CamelotKey | None = None
    bpm: float | None = None
    first_downbeat: float | None = None


@dataclass
class Tally:
    """Counters, kept explicit so the arithmetic in the report is obvious."""

    analysed: int = 0
    failed: int = 0
    keyed: int = 0
    key_exact: int = 0
    camelot_exact: int = 0
    camelot_compatible: int = 0
    mode_correct: int = 0
    relative_confusion: int = 0
    no_key_returned: int = 0

    downbeat_labelled: int = 0
    downbeat_phase_correct: int = 0
    downbeat_errors_ms: list[float] = field(default_factory=list)
    grid_mean_errors_ms: list[float] = field(default_factory=list)
    grid_p95_errors_ms: list[float] = field(default_factory=list)
    drift_counts: dict[str, int] = field(default_factory=dict)

    tempo_labelled: int = 0
    tempo_returned: int = 0
    bpm_errors: list[float] = field(default_factory=list)
    bpm_within_1: int = 0
    bpm_within_2: int = 0
    bpm_octave_tolerant: int = 0

    audio_seconds: float = 0.0
    processing_seconds: float = 0.0
    failures: list[tuple[str, str]] = field(default_factory=list)


def read_dataset(csv_path: Path) -> list[Expectation]:
    base = csv_path.parent
    rows: list[Expectation] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "file" not in reader.fieldnames:
            raise SystemExit(f"{csv_path}: needs a 'file' column")

        for line_number, row in enumerate(reader, start=2):
            name = (row.get("file") or "").strip()
            if not name or name.startswith("#"):
                continue
            path = Path(name)
            rows.append(
                Expectation(
                    path=path if path.is_absolute() else (base / path),
                    camelot=_expected_camelot(row, csv_path, line_number),
                    bpm=_expected_bpm(row),
                    first_downbeat=_expected_seconds(row, "expected_first_downbeat"),
                )
            )
    return rows


def _expected_camelot(row: dict[str, str], csv_path: Path, line: int) -> CamelotKey | None:
    """
    Read the label, from Camelot or from key+mode, and complain if they
    disagree. A dataset that contradicts itself will otherwise be measured
    against silently, and the numbers will be nonsense.
    """
    key = (row.get("expected_key") or "").strip()
    mode = (row.get("expected_mode") or "").strip()
    notation = (row.get("expected_camelot") or "").strip()

    from_key: CamelotKey | None = None
    if key and mode:
        try:
            from_key = CamelotKey.from_key(parse_pitch_class(key), parse_mode(mode))
        except InvalidKeyError as exc:
            raise SystemExit(f"{csv_path}:{line}: {exc}") from exc

    from_notation: CamelotKey | None = None
    if notation:
        try:
            from_notation = CamelotKey.parse(notation)
        except InvalidKeyError as exc:
            raise SystemExit(f"{csv_path}:{line}: {exc}") from exc

    if from_key and from_notation and from_key != from_notation:
        raise SystemExit(
            f"{csv_path}:{line}: label disagrees with itself -- "
            f"{key} {mode} is {from_key.notation}, not {from_notation.notation}"
        )
    return from_key or from_notation


def _expected_bpm(row: dict[str, str]) -> float | None:
    raw = (row.get("expected_bpm") or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _expected_seconds(row: dict[str, str], column: str) -> float | None:
    raw = (row.get(column) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _relative_of(key: CamelotKey) -> CamelotKey:
    return key.flipped()


def evaluate(
    rows: list[Expectation],
    pipeline: AnalysisPipeline,
    *,
    verbose: bool,
    verify_grid: bool = False,
) -> Tally:
    tally = Tally()

    for expectation in rows:
        if not expectation.path.is_file():
            tally.failed += 1
            tally.failures.append((expectation.path.name, "file not found"))
            continue

        started = time.perf_counter()
        try:
            result = pipeline.analyze(expectation.path)
        except (DJIntelligenceError, OSError) as exc:
            tally.failed += 1
            tally.failures.append((expectation.path.name, f"{type(exc).__name__}: {exc}"))
            continue

        elapsed = time.perf_counter() - started
        tally.analysed += 1
        tally.audio_seconds += result.audio.analysed_seconds
        tally.processing_seconds += elapsed

        detected = (
            CamelotKey.from_key(result.tonality.pitch_class, result.tonality.mode)
            if result.tonality.pitch_class is not None and result.tonality.mode is not None
            else None
        )

        if expectation.camelot is not None:
            tally.keyed += 1
            if detected is None:
                tally.no_key_returned += 1
            else:
                if detected == expectation.camelot:
                    tally.key_exact += 1
                    tally.camelot_exact += 1
                elif detected in {r.camelot for r in compatible_keys(expectation.camelot)}:
                    tally.camelot_compatible += 1
                if detected.mode is expectation.camelot.mode:
                    tally.mode_correct += 1
                if detected == _relative_of(expectation.camelot):
                    tally.relative_confusion += 1

        drift = result.rhythm.drift.classification.value
        tally.drift_counts[drift] = tally.drift_counts.get(drift, 0) + 1

        if expectation.first_downbeat is not None:
            tally.downbeat_labelled += 1
            downbeats = [entry.time for entry in result.rhythm.downbeats]
            if downbeats:
                distance = min(abs(entry - expectation.first_downbeat) for entry in downbeats)
                tally.downbeat_errors_ms.append(distance * 1000.0)
                # Half a beat: past that the grid is on the wrong pulse,
                # not merely imprecise.
                half_beat = 0.5 * 60.0 / (result.tempo.bpm or 126.0)
                if distance <= half_beat:
                    tally.downbeat_phase_correct += 1

        if verify_grid and result.tempo.bpm and result.rhythm.grid.first_downbeat_time is not None:
            errors = grid_error_ms(
                result.beats,
                target_bpm=result.tempo.bpm,
                anchor_time=result.rhythm.grid.first_downbeat_time,
            )
            if errors.size:
                tally.grid_mean_errors_ms.append(float(np.mean(errors)))
                tally.grid_p95_errors_ms.append(float(np.percentile(errors, 95)))

        if expectation.bpm is not None:
            tally.tempo_labelled += 1
            measured = result.tempo.bpm
            if measured is not None:
                tally.tempo_returned += 1
                relative = abs(measured - expectation.bpm) / expectation.bpm
                tally.bpm_errors.append(abs(measured - expectation.bpm))
                if relative <= 0.01:
                    tally.bpm_within_1 += 1
                if relative <= 0.02:
                    tally.bpm_within_2 += 1
                octave_best = min(
                    abs(candidate - expectation.bpm) / expectation.bpm
                    for candidate in (measured, measured * 2, measured / 2)
                )
                if octave_best <= 0.02:
                    tally.bpm_octave_tolerant += 1

        if verbose:
            expected_label = expectation.camelot.notation if expectation.camelot else "?"
            detected_label = detected.notation if detected else "none"
            mark = "ok " if detected and detected == expectation.camelot else "MISS"
            print(
                f"  {mark} {expectation.path.name[:48]:<48} "
                f"expected {expected_label:<4} got {detected_label:<4} "
                f"bpm {result.tempo.bpm or float('nan'):>7.2f} "
                f"(expected {expectation.bpm or float('nan'):>6.1f})"
            )

    return tally


def percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:6.1%}" if denominator else "     -"


def report(tally: Tally) -> dict[str, Any]:
    keyed = tally.keyed
    tempo = tally.tempo_labelled
    mae = sum(tally.bpm_errors) / len(tally.bpm_errors) if tally.bpm_errors else None

    summary: dict[str, Any] = {
        "analysed": tally.analysed,
        "failed": tally.failed,
        "key": {
            "labelled": keyed,
            "exact_accuracy": tally.key_exact / keyed if keyed else None,
            "camelot_exact_accuracy": tally.camelot_exact / keyed if keyed else None,
            "camelot_compatible_accuracy": (
                (tally.camelot_exact + tally.camelot_compatible) / keyed if keyed else None
            ),
            "mode_accuracy": tally.mode_correct / keyed if keyed else None,
            "relative_confusion_rate": tally.relative_confusion / keyed if keyed else None,
            "no_key_returned": tally.no_key_returned,
        },
        "tempo": {
            "labelled": tempo,
            "returned": tally.tempo_returned,
            "bpm_mae": mae,
            "within_1_percent": tally.bpm_within_1 / tempo if tempo else None,
            "within_2_percent": tally.bpm_within_2 / tempo if tempo else None,
            "octave_tolerant_within_2_percent": (
                tally.bpm_octave_tolerant / tempo if tempo else None
            ),
        },
        "rhythm": {
            "downbeat_labelled": tally.downbeat_labelled,
            "downbeat_phase_accuracy": (
                tally.downbeat_phase_correct / tally.downbeat_labelled
                if tally.downbeat_labelled
                else None
            ),
            "downbeat_mae_ms": (
                sum(tally.downbeat_errors_ms) / len(tally.downbeat_errors_ms)
                if tally.downbeat_errors_ms
                else None
            ),
            "drift_distribution": tally.drift_counts,
            "grid_mean_error_ms": (
                sum(tally.grid_mean_errors_ms) / len(tally.grid_mean_errors_ms)
                if tally.grid_mean_errors_ms
                else None
            ),
            "grid_p95_error_ms": (
                sum(tally.grid_p95_errors_ms) / len(tally.grid_p95_errors_ms)
                if tally.grid_p95_errors_ms
                else None
            ),
        },
        "performance": {
            "audio_seconds": round(tally.audio_seconds, 1),
            "processing_seconds": round(tally.processing_seconds, 2),
            "realtime_ratio": (
                round(tally.processing_seconds / tally.audio_seconds, 5)
                if tally.audio_seconds
                else None
            ),
        },
        "failures": [{"file": name, "error": reason} for name, reason in tally.failures],
    }

    print()
    print(f"  analysed              {tally.analysed}")
    print(f"  failed                {tally.failed}")
    print()
    print("  KEY")
    print(f"    labelled            {keyed}")
    print(f"    exact key           {percent(tally.key_exact, keyed)}")
    print(f"    camelot exact       {percent(tally.camelot_exact, keyed)}")
    print(
        f"    camelot compatible  {percent(tally.camelot_exact + tally.camelot_compatible, keyed)}"
    )
    print(f"    major/minor         {percent(tally.mode_correct, keyed)}")
    print(f"    relative confusion  {percent(tally.relative_confusion, keyed)}")
    print(f"    no key returned     {tally.no_key_returned}")
    print()
    print("  TEMPO")
    print(f"    labelled            {tempo}")
    print(f"    BPM MAE             {f'{mae:.3f}' if mae is not None else '-'}")
    print(f"    within 1%           {percent(tally.bpm_within_1, tempo)}")
    print(f"    within 2%           {percent(tally.bpm_within_2, tempo)}")
    print(f"    within 2% (1/2, 2x) {percent(tally.bpm_octave_tolerant, tempo)}")
    print()
    print("  RHYTHM")
    print(f"    downbeat labelled   {tally.downbeat_labelled}")
    print(
        f"    downbeat phase      {percent(tally.downbeat_phase_correct, tally.downbeat_labelled)}"
    )
    if tally.downbeat_errors_ms:
        downbeat_mae = sum(tally.downbeat_errors_ms) / len(tally.downbeat_errors_ms)
        print(f"    downbeat MAE        {downbeat_mae:.1f} ms")
    if tally.grid_mean_errors_ms:
        grid_mean = sum(tally.grid_mean_errors_ms) / len(tally.grid_mean_errors_ms)
        grid_p95 = sum(tally.grid_p95_errors_ms) / len(tally.grid_p95_errors_ms)
        print(f"    grid error mean     {grid_mean:.1f} ms")
        print(f"    grid error p95      {grid_p95:.1f} ms")
    for label, count in sorted(tally.drift_counts.items()):
        print(f"    drift: {label:<13} {count}")
    print()
    print("  PERFORMANCE")
    print(f"    audio               {tally.audio_seconds / 60:.1f} min")
    print(f"    processing          {tally.processing_seconds:.1f} s")
    if tally.audio_seconds:
        ratio = tally.processing_seconds / tally.audio_seconds
        print(f"    realtime ratio      {ratio:.4f}x  ({1 / ratio:.0f}x faster than playback)")
    print()

    if tally.failures:
        print("  FAILURES")
        for name, reason in tally.failures[:20]:
            print(f"    {name}: {reason}")
        if len(tally.failures) > 20:
            print(f"    ... and {len(tally.failures) - 20} more")
        print()

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="CSV of labelled tracks.")
    parser.add_argument("--engine", default=None, help="Force a backend: essentia | chroma.")
    parser.add_argument(
        "--profile", default=None, help="Key profile: edma | temperley | krumhansl."
    )
    parser.add_argument("--no-segments", action="store_true", help="Skip windowed key analysis.")
    parser.add_argument(
        "--verify-grid",
        action="store_true",
        help="Score every beat against the track's own constant grid; needs no labels.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Write the summary here.")
    parser.add_argument("-v", "--verbose", action="store_true", help="One line per track.")
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"no such dataset: {args.dataset}")

    configure_logging("WARNING", "console")

    overrides: dict[str, Any] = {"log_level": "WARNING", "profile": "full"}
    if args.engine:
        overrides["key_engine"] = args.engine
        overrides["tempo_engine"] = args.engine
    if args.profile:
        overrides["key_profile"] = args.profile
    if args.no_segments:
        overrides["segments_enabled"] = False

    settings = Settings(**overrides)
    pipeline = AnalysisPipeline(settings)

    rows = read_dataset(args.dataset)
    print(f"\n  {len(rows)} tracks from {args.dataset}")
    print(
        f"  key engine: {pipeline.key_analyzer.name}   tempo engine: {pipeline.tempo_analyzer.name}"
    )
    print(f"  profile: {settings.key_profile.value}   fingerprint: {settings.analysis_fingerprint}")
    if args.verbose:
        print()

    summary = report(evaluate(rows, pipeline, verbose=args.verbose, verify_grid=args.verify_grid))
    summary["configuration"] = {
        "key_engine": pipeline.key_analyzer.name,
        "tempo_engine": pipeline.tempo_analyzer.name,
        "fingerprint": settings.analysis_fingerprint,
        "parameters": settings.analysis_parameters(),
    }

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  wrote {args.json}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
