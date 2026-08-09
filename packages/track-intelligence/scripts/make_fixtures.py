#!/usr/bin/env python
"""
Write the generated audio fixtures to disk.

The test suite generates these into a temporary directory and throws them
away. This writes them somewhere you can listen to them, which is the only
way to check that a fixture actually sounds like the key it claims -- a
detail worth verifying by ear before trusting any test that depends on it.

    python scripts/make_fixtures.py                 -> fixtures/generated/
    python scripts/make_fixtures.py --all-keys      -> all 24 keys
    python scripts/make_fixtures.py --out /tmp/x

Nothing here is committed; ``fixtures/generated/`` is ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from dj_intelligence.music.notes import Mode, canonical_key_name
from dj_intelligence.synth import (
    Progression,
    render_click,
    render_noise,
    render_progression,
    write_wav,
)

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "fixtures" / "generated"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all-keys", action="store_true", help="All 24 keys, not just three.")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--bpm", type=float, default=126.0)
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    keys = (
        [(pc, mode) for pc in range(12) for mode in Mode]
        if args.all_keys
        else [(5, Mode.MINOR), (9, Mode.MINOR), (0, Mode.MAJOR)]
    )
    for pitch_class, mode in keys:
        tonic = canonical_key_name(pitch_class, mode).replace("#", "s")
        name = f"{tonic}_{mode.value}_{args.bpm:g}bpm.wav"
        written.append(
            write_wav(
                out / name,
                render_progression(
                    Progression(pitch_class, mode, bpm=args.bpm, seconds=args.seconds)
                ),
            )
        )

    written.append(write_wav(out / "click_128bpm.wav", render_click(128.0, 20.0)))
    written.append(write_wav(out / "noise.wav", render_noise(10.0)))
    written.append(write_wav(out / "silence.wav", np.zeros((44100 * 10), dtype=np.float32)))

    # A modulation, for checking segmentation by ear.
    written.append(
        write_wav(
            out / "modulation_Aminor_to_Fminor.wav",
            np.concatenate(
                [
                    render_progression(Progression(9, Mode.MINOR, bpm=args.bpm, seconds=45.0)),
                    render_progression(Progression(5, Mode.MINOR, bpm=args.bpm, seconds=45.0)),
                ]
            ),
        )
    )

    for path in written:
        print(f"  {path.relative_to(out.parent) if out.parent in path.parents else path}")
    print(f"\n  {len(written)} files in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
