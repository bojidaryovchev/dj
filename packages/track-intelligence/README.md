# DJ Track Intelligence

Audio in, DJ-relevant musical facts out.

```bash
dj-analyze track.mp3
```

```text
-----------------------------------------
 DJ TRACK ANALYSIS
-----------------------------------------

File
track.mp3

Duration
5:52

Tempo
126.04 BPM

Key
F minor

Camelot
4A

Confidence
tonal  0.87  (key_profile_correlation)
tempo  0.90  (beat_interval_consistency)

Recommended harmonic neighbours
4A  F minor   ·  same_key
3A  Bb minor  ·  adjacent_minus
5A  C minor   ·  adjacent_plus
4B  Ab major  ·  relative_major

Loudness
-9.4 LUFS integrated

Analysis engine
chroma  ·  v1.0.0

Processing time
3.03 s  0.0084x real-time
-----------------------------------------
```

The first version does key, Camelot, tempo, beat grid, tonal segmentation and
loudness. The architecture is built for what comes after that — energy, phrase
structure, cue points, "what should I play next" — without pretending any of
it exists yet.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [How key detection works](#how-key-detection-works)
- [Install](#install)
- [CLI](#cli)
- [HTTP API](#http-api)
- [Docker](#docker)
- [Analysis engines](#analysis-engines)
- [The result document](#the-result-document)
- [Confidence, and what it is not](#confidence-and-what-it-is-not)
- [Configuration](#configuration)
- [Testing](#testing)
- [Validating against real music](#validating-against-real-music)
- [Performance](#performance)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## What it does

Give it a file in any format ffmpeg can open — MP3, WAV, FLAC, M4A/AAC, OGG,
Opus, AIFF, WMA — and it returns:

| | |
|---|---|
| **Key** | tonic, mode, confidence, and the runners-up |
| **Camelot** | 4A, 8B … derived from the key, plus the keys that mix with it |
| **Tempo** | BPM, beat grid, stability, and the half/double readings |
| **Tonal segments** | where in the track the key changes |
| **Loudness** | integrated LUFS, loudness range, true peak (EBU R128) |
| **Identity** | SHA-256 of the file, so nothing is analysed twice |
| **Provenance** | which engine, which version, which settings |

What it does **not** do is guess. Percussion-only material, noise, silence and
atonal recordings come back with `key: null` and a warning saying why, because
a plausible wrong key is worse than an honest gap — it is indistinguishable
from a real answer once it is in a library.

---

## Architecture

```text
                 audio file
                     |
        +------------v------------+
        |  ingest                 |   sha256 · ffprobe · ffmpeg decode
        |  audio/                 |   -> mono float32 @ 44.1 kHz
        +------------+------------+
                     |
        +------------v------------+
        |  measurement            |   "what is in this signal?"
        |  analysis/              |
        |    tonal content gate   |   is this even pitched?
        |    tempo analyzer       |   BPM · beats · stability
        |    key analyzer         |   tonic · mode · confidence
        |    segment analyzer     |   key over time
        |    loudness analyzer    |   LUFS · LRA · true peak
        +------------+------------+
                     |
        +------------v------------+
        |  interpretation         |   "what should a DJ do about it?"
        |  dj/                    |   Camelot · neighbours · mix BPM
        +------------+------------+          · compatibility scoring
                     |
              TrackAnalysis
```

Three rules hold the shape together.

**Measurement is separate from interpretation.** `analysis/` answers questions
about a signal and has never heard of the Camelot wheel. `dj/` turns those
answers into DJ convention. The dependency runs one way, enforced by the
import graph: nothing under `analysis/` imports `dj/` except the pipeline, in
its role as the aggregator. Swap the key backend and "4A" cannot change
meaning; change what neighbours we recommend and no measurement moves.

**Backends are interchangeable.** Every analyser satisfies a `Protocol` in
[analysis/base.py](src/dj_intelligence/analysis/base.py) — `KeyAnalyzer`,
`TempoAnalyzer`, `SegmentKeyAnalyzer`, `LoudnessAnalyzer`. Essentia, librosa,
madmom, a trained model or a third-party service all fit the same three
methods, and the pipeline cannot tell them apart. Selection happens in one
place, [analysis/registry.py](src/dj_intelligence/analysis/registry.py).

**One analysis path.** The CLI and the API both call
[`engine.analyze()`](src/dj_intelligence/engine.py). There is no second
implementation to drift.

```text
src/dj_intelligence/
├── audio/           decode, probe, hash — the codec stops here
├── analysis/        measurement
│   ├── base.py        the Protocols every backend implements
│   ├── registry.py    which backend, and the fallback
│   ├── pipeline.py    orchestration, timing, error containment
│   ├── key/           chroma + essentia backends, profiles,
│   │                  segmentation, the tonal-content gate
│   ├── tempo/         librosa + essentia backends, shared reasoning
│   └── loudness/      EBU R128 via ffmpeg
├── music/           theory: notes, Camelot, harmonic relationships
├── dj/              interpretation: Camelot view, compatibility scoring
├── models/          the canonical result document
├── api/             FastAPI: routes, uploads, error translation
├── cli.py           dj-analyze
├── engine.py        the single entry point both surfaces use
├── config.py        every knob, and the fingerprint over them
└── synth.py         reference-signal generator for tests
```

---

## How key detection works

None of this is invented here. It is the standard music-information-retrieval
chain, and the value added is in wiring it correctly, refusing to answer when
the audio does not support an answer, and recording enough to reproduce the
result later.

```text
audio
  │
  ├─ constant-Q transform ─────────  a spectrum whose bins are logarithmically
  │                                  spaced, so one bin is one musical interval
  │                                  at every octave
  │
  ├─ fold to 12 pitch classes ─────  the chromagram: how much energy sits on C,
  │                                  C#, D … regardless of octave
  │
  ├─ estimate tuning ──────────────  plenty of records are not at A=440; the
  │                                  offset is measured, not assumed
  │
  ├─ aggregate over time ──────────  median across frames, so one crash cymbal
  │                                  does not decide the key
  │
  ├─ is it even tonal? ────────────  a flat chroma means percussion or noise.
  │                                  Stop here if so — see below
  │
  ├─ correlate against 24 profiles ─ one template per key. The best-fitting
  │                                  template names the tonic and the mode
  │
  └─ map to Camelot ───────────────  deterministic, in the DJ layer
```

**The profiles** are published, not tuned by us:

- `edma` — Faraldo, Gómez, Jordà & Herrera (2016), fitted to a corpus of
  electronic dance music. The default, because that is the corpus this tool
  analyses.
- `temperley` — Temperley (2001), fitted to the Kostka-Payne corpus.
- `krumhansl` — Krumhansl & Kessler (1982) probe-tone ratings. The original;
  assumes classical voice leading.

**The "is it even tonal?" step is not decoration.** Pearson correlation is
invariant to scale and offset, so it measures the *shape* of a chroma vector
and ignores its magnitude — and noise has a shape. Measured here:

| material | tonal salience | correlation would say |
|---|---|---|
| white noise | 0.0001 | F# minor, 0.54 |
| bare click track | 0.0007 | Db major, 0.53 |
| synthesised chord progressions | 0.080 – 0.116 | correct, 0.81 – 0.88 |
| real mastered track | 0.097 | D minor, 0.83 |

Two orders of magnitude separate "no tonal content" from "any tonal content",
with nothing in between, so the gate sits at 0.01. It runs *before* the key
backend and applies to whichever one is configured — including Essentia, whose
`KeyExtractor` otherwise reports **C major, strength 0.76** for a bare click
track and **F major, 0.70** for white noise. That is not a defect in Essentia;
key estimators are built to answer "which key", not "is this tonal". The
question just has to be asked somewhere, and this is where.

**Tempo** follows the same shape: a spectral-flux onset envelope, then
dynamic-programming beat tracking (Ellis, 2007). The BPM is then fitted by
least squares across the whole beat grid rather than taken from the median
interval — beat trackers place beats on onset frames, and at librosa's default
hop a 126 BPM beat lasts 20.5 frames, so every individual interval must round
to 20 (129.20 BPM) or 21 (123.05 BPM). Taking the median inherits that
quantisation and is wrong by up to 2.4%; fitting across the grid brings the
mean error on generated fixtures from **1.03% to 0.016%**.

---

## Install

Requires **Python 3.12+** and **ffmpeg** on `PATH`.

```bash
cd packages/track-intelligence
uv sync --extra dev        # or: make dev
```

[uv](https://docs.astral.sh/uv/) handles the virtualenv and the interpreter. If
you would rather not:

```bash
python -m venv .venv && . .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

**ffmpeg.** Any recent build. On Windows the crate helper in this monorepo
already downloads one, and you can point at it instead of installing another:

```powershell
$env:DJTI_FFMPEG_PATH  = "d:\repos\dj\packages\crate-helper\bin\ffmpeg.exe"
$env:DJTI_FFPROBE_PATH = "d:\repos\dj\packages\crate-helper\bin\ffprobe.exe"
```

**Essentia** is optional and is not installed by the command above — see
[Analysis engines](#analysis-engines).

---

## CLI

```bash
dj-analyze track.mp3                    # human-readable report
dj-analyze track.mp3 --json             # the canonical document
dj-analyze track.mp3 --json | jq .dj    # logs go to stderr, so this is safe
dj-analyze track.mp3 -v                 # log every stage and its duration

dj-analyze keys 4A                      # what mixes with 4A
dj-analyze keys 4A --extended           # ...including the deliberate moves
dj-analyze compat 4A 126 5A 126.5       # score a transition
dj-analyze engines                      # which backends this install can run
dj-analyze serve                        # start the HTTP API
```

Options on `analyze`:

| flag | effect |
|---|---|
| `--json` | emit the result document instead of the report |
| `--engine essentia\|chroma\|auto` | force a backend |
| `--profile edma\|temperley\|krumhansl` | key profile to score against |
| `--no-segments` | skip time-windowed key analysis |
| `--max-seconds N` | analyse only the first N seconds |
| `-v`, `--verbose` | stage-by-stage logging on stderr |

`dj-analyze track.mp3` and `dj-analyze keys 4A` both work: if the first
argument is not a subcommand, `analyze` is implied.

---

## HTTP API

```bash
dj-analyze serve                     # or: uvicorn dj_intelligence.api.app:create_app --factory
```

Interactive docs at `http://localhost:8000/docs`.

### `POST /v1/tracks/analyze`

```bash
curl -X POST -F "file=@track.mp3" http://localhost:8000/v1/tracks/analyze
```

Optional form fields: `segments` (bool), `max_seconds` (float),
`include_beats` (bool — a beat grid is thousands of floats on a long track).

### `GET /v1/dj/camelot/{camelot}/compatible`

```bash
curl http://localhost:8000/v1/dj/camelot/4A/compatible
```

```json
[
  { "camelot": "4A", "relationship": "same_key",       "key": "F",  "mode": "minor" },
  { "camelot": "3A", "relationship": "adjacent_minus", "key": "Bb", "mode": "minor" },
  { "camelot": "5A", "relationship": "adjacent_plus",  "key": "C",  "mode": "minor" },
  { "camelot": "4B", "relationship": "relative_major", "key": "Ab", "mode": "major" }
]
```

### `POST /v1/dj/compatibility`

```bash
curl -X POST http://localhost:8000/v1/dj/compatibility \
  -H 'content-type: application/json' \
  -d '{"track_a":{"camelot":"4A","bpm":126},"track_b":{"camelot":"5A","bpm":126.5}}'
```

```json
{
  "score": 0.9502,
  "comparable": true,
  "components": { "harmonic": 0.95, "tempo": 0.9505 },
  "reasons": [
    "Adjacent Camelot key, up a fifth (energy lift)",
    "Tempo difference 0.4%"
  ],
  "harmonic_relationship": "adjacent_plus",
  "tempo_relation": "primary",
  "bpm_difference_percent": 0.396,
  "rules_version": "1.0.0"
}
```

The scoring is a **rule system, not a model**. It does not predict whether a
mix will sound good; it encodes what harmonic mixing and beatmatching
conventionally consider workable, so a library can be sorted by it. Every rule
is a documented number in
[`ScoringRules`](src/dj_intelligence/dj/compatibility.py): harmonic score by
distance on the wheel (mode changes penalised separately from position
changes), tempo score linear to zero at 8% — the pitch range of a CDJ — with
half- and double-time matches considered and penalised rather than ignored.
Components that cannot be computed are dropped and the remaining weights
renormalised, so a track with no detected key is scored on tempo alone instead
of punished for a measurement we failed to make.

### `GET /health`, `GET /ready`

`/health` is liveness and touches nothing. `/ready` resolves ffmpeg and reports
which backends are importable, returning 503 when the answer is no.

### Errors

Every deliberate failure has a stable code and the request id:

```json
{ "error": "unsupported_format", "detail": "ffprobe could not read track.mp3: …", "request_id": "a1b2c3d4e5f6" }
```

| code | status | |
|---|---|---|
| `file_too_large` | 413 | over `DJTI_MAX_UPLOAD_BYTES` |
| `unsupported_format` | 415 | not decodable audio, whatever the extension claims |
| `empty_audio` | 422 | zero decodable samples |
| `audio_too_short` | 422 | decodes, too short to analyse |
| `decode_failed` | 422 | truncated, corrupt or encrypted |
| `ffmpeg_unavailable` | 503 | ffmpeg is not installed |
| `backend_unavailable` | 503 | a named engine cannot be imported |

Uploads are treated as hostile: streamed to a `mkstemp` file so the client's
filename is never used as a path, size-limited *while* streaming rather than
after, format-validated by ffprobe rather than by extension, always deleted,
and every ffmpeg invocation is a list of arguments with no shell. Server paths
are stripped out of error text before it reaches a client.

---

## Docker

The container is where Essentia actually runs.

```bash
docker build -t dj-track-intelligence .
docker run --rm -p 8000:8000 dj-track-intelligence
# or
docker compose up --build
```

Analyse a file by path:

```bash
docker run --rm -v "$PWD/music:/music:ro" dj-track-intelligence dj-analyze /music/track.mp3
```

The build fails rather than the first request if ffmpeg or either backend is
missing — the image asserts all three at build time. It runs as a non-root
user and ships a healthcheck.

---

## Analysis engines

Two backends, one interface.

| | `chroma` (librosa) | `essentia` |
|---|---|---|
| key | CQT chroma + profile correlation | `KeyExtractor` |
| tempo | `beat_track`, DP (Ellis 2007) | `RhythmExtractor2013` multifeature |
| confidence | derived, documented | reported by the library |
| runner-up keys | yes | no — `KeyExtractor` exposes only the winner |
| **Windows** | **works** | **no wheel exists** |
| Linux / macOS | works | works |
| Docker image | works | works, and is the default there |

**Essentia has no Windows wheel.** It publishes manylinux and macOS wheels
only, so `pip install 'dj-track-intelligence[essentia]'` cannot succeed on
Windows. This is not a packaging oversight to be worked around; it is why the
chroma backend is a first-class implementation rather than a toy, and why the
Docker image exists. On Linux and macOS:

```bash
uv sync --extra essentia
```

`DJTI_KEY_ENGINE=auto` (the default) prefers Essentia and falls back to chroma
when it cannot be imported. Naming an engine explicitly and not having it is an
error, not a silent substitution. The engine that ran is always recorded in
`analysis.key_engine`.

**Do they agree?** On the generated fixtures, exactly — same key and same
Camelot on all of them, with Essentia's confidences running a few points
higher:

```text
file                     essentia                   chroma
F_minor_126bpm.wav       F minor 4A c=0.90 126.0    F minor 4A c=0.87 125.98
A_minor_126bpm.wav       A minor 8A c=0.88 126.0    A minor 8A c=0.83 125.98
C_major_126bpm.wav       C major 8B c=0.96 126.0    C major 8B c=0.90 125.97
Ab_major_126bpm.wav      Ab major 4B c=0.95 126.0   Ab major 4B c=0.89 125.99
Bb_minor_126bpm.wav      Bb minor 3A c=0.89 126.0   Bb minor 3A c=0.82 125.99
```

Reproduce with `scripts/evaluate_dataset.py dataset.csv --engine essentia`.
Consensus between the two is not used to adjust confidence yet; the
disagreement is worth measuring on real music first.

---

## The result document

```jsonc
{
  "schema_version": "1.0",
  "track":   { "filename": "track.mp3", "sha256": "…", "size_bytes": 9310720 },
  "audio":   { "duration_seconds": 352.31, "analysis_sample_rate": 44100,
               "source_sample_rate": 44100, "source_channels": 2,
               "codec": "mp3", "container": "mp3", "bit_rate_bps": 320000,
               "analysed_seconds": 352.31 },

  // measurement --------------------------------------------------------
  "tempo":   { "bpm": 126.04, "confidence": 0.9, "confidence_type": "beat_interval_consistency",
               "reliable": true, "stable": true, "beat_interval_cv": 0.0255,
               "candidates": [ { "bpm": 126.04, "relation": "primary",     "in_dj_range": true  },
                               { "bpm": 63.02,  "relation": "half_time",   "in_dj_range": false },
                               { "bpm": 252.08, "relation": "double_time", "in_dj_range": false } ],
               "beat_count": 739 },
  "tonality":{ "pitch_class": 5, "mode": "minor", "key": "F",
               "confidence": 0.8703, "confidence_type": "key_profile_correlation",
               "reliable": true,
               "alternatives": [ { "pitch_class": 8, "mode": "major", "key": "Ab", "score": 0.6166 } ] },
  "loudness":{ "integrated_lufs": -9.4, "loudness_range_lu": 5.6,
               "true_peak_dbtp": -0.2, "sample_peak_dbfs": -0.3, "rms_dbfs": -12.1 },
  "tonal_segments": [ { "start_seconds": 0.0, "end_seconds": 96.0,
                        "pitch_class": 5, "mode": "minor", "key": "F",
                        "confidence": 0.84, "reliable": true } ],
  "beats": [0.371, 0.847, 1.323],
  "downbeats": null,

  // interpretation -----------------------------------------------------
  "dj": {
    "camelot": "4A",
    "key_label": "F minor",
    "compatible_keys": [ { "camelot": "4A", "relationship": "same_key", "key": "F", "mode": "minor" } ],
    "mix_bpm": 126.04,
    "mix_bpm_relation": "primary",
    "segments": [ { "start_seconds": 0.0, "end_seconds": 96.0, "camelot": "4A",
                    "key_label": "F minor", "relationship_to_global": "same_key",
                    "reliable": true } ]
  },

  "warnings": [ { "code": "tempo_unstable", "message": "…", "stage": "tempo" } ],
  "analysis": {
    "analysis_version": "1.0.0", "schema_version": "1.0", "package_version": "0.1.0",
    "key_engine":   { "name": "chroma", "algorithm": "librosa.chroma_cqt + key-profile correlation",
                      "library_version": "0.11.0", "parameters": { "profile": "edma" } },
    "tempo_engine": { "name": "librosa", "algorithm": "librosa.beat.beat_track (…)" },
    "configuration_fingerprint": "464f7ab646452839",
    "parameters": { "key_profile": "edma" },
    "processing_time_ms": 3030.4,
    "realtime_ratio": 0.0084,
    "stages": [ { "stage": "decode", "duration_ms": 365.0 } ]
  }
}
```

Two notes on shape.

**Camelot lives under `dj`, not under `tonality`.** Separating musical analysis
from DJ interpretation is an explicit design rule, and a schema that mixes them
undoes it in the one place every consumer looks. `tonality` is what the signal
says; `dj` is what a DJ does with it. The CLI prints Camelot next to the key
because that is a presentation decision, which is exactly the point.

**Not-yet-implemented features are absent, not `null`.** There is no
`energy: null` or `danceability: null`, because a null field implies we tried.
When they arrive they will be real measurements. `downbeats` is the exception —
it is explicitly `null` because beat tracking runs and downbeat tracking is a
separate algorithm neither backend performs, and saying so is more useful than
omitting it.

### Versioning

`analysis_version` (currently `1.0.0`) is the semantics of the numbers. It is
bumped whenever a change could make the same file analyse differently — a new
default profile, a changed confidence formula, a fixed bug. A stored result
carrying an older value is stale and should be re-analysed.

`schema_version` is the shape of the document. `configuration_fingerprint` is a
hash of the settings that can move a number — deliberately excluding transport,
logging and limits, so moving the API to another port does not invalidate a
library. Together they answer "would re-running this today give the same
answer?" without storing the whole configuration next to every row.

---

## Confidence, and what it is not

None of these numbers are probabilities. `confidence_type` says which quantity
you are looking at so that no consumer can mistake one for another:

| `confidence_type` | what it is |
|---|---|
| `key_profile_correlation` | Pearson correlation of the chroma vector with the winning key profile, negatives clipped to 0. |
| `essentia_key_strength` | Essentia's `strength` output, unmodified. Also a profile correlation, over its own HPCP. |
| `essentia_beat_confidence` | `RhythmExtractor2013` multifeature confidence ÷ its documented maximum of 5.32. Essentia calls ~1.5 low and ~3.5 high, i.e. 0.28 and 0.66 here. |
| `beat_interval_consistency` | **Derived, not reported.** How regular the detected beats are. librosa gives no confidence and a steady grid is the best available proxy — with a known blind spot: a tracker locked onto a steady but *wrong* grid scores just as high. |
| `none` | no estimate was produced. |

`reliable` is `confidence >= DJTI_KEY_MIN_RELIABILITY` (0.35 by default) and
nothing more. Low-confidence results are still returned, with a warning —
hiding them would throw away the only estimate there is.

A high confidence is not the same as an unambiguous one. A track scoring 0.62
for F minor and 0.61 for Ab major is genuinely undecided, and only
`alternatives` can tell you that. Relative major/minor pairs share every note,
so this is the normal case rather than an edge case.

---

## Configuration

Environment variables, prefix `DJTI_`, or a `.env` file. Full list with
defaults in [.env.example](.env.example).

The ones that matter most:

| variable | default | |
|---|---|---|
| `DJTI_KEY_ENGINE` | `auto` | `auto` \| `essentia` \| `chroma` |
| `DJTI_TEMPO_ENGINE` | `auto` | as above |
| `DJTI_KEY_PROFILE` | `edma` | `edma` \| `temperley` \| `krumhansl` |
| `DJTI_KEY_MIN_RELIABILITY` | `0.35` | below this, `reliable: false` |
| `DJTI_KEY_MIN_TONAL_SALIENCE` | `0.01` | below this, no key is claimed at all |
| `DJTI_SEGMENT_WINDOW_SECONDS` | `30.0` | tonal analysis window |
| `DJTI_SEGMENT_HOP_SECONDS` | `15.0` | how far the window slides |
| `DJTI_SEGMENT_MIN_CONFIDENCE` | `0.3` | floor for claiming a key change |
| `DJTI_DJ_BPM_MIN` / `_MAX` | `70` / `180` | the range `dj.mix_bpm` folds into |
| `DJTI_MAX_ANALYSIS_SECONDS` | `0` | analyse only the first N seconds; 0 = all |
| `DJTI_MAX_UPLOAD_BYTES` | `268435456` | 256 MB |
| `DJTI_LOG_FORMAT` | `console` | `console` \| `json` |

---

## Testing

```bash
make test          # everything
make test-fast     # unit tests only, no audio decoding
make check         # lint + typecheck + test
```

227 tests. Ruff and mypy (`strict`) both clean.

**Audio fixtures are generated, not committed.** A repository cannot legally
carry someone's record, and a generated fixture is reproducible and reviewable
in a way a binary blob is not.
[`synth.py`](src/dj_intelligence/synth.py) renders chord progressions, bass
lines and kick patterns in any key at any tempo, plus click tracks, noise and
silence. Listen to them with `make fixtures`.

One thing that took a wrong turn and is worth recording: the first version of
the minor-key fixture used `i-VI-III-VII`, which in A minor spells Am-F-C-G —
and `I-V-vi-IV` in C major spells C-G-Am-F. The *same four chords*. The two
fixtures contained an identical multiset of pitch classes, so no chroma-based
method could ever have told them apart, and the analyser "failing" them was
correct behaviour. What distinguishes a key from its relative is which chord
the music keeps returning to, so the fixtures now hold the tonic for half of
every four bars, as real minor-key dance music does. A fixture that is
genuinely ambiguous tests nothing except a willingness to accept a coin flip.

**What the tests prove**: the chain is wired up, the profiles point the right
way, uncertainty is represented rather than papered over, all 24 Camelot
mappings are right, and a backend crashing does not take the request with it.
**What they do not prove**: accuracy on real records. Synthetic audio has no
mastering, no reverb tail, no bleed, no vocals and a far cleaner spectrum than
anything a producer prints.

---

## Validating against real music

That is what the harness is for.

```csv
file,expected_key,expected_mode,expected_camelot,expected_bpm
track1.mp3,F,minor,4A,126
track2.mp3,A,minor,8A,128
```

```bash
python scripts/evaluate_dataset.py dataset.csv
python scripts/evaluate_dataset.py dataset.csv --engine essentia --json results.json
```

```text
  KEY
    labelled            24
    exact key           100.0%
    camelot exact       100.0%
    camelot compatible  100.0%
    major/minor         100.0%
    relative confusion    0.0%

  TEMPO
    BPM MAE             0.036
    within 1%           100.0%
    within 2%           100.0%
    within 2% (1/2, 2x) 100.0%

  PERFORMANCE
    realtime ratio      0.0208x  (48x faster than playback)
```

*(those numbers are against generated fixtures — the point of the tool is to
produce the equivalent table for your own library)*

Paths resolve relative to the CSV, so a dataset can sit next to the audio.
`expected_camelot` is optional and derived from key and mode when absent —
and cross-checked against them when present, because a dataset that
contradicts itself produces confident nonsense.

**Camelot compatible** counts a detection that is one of the four keys that mix
with the true one. The gap between that and **camelot exact** is the most
useful diagnostic here: a tool that is wrong but harmonically adjacent has
failed much less badly than one that is a tritone out. **Relative confusion**
is broken out separately because F minor vs Ab major is the dominant failure
mode of every chroma-based method and does not belong buried in "wrong".

---

## Performance

A 6-minute track, chroma backend, warm process:

```text
audio                360 s
analysis            3.03 s
ratio            0.0084x real-time   (119x faster than playback)

hash               33 ms
decode            365 ms
tempo             586 ms
tonal content    1042 ms   ← the chromagram
key                 0 ms   ← reuses the chromagram above
segments            3 ms   ← reuses it again
loudness          948 ms   ← a second ffmpeg pass, EBU R128
```

The chromagram is the expensive stage and it is computed **once**, memoised on
the decoded audio, then reused by the gate, the global key and all 23
overlapping segmentation windows. Without that, segmentation alone would cost
23 extractions.

Two things were measured rather than assumed:

- **Harmonic/percussive separation before chroma** is the textbook defence
  against a kick drum smearing the chroma. It costs ~30× the rest of key
  detection and changed no answer — 24/24 synthetic keys either way, same key
  and same confidence on real material. It is therefore *off* by default and
  available as `DJTI_KEY_HARMONIC_SEPARATION=true` for anyone who wants to
  settle it against a labelled library.
- **Median vs mean chroma aggregation** agree on `edma`, but the mean collapses
  to 50% accuracy on `temperley` while the median holds at 100%. Median wins.

**Cold start is not free.** librosa's numba kernels compile on first use, so
the *first* analysis in a fresh process pays roughly 3-4 seconds regardless of
track length — a 24-second file measured 0.168× real-time cold and 0.016×
warm. The server builds its pipeline once and keeps it, so this is a
per-process cost, not a per-request one; it matters for one-shot CLI runs and
disappears in a bulk run.

Nothing is held in memory beyond one track: a 6-minute file is ~63 MB of
float32, and it is released when the request ends.

---

## Limitations

Read this section before trusting a number.

**Relative major/minor confusion.** F minor and Ab major contain exactly the
same notes. Chroma-based key detection distinguishes them only by which pitch
class dominates, and on a track with a busy top line and a quiet root that
signal is weak. This is the single most common error class in every key
detector, ours included, and it is why the Camelot *number* is usually right
even when the letter is not.

**Tracks that change key** get one global answer plus `tonal_segments`. The
global key is whichever reading wins over the whole track, which for a genuine
50/50 modulation is close to arbitrary. Look at the segments.

**Segment boundaries land on the window grid**, not on musical events. This is
a sliding window, not structural segmentation — good enough to notice that a
breakdown lifts to the relative major, not good enough to tell you the bar it
happens on.

**Percussion-heavy and atonal material** returns no key. That is correct, not a
failure, but it does mean a drum tool or a noise record analyses as
`key: null`.

**Non-standard tuning** is handled — the offset is estimated per track — but a
recording that drifts in pitch (tape, unlocked turntables) gets one average
tuning and a smeared chroma.

**Half and double time.** `tempo.bpm` is what the algorithm measured;
`dj.mix_bpm` folds it into 70–180 and says which reading it chose. Neither is
overwritten and both are reported, because no algorithm can tell 87 from 174
from the signal alone — that is a genre convention, not an acoustic fact.

**Beat-grid confidence has a blind spot.** For the librosa backend it measures
how *regular* the beats are, so a tracker locked onto a steady but wrong grid —
half-time, or the off-beat — scores just as high as a correct one.

**Live recordings, long ambient intros and DJ mixes** are all poorly served:
tempo drifts, the first 90 seconds may not represent the track, and a mix has
several keys and tempos by construction.

**No downbeats.** Beat tracking runs; bar-position tracking does not. A wrong
downbeat is worse than no downbeat, so the field is `null`.

**The compatibility score is a rule system**, not a prediction. It knows about
key and tempo and nothing else — not energy, not phrase structure, not whether
either track has vocals.

---

## Roadmap

Ordered roughly by how much each unlocks, and by how confident the underlying
MIR is. Nothing here is implemented; nothing here is faked in the meantime.

**Next, and well-founded**

- Structural segmentation — novelty over a self-similarity matrix — replacing
  the sliding window behind the existing `SegmentKeyAnalyzer` interface.
- Downbeats and phrase boundaries (8/16/32 bars), which most DJ features below
  depend on.
- Energy and spectral character from measurements already taken: band-limited
  RMS over time, spectral centroid and rolloff, kick and bass intensity.
- Intro, outro, breakdown and drop detection, once phrase structure exists.
- Mix-in and mix-out points, and cue-point recommendations, on top of that.
- Consensus confidence: run both backends and treat agreement as evidence.
- Analysis caching keyed on `sha256` + `analysis_version` +
  `configuration_fingerprint` — every field it needs is already recorded.

**Further out**

- Vocal/instrumental detection and vocal activity over time.
- Chord progression, mood, genre and subgenre classification.
- A persistence layer (PostgreSQL) for libraries at scale, with the analysis
  versioning already designed for re-analysis rather than overwriting.
- Ranked "what should I play next", combining harmonic compatibility, tempo,
  energy progression, phrase structure and intro/outro fit.

**Where an LLM fits, and where it does not.** Key and BPM detection are
deterministic DSP and will stay that way — a language model has no access to
the signal and would be guessing. Semantic tags, track descriptions, natural
language crate search and explaining *why* a transition works are a different
problem, and a good fit.
