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

Key, Camelot, tempo, bars and downbeats, tempo drift, a beat grid you can
navigate in musical time, optional grid correction by time-stretching, tonal
segmentation and loudness. The architecture is built for what comes after that
— energy, cue points, "what should I play next" — without pretending any of it
exists yet.

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [How key detection works](#how-key-detection-works)
- [Beat grids, warping and quantized navigation](#beat-grids-warping-and-quantized-navigation)
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
| **Bars & downbeats** | which beat is beat one, and how many beats a bar has |
| **Tempo drift** | local tempo through the track, and whether it can be called constant |
| **Beat grid** | every beat indexed to a bar and a position within it, with per-region confidence |
| **Warping** | a plan to move a drifting grid onto a constant one, and a renderer that applies it with pitch preserved |
| **Navigation** | snap to beat/bar, jump 8/16/32 bars, phrase grid, quantized scheduling |
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
        |    rhythm stage         |   grid phase · bars · meter
        |                         |   · local tempo · drift
        |    segment analyzer     |   key over time
        |    structure analyzer   |   where the music changes
        |    loudness analyzer    |   LUFS · LRA · true peak
        +------------+------------+
                     |
        +------------v------------+
        |  musical time           |   audio seconds <-> musical beats
        |  timeline/              |   TempoMap · Navigator · WarpMap
        +------------+------------+
                     |
        +------------v------------+
        |  interpretation         |   "what should a DJ do about it?"
        |  dj/                    |   Camelot · neighbours · mix BPM
        +------------+------------+   · compatibility · warp advice
                     |
              TrackAnalysis
                     |
        +------------v------------+   optional, explicit, never automatic
        |  warp/                  |   Rubber Band render + verification
        +------------+------------+
                     |
           grid-corrected audio
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
│   ├── rhythm/        downbeats, grid phase, tempo curve, grid assembly
│   ├── structure/     self-similarity novelty boundaries
│   └── loudness/      EBU R128 via ffmpeg
├── timeline/        musical time: TempoMap, Navigator, WarpMap (pure maths)
├── warp/            applying a warp map: Rubber Band render, verification
├── music/           theory: notes, Camelot, harmonic relationships
├── dj/              interpretation: Camelot view, compatibility, warp advice
├── models/          the canonical result document
├── api/             FastAPI: routes, uploads, error translation
├── cli.py           dj-analyze
├── reporting.py     terminal rendering for the three report layouts
├── engine.py        the single entry point both surfaces use
├── config.py        every knob, and the fingerprint over them
└── synth.py         reference-signal generator for tests
```

``timeline/`` is a third pure layer alongside ``music/``: it measures nothing
and touches no audio, it just maps between audio seconds and musical beats.
Both the analysis and the DJ layers use it, which is why it belongs to
neither.

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

## Beat grids, warping and quantized navigation

This is the part that turns "we know the BPM" into "we know where every bar
starts". Six things are involved and they are easy to run together, so here is
each one separately, in the order they depend on each other.

```text
SOURCE AUDIO
     |
BEAT DETECTION          where are the pulses?
     |
GRID PHASE CORRECTION   they are all ~30 ms late; fix that
     |
BAR / DOWNBEAT DETECTION  which pulse is beat one?
     |
TEMPO MAP               audio time <-> musical time, both ways
     |
WARP MAP                which source instants belong on which grid instants
     |
OPTIONAL TIME STRETCH   actually move the audio (pitch preserved)
     |
GRID-CORRECTED AUDIO
```

and, separately, the reason the tempo map exists at all:

```text
TEMPO MAP
    |
MUSICAL POSITION        "bar 65, beat 1"
    |
QUANTIZED SEEK / JUMP   +16 bars, next phrase, snap to bar
```

### 1. Beat detection — *where are the pulses?*

A list of times. Nothing more: no bars, no phase, no musical meaning.

```text
0.000  0.477  0.953  1.430  ...
```

### 2. The beat grid — *which musical beat is each pulse?*

Detection gives times; the grid gives each time a **musical index**, and that
is a different thing. If the tracker misses a beat, the next detection is
still musical beat 5 — numbering the survivors 0,1,2,3,4 shifts every bar line
after the gap by one for the rest of the track. Indices are assigned by
accumulating rounded interval counts, never by ordinal position.

Conventions, fixed everywhere:

| | |
|---|---|
| `index` | zero-based, counts musical beats |
| `bar` | zero-based, counted from the first downbeat; negative before it |
| `beat_in_bar` | **one**-based, because that is how people count |

**Grid phase.** Beat trackers report beats systematically *late*: the onset
envelope they work from is a spectral flux, which peaks after the transient
that caused it. Measured against fixtures with a known grid, librosa's beats
land **+29.8 ms late with a standard deviation of only 7.1 ms** — the spacing
is good, the phase is wrong. So one offset is estimated robustly and applied
to the whole grid, which is what DJ software calls the grid offset. Per-beat
snapping was tried and rejected: it cut the bias but raised the jitter from
7 ms to 10 ms, for a worse result overall.

| | mean error | p95 |
|---|---|---|
| raw librosa beats | 29.8 ms | 40.5 ms |
| per-beat snapping | 11.5 ms | 25.8 ms |
| **one global offset** | **9.5 ms** | **19 ms** |

### 3. Downbeats and meter — *which beat is beat one?*

Beat tracking cannot answer this, which is why the previous version reported
`downbeats: null`. Calling every fourth beat a downbeat is not detection, it
is a coin flip with four sides — and it is wrong three times in four on any
track whose first detected beat is not a bar line.

Three beat-synchronous features vote, each peaking on bar lines for a
different reason: **low-band energy** (the bar-one kick), **onset strength**
(new material enters on bar lines) and **harmonic change** (chords change on
bar lines — Goto's heuristic, and the one that rescues tracks with a perfectly
uniform kick). Each is standardised, then every (beats-per-bar, phase)
hypothesis is scored by how far above average its downbeats sit.

The confidence is the margin over the runner-up **divided by its standard
error**, which matters more than it sounds: how big a margin chance produces
depends entirely on how many beats went into each average. Ten seconds of
white noise gives six beats per phase, where a "0.4 SD win" is pure luck;
across 48 beats per phase the same margin is overwhelming. Without that
normalisation, noise reported a confident 4/4.

Meter is decided on the same terms — 3/4 is evaluated, not assumed away — and
`beats_per_bar` is `null` when the evidence does not support a choice. A DJ
workflow can opt into 4/4 with `DJTI_FALLBACK_BEATS_PER_BAR=4`; the
measurement layer never guesses.

### 4. Tempo drift — *can this be treated as constant?*

Local tempo is fitted by least squares over sliding windows of the beat grid
(64 beats, hopping 32) — fitted to the beats already found, never by
re-running a tracker per window, which would let each window pick its own
metrical level and invent a curve that jumps between 63 and 126 BPM.

Classification is on the relative range of local tempo, and the thresholds sit
above the measurement noise floor so that a sequenced track is not flagged for
estimator jitter:

| classification | relative range |
|---|---|
| `stable` | < 0.2% |
| `minor_drift` | < 1.0% |
| `variable_tempo` | < 3.0% |
| `highly_variable` | >= 3.0% |
| `unknown` | too few beats |

The metrics behind the label — `local_bpm_min`, `local_bpm_max`,
`relative_percent`, `max_absolute_bpm_delta` — are always reported, so you can
apply your own threshold instead.

### 5. The tempo map — *audio time <-> musical time*

The primitive everything else is built on, in both directions and fractional:

```python
tempo_map.time_to_beat(91.243)  # -> 192.42
tempo_map.beat_to_time(192)  # -> 91.107
tempo_map.time_to_bar(91.243)  # -> bar 48 beat 1
tempo_map.bar_to_time(48)  # -> 91.107
tempo_map.local_bpm(91.243)  # -> 126.04
```

Piecewise linear between detected beats, extrapolating at the tempo of the
nearest one — so it follows a drifting record instead of averaging it, and
positions before the first beat or after the last are still real positions
rather than errors.

### 6. Warp markers — *where should the audio actually be?*

A warp map is a **plan**, not an edit. Beat *b* belongs at
`anchor + (b - anchor_beat) x 60/target_bpm`, where the anchor is the first
reliable downbeat, so bar lines stay put and the intro is not squeezed to make
room for a timeline that starts at zero.

```json
{
  "target_bpm": 126.0,
  "markers": [
    { "source_time": 1.428,  "source_beat": 0,  "target_time": 1.428 },
    { "source_time": 16.673, "source_beat": 32, "target_time": 16.666 },
    { "source_time": 31.889, "source_beat": 64, "target_time": 31.904 }
  ]
}
```

**Markers are minimised, not maximised.** A marker per beat pins the
detector's noise into the render, forces a different stretch ratio on every
half second of audio, and smears the transients it was supposed to protect.
Instead the source-to-target curve is simplified greedily: extend from the last
marker until linear interpolation would put some beat more than
`max_grid_error_ms` (10 ms) from where it belongs, then plant the next one. A
400-beat drifting track needs **6 markers**, not 400.

Two more defences against warping noise instead of drift:

* the source grid is **smoothed** with a moving least-squares fit before
  planning, so markers follow tempo rather than jitter;
* the decision uses **`systematic_error_ms`** — the error of the *smoothed*
  grid — not the raw maximum. On a perfect track the raw worst beat is 48 ms
  out from detector jitter alone, and deciding on that number had the engine
  recommending warps for sequenced tracks.

### 7. Audio warping — *actually moving it*

Rubber Band, through ffmpeg's `rubberband` filter. Time changes, pitch does
not; resampling is never used because it changes both. Each span between two
markers is rendered with **its own ratio** — genuine variable warping, not one
global ratio that would leave the drift in place — and the spans are assembled
onto the target timeline with an equal-power crossfade at the joins
(`DJTI_WARP_CROSSFADE_MS`, default 8 ms) so that a ratio change cannot click.
Every segment is trimmed or padded to the exact sample count its target span
demands, because a stretcher that returns three samples too many is normal and
letting that compound across forty segments puts the end of the track a beat
late.

Rubber Band's own CLI takes a time map directly (`rubberband --timemap`) and
would do this in one pass with no joins at all. It is not installed everywhere
— notably not on Windows — so the segment renderer is the portable path and
drives the same library underneath.

**The source file is never modified.** Output is a new lossless artifact.

### 8. Never over-correcting

Most modern electronic music is sequenced and already on a perfect grid;
stretching it trades an inaudible timing gain for real transient damage. So
warping is refused unless it earns its place:

| refusal | why |
|---|---|
| `source_grid_already_within_tolerance` | systematic error under 15 ms |
| `no_reliable_beat_grid` | grid confidence under 0.5 — a wrong grid warped confidently is the worst outcome available |
| `tempo_interpretation_not_trusted` | half/double-time protection: warping a 128 track read as 64 would stretch it to twice its length |
| `required_stretch_exceeds_safe_threshold` | a 0.9-1.1 ratio is a loose recording; outside it, the grid or the metrical level is wrong |

`--force` overrides, and an explicitly requested `--target-bpm` is never
subject to the "already close enough" test — a user asking for 128 does not
want to hear that 126 is tidy.

### 9. Verification — *did it actually work?*

Rendering audio and assuming it worked is not a pipeline, it is a hope. The
rendered file is analysed again from scratch and its beats are compared with
the grid the warp aimed at, alongside the same measurement on the input.

```text
Verification
before       130.9 ms mean
after          8.6 ms mean
p95           20.7 ms
max           23.7 ms
improvement   15.2x
result        PASSED
```

### 10. Quantized navigation

Pure timeline maths over a tempo map — no audio, no playback, no state.

```python
navigator.snap(92.131, Unit.BAR, Direction.NEXT)  # -> 92.929 s, bar 48 beat 1
navigator.jump_bars(91.227, 16)  # -> 121.703 s, bar 63 beat 1
navigator.jump_beats(current, -4)
navigator.next_boundary(current, bars=16)
navigator.phrase_grid(duration, phrase_bars=8)
```

**Jumps preserve rhythmic phase.** Moving 16 bars from 40% through beat 3
lands 40% through beat 3 of the destination bar. Landing on the bar line
instead would silently shift a loop every time it repeated.

Bar- and phrase-level operations raise if downbeat detection established no
phase, rather than assuming 4/4 and putting every jump in the wrong place.
Beat-level operations always work.

**Deferred actions.** A deck needs a jump requested mid-bar to happen on the
bar line. `schedule()` returns the timing and nothing else:

```json
{
  "requested_at": 91.320,
  "execute_at": 92.929,
  "destination": 121.703,
  "quantization": "bar"
}
```

Both ends are quantised — executing on the bar line but landing mid-bar would
still break phase. What a player does with those numbers, including the short
crossfade that stops a seek clicking, is left to the player.

### 11. Phrase grid vs structural boundaries

Two different things, deliberately not merged.

The **phrase grid** is deterministic: bars grouped in fours, eights, sixteens
or thirty-twos from the first downbeat. It is arithmetic and it is always right
if the grid is right. It is what "jump 16 bars" means.

**Structural boundaries** are evidence-based — a self-similarity matrix over
beat-synchronous MFCCs and chroma, with a Foote checkerboard kernel run down
its diagonal — and they land where the music actually changes, usually but not
always on a phrase boundary.

Every boundary is reported as `structural_boundary` and nothing more. Novelty
detection finds *that* something changed, never *what*; calling one of them
"the drop" needs a classifier, and inventing the label without one would put
confident nonsense into a library.

### Analysis profiles

Rhythmic and structural analysis cost real time, and not every caller needs
them.

| profile | what runs |
|---|---|
| `basic` | decode, hash, tempo, key, loudness |
| `full` *(default)* | + downbeats, meter, tempo curve, grid, tonal segments, structure |
| `warp` | + tempo map, warp map, warp recommendation |

```bash
dj-analyze track.mp3 --depth basic
curl -F "file=@track.mp3" -F "profile=basic" localhost:8000/v1/tracks/analyze
```

### CLI

```bash
dj-analyze grid track.mp3            # bars, meter, drift, warp advice
dj-analyze grid track.mp3 --beats    # every beat with its bar position
dj-analyze grid track.mp3 --json

dj-analyze warp track.mp3 --dry-run             # plan only, write nothing
dj-analyze warp track.mp3 --target-bpm 126 -o corrected.wav
dj-analyze warp track.mp3 --force               # override the refusal
dj-analyze warp track.mp3 --no-verify           # skip re-analysing the output
```

### API

```bash
# Plan: always JSON, cheap, the sensible first call.
curl -X POST -F "file=@track.mp3" localhost:8000/v1/tracks/warp/plan

# Render: always audio. 409 when the track does not need it.
curl -X POST -F "file=@track.mp3" -F "target_bpm=126" \
     -o corrected.wav localhost:8000/v1/tracks/warp/render
```

`/render` puts the verification summary in `X-Warp-Verification`, plus
`X-Warp-Markers`, `X-Warp-Target-BPM` and `X-Warp-Stretch-Range`, so a client
gets them without a second request.

`POST /v1/tracks/analyze` is unchanged and now carries `rhythm`, `structure`
and (on the `warp` profile) `warp`. The flat `beats` list and the previously
always-null `downbeats` are still there and are now both populated.

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

333 tests. Ruff and mypy (`strict`) both clean.

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

### Rhythm, bars and warping

**Beat timing is good to about 10 ms, not 1 ms.** After grid-phase correction
the mean error against a known grid is ~9.5 ms with a p95 near 19 ms. That is
inside the range where two percussive sources still read as one hit, but it is
not sample-accurate, and the residual is at the resolution limit of the
window needed to see a kick drum at all.

**Downbeat phase can be wrong, and when it is, it is wrong by a whole bar.**
The detector needs bar lines to be marked by *something* — a louder kick, new
material, a chord change. Music with a perfectly uniform bar (a static loop, a
long tonal drone over an unvarying pattern) gives it nothing to find, and it
reports low confidence rather than guessing. A wrong phase is invisible in the
BPM and ruins every bar jump and phrase boundary, so check
`rhythm.meter.confidence` before trusting bar numbers.

**Live drummers and rubato are described, not fixed.** The tempo curve will
follow a human performance and classify it `variable_tempo` or
`highly_variable`, and warping such a track to a constant grid is usually the
wrong thing to want — it removes the performance. The safety limits refuse
corrections beyond 0.9-1.1x for that reason.

**Odd meters are barely supported.** Only 3 and 4 beats per bar are evaluated.
A 7/8 track will get no meter or a wrong one; the data model represents any
`beats_per_bar`, but the detector does not look for them.

**Half/double-time ambiguity now matters more.** A tempo read at the wrong
metrical level produces a *plausible* grid and a warp that stretches the track
to twice or half its length. The warp layer refuses when the tempo estimate is
not marked reliable, but a confidently wrong half-time reading would still get
through — this is the failure mode most worth checking against a real library.

**Structural boundaries are boundaries, not sections.** They mark where the
music changes; nothing names them, and the count varies with material. Do not
build "jump to the drop" on them yet.

**Warping is lossy.** Rubber Band is very good and it is still a phase
vocoder: a heavily warped track will not sound identical to the original.
The engine's default is therefore to refuse, and the metrics — marker count,
stretch range, max correction — exist so an unusual correction is visible
before it is rendered.

---

## Roadmap

Ordered roughly by how much each unlocks, and by how confident the underlying
MIR is. Nothing here is implemented; nothing here is faked in the meantime.

**Done since the first version**

- Downbeats, bars and meter — see
  [beat grids](#beat-grids-warping-and-quantized-navigation).
- Local tempo, drift classification, and a beat grid with per-region
  confidence.
- The tempo map, and beat/bar/phrase navigation built on it.
- Warp markers, Rubber Band rendering and self-verification.
- Structural boundaries from self-similarity novelty, and the phrase grid.

**Next, and well-founded**

- Energy and spectral character from measurements already taken: band-limited
  RMS over time, spectral centroid and rolloff, kick and bass intensity. The
  beat grid makes these per-bar rather than per-second, which is what makes
  them comparable across tracks.
- Intro, outro, breakdown and drop detection: the boundaries exist, what is
  missing is a classifier to name the sections between them.
- Mix-in and mix-out points, and cue-point recommendations, on top of that.
- Structural segmentation of *key* — replacing the sliding window behind the
  existing `SegmentKeyAnalyzer` with the boundaries the structure analyser
  already finds.
- One-pass warping through `rubberband --timemap`, removing the segment joins
  entirely where the CLI is installed.
- Consensus confidence: run both backends and treat agreement as evidence.
  Now more valuable than before — the two engines can disagree about downbeat
  phase, and that disagreement is itself a useful signal.
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
