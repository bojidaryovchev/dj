"""
Reference-signal generator.

Test fixtures for an audio analyser are a licensing problem: you cannot commit
someone's record to a repository. So the integration tests synthesise their
own material with a known key and a known tempo, and this module is that
synthesiser. It ships with the package rather than sitting in ``tests/``
because ``scripts/`` uses it too, and because "generate a signal whose answer
you already know" is a genuinely useful thing to have when adding a backend.

What it produces is a chord progression, a bass line and a kick pattern --
not a sine wave. A single sine has one pitch class and would let a broken
chroma implementation pass; a functional progression exercises the thing that
actually decides a key, which is the *distribution* of pitch classes and the
relationship between them.

**It is not a substitute for real music.** Synthetic audio has no mastering,
no reverb tail, no bleed between instruments, no vocals, no sidechain
pumping, and a spectrum far cleaner than anything a producer would print.
Passing on these fixtures proves the pipeline is wired up and the profiles
point the right way. It proves nothing about accuracy on records --
``scripts/evaluate_dataset.py`` is for that.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from .music.notes import Mode

__all__ = [
    "Groove",
    "GrooveRender",
    "Progression",
    "render_click",
    "render_groove",
    "render_noise",
    "render_progression",
    "write_wav",
]

_A4_MIDI: Final = 69
_A4_HZ: Final = 440.0

# Relative amplitudes of the first four harmonics. Roughly a filtered saw:
# enough overtone energy for a constant-Q transform to lock on, not so much
# that the third harmonic starts voting for the wrong pitch class.
_HARMONICS: Final = ((1, 1.0), (2, 0.5), (3, 0.28), (4, 0.16))

# Scale degrees and chord qualities, in semitones from the tonic, one entry
# per bar.
#
# The tonic chord holds two of every four bars. That is not padding, it is
# the whole point: i-VI-III-VII in A minor spells Am-F-C-G, and I-V-vi-IV in
# C major spells C-G-Am-F. Those are the *same four chords*, so the two
# progressions contain an identical multiset of pitch classes and no
# chroma-based method -- ours, Essentia's, or anyone else's -- can tell them
# apart. What distinguishes a key from its relative is which chord the music
# keeps returning to, so the fixture has to actually do that. Real minor-key
# dance music does it with a tonic bass line hammering under everything.
#
# A fixture that is genuinely ambiguous tests nothing except our willingness
# to accept a coin flip as a pass.
_MINOR_PROGRESSION: Final = (
    (0, Mode.MINOR),
    (0, Mode.MINOR),
    (8, Mode.MAJOR),  # VI
    (10, Mode.MAJOR),  # VII
)
_MAJOR_PROGRESSION: Final = (
    (0, Mode.MAJOR),
    (0, Mode.MAJOR),
    (7, Mode.MAJOR),  # V
    (5, Mode.MAJOR),  # IV
)

_TRIAD: Final = {Mode.MAJOR: (0, 4, 7), Mode.MINOR: (0, 3, 7)}


def midi_to_hz(midi: float) -> float:
    return float(_A4_HZ * 2.0 ** ((midi - _A4_MIDI) / 12.0))


@dataclass(frozen=True, slots=True)
class Progression:
    """A stretch of tonal material with a known answer."""

    tonic_pitch_class: int
    mode: Mode
    bpm: float = 126.0
    seconds: float = 32.0
    with_drums: bool = True


def _envelope(length: int, sample_rate: int) -> np.ndarray:
    """Percussive-ish AD envelope. Sharp enough to give onset detection
    something to find, long enough to sustain a chord."""
    # Clamped to the note: the last note of a render is whatever is left of
    # the buffer, which can be shorter than the attack itself.
    attack = max(1, min(length, int(0.01 * sample_rate)))
    ramp = np.ones(length, dtype=np.float64)
    ramp[:attack] = np.linspace(0.0, 1.0, attack)
    decay = np.exp(-np.linspace(0.0, 3.0, length))
    return ramp * (0.35 + 0.65 * decay)


def _tone(midi: float, length: int, sample_rate: int) -> np.ndarray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    frequency = midi_to_hz(midi)
    wave_out = np.zeros(length, dtype=np.float64)
    for multiple, amplitude in _HARMONICS:
        if frequency * multiple >= sample_rate / 2:
            break
        wave_out += amplitude * np.sin(2 * np.pi * frequency * multiple * time)
    return np.asarray(wave_out * _envelope(length, sample_rate), dtype=np.float64)


def _kick(length: int, sample_rate: int) -> np.ndarray:
    """A pitch-swept sine. Broadband at the transient, which is what a beat
    tracker keys on and what harmonic separation has to remove."""
    time = np.arange(length, dtype=np.float64) / sample_rate
    frequency = 120.0 * np.exp(-time * 40.0) + 45.0
    phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
    return np.sin(phase) * np.exp(-time * 18.0)


def _hat(length: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """Filtered noise burst. Off-beat percussion, to give a phase detector
    something wrong to latch onto if it is not actually looking at bar lines."""
    noise = rng.normal(0.0, 1.0, size=length)
    decay = np.exp(-np.linspace(0.0, 30.0, length))
    # Crude high pass: differencing removes the low end where the kick lives.
    return np.diff(noise, prepend=0.0) * decay


@dataclass(frozen=True, slots=True)
class Groove:
    """
    A rhythmic fixture with a known answer, including a known beat grid.

    Everything a grid algorithm can get wrong has a switch here: tempo that
    ramps or steps, a first downbeat that is not at time zero, an intro with
    no beats in it at all, missing kicks, and off-beat percussion that a naive
    "every transient is a beat" detector would trip over.
    """

    tonic_pitch_class: int = 9
    mode: Mode = Mode.MINOR
    bpm: float = 126.0
    bars: int = 32
    beats_per_bar: int = 4

    bpm_end: float | None = None
    """Linear ramp to this tempo by the last beat."""

    step_at_bar: int | None = None
    step_bpm: float | None = None
    """Sudden tempo change at the start of this bar."""

    lead_in_seconds: float = 0.0
    """Silence before the first beat, so bar 1 is not at timestamp zero."""

    beatless_intro_bars: int = 0
    """Bars of sustained chords with no drums at the start."""

    drop_kick_beats: tuple[int, ...] = ()
    """Beat indices whose kick is omitted."""

    syncopated_percussion: bool = False
    """Add off-beat hats between the beats."""

    downbeat_emphasis: float = 1.7
    """How much louder the bar-one kick is. This is the phase evidence."""


@dataclass(frozen=True, slots=True)
class GrooveRender:
    """Rendered audio plus the ground truth it was built from."""

    samples: np.ndarray
    sample_rate: int
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    bpm_per_beat: np.ndarray
    lead_in_seconds: float

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def nominal_bpm(self) -> float:
        return float(np.mean(self.bpm_per_beat))


def _bpm_per_beat(groove: Groove, total_beats: int) -> np.ndarray:
    if groove.bpm_end is not None:
        return np.linspace(groove.bpm, groove.bpm_end, total_beats)
    if groove.step_at_bar is not None and groove.step_bpm is not None:
        step_beat = groove.step_at_bar * groove.beats_per_bar
        tempos = np.full(total_beats, groove.bpm, dtype=np.float64)
        tempos[step_beat:] = groove.step_bpm
        return tempos
    return np.full(total_beats, groove.bpm, dtype=np.float64)


def render_groove(groove: Groove, sample_rate: int = 44100) -> GrooveRender:
    """
    Render a groove and return its exact beat grid alongside the audio.

    The ground truth is returned rather than recomputed, so a test can measure
    beat-timing error against what was actually synthesised instead of against
    another estimate.
    """
    rng = np.random.default_rng(seed=1000 + groove.tonic_pitch_class)
    total_beats = groove.bars * groove.beats_per_bar
    tempos = _bpm_per_beat(groove, total_beats)
    periods = 60.0 / tempos

    beat_times = groove.lead_in_seconds + np.concatenate([[0.0], np.cumsum(periods[:-1])])
    tail = float(periods[-1]) * 2.0
    total_samples = int((float(beat_times[-1]) + tail) * sample_rate)
    out = np.zeros(total_samples, dtype=np.float64)

    progression = _MINOR_PROGRESSION if groove.mode is Mode.MINOR else _MAJOR_PROGRESSION
    tonic_midi = 60 + groove.tonic_pitch_class
    beatless_beats = groove.beatless_intro_bars * groove.beats_per_bar

    for beat in range(total_beats):
        bar = beat // groove.beats_per_bar
        beat_in_bar = beat % groove.beats_per_bar
        start = int(beat_times[beat] * sample_rate)
        length = min(int(periods[beat] * sample_rate), total_samples - start)
        if length <= 0:
            continue

        offset, quality = progression[bar % len(progression)]
        root = tonic_midi + offset

        # Chords land on the bar, and sustain across it.
        if beat_in_bar == 0:
            chord_length = min(
                int(periods[beat] * groove.beats_per_bar * sample_rate), total_samples - start
            )
            for interval in _TRIAD[quality]:
                out[start : start + chord_length] += 0.26 * _tone(
                    root + interval, chord_length, sample_rate
                )

        in_beatless_intro = beat < beatless_beats
        if not in_beatless_intro:
            out[start : start + length] += 0.42 * _tone(root - 24, length, sample_rate)

            if beat not in groove.drop_kick_beats:
                kick_length = min(int(0.25 * sample_rate), total_samples - start)
                if kick_length > 0:
                    gain = 0.8 * (groove.downbeat_emphasis if beat_in_bar == 0 else 1.0)
                    out[start : start + kick_length] += gain * _kick(kick_length, sample_rate)

            if groove.syncopated_percussion:
                hat_start = start + int(periods[beat] * 0.5 * sample_rate)
                hat_length = min(int(0.06 * sample_rate), total_samples - hat_start)
                if hat_length > 0:
                    out[hat_start : hat_start + hat_length] += 0.22 * _hat(
                        hat_length, sample_rate, rng
                    )

    out += rng.normal(0.0, 1e-4, size=total_samples)
    peak = float(np.max(np.abs(out)))
    samples = (out / peak * 0.89).astype(np.float32) if peak > 0 else out.astype(np.float32)

    return GrooveRender(
        samples=samples,
        sample_rate=sample_rate,
        beat_times=beat_times,
        downbeat_times=beat_times[:: groove.beats_per_bar],
        bpm_per_beat=tempos,
        lead_in_seconds=groove.lead_in_seconds,
    )


def render_progression(spec: Progression, sample_rate: int = 44100) -> np.ndarray:
    """
    Render a four-chord loop in the given key.

    One chord per bar at 4/4, with a root-note bass and, optionally, a
    four-to-the-floor kick.
    """
    rng = np.random.default_rng(seed=spec.tonic_pitch_class * 13 + int(spec.mode is Mode.MAJOR))
    beat_seconds = 60.0 / spec.bpm
    bar_seconds = beat_seconds * 4
    total = int(spec.seconds * sample_rate)
    out = np.zeros(total, dtype=np.float64)

    progression = _MINOR_PROGRESSION if spec.mode is Mode.MINOR else _MAJOR_PROGRESSION
    # Root octave 4 (MIDI 60 = C4), transposed to the tonic.
    tonic_midi = 60 + spec.tonic_pitch_class

    bar = 0
    while bar * bar_seconds < spec.seconds:
        offset, quality = progression[bar % len(progression)]
        start = int(bar * bar_seconds * sample_rate)
        length = min(int(bar_seconds * sample_rate), total - start)
        if length <= 0:
            break

        root = tonic_midi + offset
        for interval in _TRIAD[quality]:
            out[start : start + length] += 0.30 * _tone(root + interval, length, sample_rate)
        # Bass two octaves down, re-struck on every beat so the low end is
        # not a single sustained tone.
        for beat in range(4):
            beat_start = start + int(beat * beat_seconds * sample_rate)
            beat_length = min(int(beat_seconds * sample_rate), total - beat_start)
            if beat_length > 0:
                out[beat_start : beat_start + beat_length] += 0.45 * _tone(
                    root - 24, beat_length, sample_rate
                )

        if spec.with_drums:
            for beat in range(4):
                beat_start = start + int(beat * beat_seconds * sample_rate)
                kick_length = min(int(0.25 * sample_rate), total - beat_start)
                if kick_length > 0:
                    out[beat_start : beat_start + kick_length] += 0.8 * _kick(
                        kick_length, sample_rate
                    )
        bar += 1

    # A whisper of noise: exactly-zero samples between notes are not
    # something any real recording contains, and silence handling should not
    # be exercised accidentally by a fixture.
    out += rng.normal(0.0, 1e-4, size=total)

    peak = np.max(np.abs(out))
    return (out / peak * 0.89).astype(np.float32) if peak > 0 else out.astype(np.float32)


def render_click(bpm: float, seconds: float, sample_rate: int = 44100) -> np.ndarray:
    """A bare click track: tempo with no tonality at all."""
    total = int(seconds * sample_rate)
    out = np.zeros(total, dtype=np.float64)
    interval = 60.0 / bpm
    click_length = int(0.05 * sample_rate)
    position = 0.0
    while position < seconds:
        start = int(position * sample_rate)
        length = min(click_length, total - start)
        if length <= 0:
            break
        out[start : start + length] += _kick(length, sample_rate)
        position += interval
    peak = np.max(np.abs(out))
    return (out / peak * 0.89).astype(np.float32) if peak > 0 else out.astype(np.float32)


def render_noise(seconds: float, sample_rate: int = 44100, seed: int = 7) -> np.ndarray:
    """White noise: no key, no beat. The 'do not guess' fixture."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 0.2, size=int(seconds * sample_rate))).astype(np.float32)


def write_wav(path: Path | str, samples: np.ndarray, sample_rate: int = 44100) -> Path:
    """
    Write mono 16-bit PCM.

    ``wave`` from the standard library, not soundfile: fixtures should not
    depend on the stack they are meant to test.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return destination
