"""
Shared fixtures.

Audio fixtures are *generated*, not committed: a repository cannot legally
carry someone's record, and a generated fixture is reproducible and reviewable
in a way a binary blob is not. They are built once per test session because
rendering and encoding them costs more than any single assertion.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from dj_intelligence.config import Settings
from dj_intelligence.music.notes import Mode
from dj_intelligence.synth import (
    Progression,
    render_click,
    render_noise,
    render_progression,
    write_wav,
)

# Long enough that the 30 s segmentation window fits with room to slide.
FIXTURE_SECONDS = 40.0
FIXTURE_BPM = 126.0


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("audio")


@pytest.fixture(scope="session")
def f_minor_wav(audio_dir: Path) -> Path:
    return write_wav(
        audio_dir / "f_minor.wav",
        render_progression(Progression(5, Mode.MINOR, bpm=FIXTURE_BPM, seconds=FIXTURE_SECONDS)),
    )


@pytest.fixture(scope="session")
def c_major_wav(audio_dir: Path) -> Path:
    return write_wav(
        audio_dir / "c_major.wav",
        render_progression(Progression(0, Mode.MAJOR, bpm=FIXTURE_BPM, seconds=FIXTURE_SECONDS)),
    )


@pytest.fixture(scope="session")
def a_minor_wav(audio_dir: Path) -> Path:
    return write_wav(
        audio_dir / "a_minor.wav",
        render_progression(Progression(9, Mode.MINOR, bpm=FIXTURE_BPM, seconds=FIXTURE_SECONDS)),
    )


@pytest.fixture(scope="session")
def modulating_wav(audio_dir: Path) -> Path:
    """A minor for 45 s, then F minor for 45 s. The segmentation fixture."""
    first = render_progression(Progression(9, Mode.MINOR, bpm=FIXTURE_BPM, seconds=45.0))
    second = render_progression(Progression(5, Mode.MINOR, bpm=FIXTURE_BPM, seconds=45.0))
    return write_wav(audio_dir / "modulating.wav", np.concatenate([first, second]))


@pytest.fixture(scope="session")
def silence_wav(audio_dir: Path) -> Path:
    return write_wav(audio_dir / "silence.wav", np.zeros(int(44100 * 10.0), dtype=np.float32))


@pytest.fixture(scope="session")
def noise_wav(audio_dir: Path) -> Path:
    """White noise: no key, no beat. Nothing may be claimed about it."""
    return write_wav(audio_dir / "noise.wav", render_noise(10.0))


@pytest.fixture(scope="session")
def click_wav(audio_dir: Path) -> Path:
    """Percussion only at a known tempo: tempo yes, key no."""
    return write_wav(audio_dir / "click_128.wav", render_click(128.0, 20.0))


@pytest.fixture(scope="session")
def tiny_wav(audio_dir: Path) -> Path:
    """80 ms. Decodes fine, far too short to analyse."""
    return write_wav(audio_dir / "tiny.wav", render_click(120.0, 0.08))


@pytest.fixture(scope="session")
def empty_wav(audio_dir: Path) -> Path:
    return write_wav(audio_dir / "empty.wav", np.zeros(0, dtype=np.float32))


@pytest.fixture
def corrupt_file(tmp_path: Path) -> Path:
    """A file that claims to be an MP3 and is not."""
    path = tmp_path / "corrupt.mp3"
    path.write_bytes(b"ID3 this is not audio, it is a lie" * 40)
    return path


@pytest.fixture
def settings() -> Settings:
    """Defaults, pinned to the portable backend so results do not depend on
    whether Essentia happens to be installed on the machine running tests."""
    return Settings(
        key_engine="chroma",
        tempo_engine="chroma",
        log_level="ERROR",
    )


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging() -> Iterator[None]:
    from dj_intelligence.observability import configure_logging

    configure_logging("ERROR", "console")
    yield
