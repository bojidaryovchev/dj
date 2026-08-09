"""
Configuration.

Every value that can change a number in the output is here, and the subset
that does so is hashed into ``analysis_fingerprint``. That fingerprint is
recorded with each result, which is what lets a library later answer "was
this track analysed with the settings I am using now?" without storing the
whole configuration next to every row.

Settings are read from the environment (prefix ``DJTI_``) and from a ``.env``
file if one is present. See ``.env.example``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["EngineChoice", "KeyProfile", "Settings", "get_settings"]


class EngineChoice(StrEnum):
    """Which backend to use. ``AUTO`` prefers Essentia and falls back."""

    AUTO = "auto"
    ESSENTIA = "essentia"
    CHROMA = "chroma"


class KeyProfile(StrEnum):
    """
    Key profile used to score a chroma vector against the 24 candidate keys.

    ``EDMA`` comes from Faraldo et al.'s work on key estimation for
    electronic dance music and is the sane default for a DJ tool; the
    classical profiles assume voice-leading conventions that four-to-the-floor
    material simply does not follow. Both backends accept the same names so
    that a comparison between them is a like-for-like one.
    """

    EDMA = "edma"
    TEMPERLEY = "temperley"
    KRUMHANSL = "krumhansl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DJTI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- engines -----------------------------------------------------------
    key_engine: EngineChoice = EngineChoice.AUTO
    tempo_engine: EngineChoice = EngineChoice.AUTO
    key_profile: KeyProfile = KeyProfile.EDMA

    key_harmonic_separation: bool = False
    """
    Run harmonic/percussive separation before extracting chroma.

    Chroma backend only; Essentia does its own front end. Off by default
    because it measured ~30x slower for no change in result. See the module
    docstring of ``analysis/key/chroma.py``.
    """

    # -- decoding ----------------------------------------------------------
    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    """
    Rate the decoder normalises to. 44.1 kHz keeps the whole audible band;
    analysers that want less resample downstream, so nothing is thrown away
    at ingest that a future feature extractor might want.
    """

    max_analysis_seconds: float = Field(default=0.0, ge=0.0)
    """Analyse at most this much audio, from the start. 0 = the whole file."""

    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # -- tonal segmentation ------------------------------------------------
    segments_enabled: bool = True
    segment_window_seconds: float = Field(default=30.0, gt=0.0)
    segment_hop_seconds: float = Field(default=15.0, gt=0.0)
    segment_min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)

    # -- reliability -------------------------------------------------------
    key_min_reliability: float = Field(default=0.35, ge=0.0, le=1.0)

    key_min_tonal_salience: float = Field(default=0.01, ge=0.0, le=1.0)
    """
    How far the chroma must be from flat before any key is claimed.

    Chroma backend only. Guards against reporting a key for material that has
    no pitch content -- noise, percussion, silence -- where profile
    correlation will still happily pick a winner. See
    ``analysis/key/chroma.py`` for the measurements behind the default.
    """
    tempo_min_reliability: float = Field(default=0.35, ge=0.0, le=1.0)
    silence_rms_dbfs: float = Field(default=-60.0, le=0.0)
    min_duration_seconds: float = Field(default=1.0, gt=0.0)

    # -- tempo interpretation ----------------------------------------------
    dj_bpm_min: float = Field(default=70.0, gt=0.0)
    dj_bpm_max: float = Field(default=180.0, gt=0.0)
    tempo_stability_max_cv: float = Field(default=0.04, gt=0.0)

    # -- loudness ----------------------------------------------------------
    loudness_enabled: bool = True

    # -- api ---------------------------------------------------------------
    max_upload_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    host: str = "0.0.0.0"  # noqa: S104 -- containers need to bind all interfaces
    port: int = Field(default=8000, gt=0, lt=65536)

    # -- logging -----------------------------------------------------------
    log_format: Literal["json", "console"] = "console"
    log_level: str = "INFO"

    @field_validator("dj_bpm_max")
    @classmethod
    def _range_is_ordered(cls, value: float, info: Any) -> float:
        minimum = info.data.get("dj_bpm_min")
        if minimum is not None and value <= minimum:
            raise ValueError("dj_bpm_max must be greater than dj_bpm_min")
        return value

    @field_validator("log_level")
    @classmethod
    def _known_level(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"unknown log level: {value}")
        return level

    # -- determinism -------------------------------------------------------

    def analysis_parameters(self) -> dict[str, Any]:
        """
        The settings that can change a measured number.

        Deliberately excludes anything about transport, logging or limits:
        moving the API to another port must not invalidate a library.
        """
        return {
            "key_engine": self.key_engine.value,
            "tempo_engine": self.tempo_engine.value,
            "key_profile": self.key_profile.value,
            "key_harmonic_separation": self.key_harmonic_separation,
            "sample_rate": self.sample_rate,
            "max_analysis_seconds": self.max_analysis_seconds,
            "segments_enabled": self.segments_enabled,
            "segment_window_seconds": self.segment_window_seconds,
            "segment_hop_seconds": self.segment_hop_seconds,
            "segment_min_confidence": self.segment_min_confidence,
            "key_min_reliability": self.key_min_reliability,
            "key_min_tonal_salience": self.key_min_tonal_salience,
            "tempo_min_reliability": self.tempo_min_reliability,
            "silence_rms_dbfs": self.silence_rms_dbfs,
            "tempo_stability_max_cv": self.tempo_stability_max_cv,
            "loudness_enabled": self.loudness_enabled,
        }

    @property
    def analysis_fingerprint(self) -> str:
        """Short stable hash of :meth:`analysis_parameters`."""
        payload = json.dumps(self.analysis_parameters(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings. Analysers take a ``Settings`` argument instead
    of reaching for this, so tests can vary configuration freely."""
    return Settings()
