"""Ingest: turning files into one normalised signal, plus their identity."""

from .decoder import DecodedAudio, FFmpegDecoder
from .hashing import sha256_file
from .probe import SourceInfo, probe

__all__ = ["DecodedAudio", "FFmpegDecoder", "SourceInfo", "probe", "sha256_file"]
