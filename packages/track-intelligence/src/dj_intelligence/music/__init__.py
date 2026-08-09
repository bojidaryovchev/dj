"""
Music theory. Deterministic, dependency-free, and unaware that audio exists.

Nothing in here measures anything -- it converts between representations a
musician already agrees on (note names, keys, the Camelot wheel) and answers
questions about them. That is why it can be exhaustively unit-tested, and why
the analysis layer is not allowed to import DJ notation from it by accident:
measurement lives in ``dj_intelligence.analysis``, interpretation of
measurements lives in ``dj_intelligence.dj``.
"""

from .camelot import WHEEL_POSITIONS, CamelotKey
from .harmony import HarmonicRelation, HarmonicRelationship, classify, compatible_keys
from .notes import InvalidKeyError, Mode, canonical_key_name, parse_mode, parse_pitch_class

__all__ = [
    "WHEEL_POSITIONS",
    "CamelotKey",
    "HarmonicRelation",
    "HarmonicRelationship",
    "InvalidKeyError",
    "Mode",
    "canonical_key_name",
    "classify",
    "compatible_keys",
    "parse_mode",
    "parse_pitch_class",
]
