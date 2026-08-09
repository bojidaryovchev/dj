"""
Measurement.

Analysers here answer "what is in this signal?" and nothing else. They do not
know about the Camelot wheel, they do not know which of two metrical readings
a DJ prefers, and they never import ``dj_intelligence.dj``. The one exception
is :mod:`~dj_intelligence.analysis.pipeline`, which is the aggregator and
whose job is precisely to hand measurements to the interpretation layer.
"""

from .base import (
    Analyzer,
    KeyAnalyzer,
    LoudnessAnalyzer,
    SegmentKeyAnalyzer,
    SupportsChromagram,
    TempoAnalysis,
    TempoAnalyzer,
)
from .pipeline import AnalysisPipeline
from .registry import (
    available_engines,
    build_key_analyzer,
    build_loudness_analyzer,
    build_segment_analyzer,
    build_tempo_analyzer,
)

__all__ = [
    "AnalysisPipeline",
    "Analyzer",
    "KeyAnalyzer",
    "LoudnessAnalyzer",
    "SegmentKeyAnalyzer",
    "SupportsChromagram",
    "TempoAnalysis",
    "TempoAnalyzer",
    "available_engines",
    "build_key_analyzer",
    "build_loudness_analyzer",
    "build_segment_analyzer",
    "build_tempo_analyzer",
]
