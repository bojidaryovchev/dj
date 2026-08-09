"""Key estimation backends and time-windowed key analysis."""

from .chroma import ChromaKeyAnalyzer
from .essentia import EssentiaKeyAnalyzer, essentia_available
from .profiles import PROFILES, KeyScores
from .segmentation import SlidingWindowKeyAnalyzer

__all__ = [
    "PROFILES",
    "ChromaKeyAnalyzer",
    "EssentiaKeyAnalyzer",
    "KeyScores",
    "SlidingWindowKeyAnalyzer",
    "essentia_available",
]
