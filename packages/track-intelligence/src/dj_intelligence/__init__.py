"""
DJ Track Intelligence -- audio analysis for DJs.

The one-liner:

    from dj_intelligence import analyze
    result = analyze("track.mp3")
    print(result.dj.camelot, result.tempo.bpm)

Everything else in the package is that call taken apart: ``audio`` decodes,
``analysis`` measures, ``music`` knows theory, ``dj`` interprets, ``models``
is the shape of the answer.
"""

from .config import Settings, get_settings
from .engine import analyze, get_pipeline
from .models import TrackAnalysis
from .version import ANALYSIS_VERSION, SCHEMA_VERSION, package_version

__all__ = [
    "ANALYSIS_VERSION",
    "SCHEMA_VERSION",
    "Settings",
    "TrackAnalysis",
    "analyze",
    "get_pipeline",
    "get_settings",
    "package_version",
]
