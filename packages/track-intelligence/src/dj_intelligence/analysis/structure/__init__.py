"""
Structural analysis: where the track changes.

Evidence-based boundary detection only. The deterministic phrase grid — bars
grouped in eights and sixteens — is arithmetic over the tempo map and lives in
``timeline.navigation``, because it is navigation rather than measurement.
"""

from .novelty import NoveltyStructureAnalyzer

__all__ = ["NoveltyStructureAnalyzer"]
