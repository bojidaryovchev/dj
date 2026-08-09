"""
DJ interpretation.

Everything here reads measurements and applies convention: Camelot notation,
harmonic neighbours, the tempo a DJ would beatmatch to, how well two tracks
go together. Nothing here touches audio, and nothing in
``dj_intelligence.analysis`` imports from this package -- the dependency runs
one way only, which is what keeps "what is in the signal" separable from
"what a DJ should do about it".
"""

from .compatibility import RULES_VERSION, ScoringRules, score_pair
from .interpret import camelot_for, interpret, preferred_mix_bpm
from .warp_advice import WarpAdviceRules, recommend_warp

__all__ = [
    "RULES_VERSION",
    "ScoringRules",
    "WarpAdviceRules",
    "camelot_for",
    "interpret",
    "preferred_mix_bpm",
    "recommend_warp",
    "score_pair",
]
