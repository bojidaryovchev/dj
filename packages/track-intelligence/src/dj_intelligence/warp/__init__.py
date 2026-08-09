"""
Applying a warp map to audio.

Separate from ``timeline.warp_map``, which only plans. This package does the
things that touch the world: running a time stretcher, writing a new file, and
checking the result by analysing it again.
"""

from .renderer import WARP_ALGORITHM_VERSION, RenderSegment, WarpRenderer
from .service import WarpOutcome, warp_track
from .verify import grid_error_ms, verify_render

__all__ = [
    "WARP_ALGORITHM_VERSION",
    "RenderSegment",
    "WarpOutcome",
    "WarpRenderer",
    "grid_error_ms",
    "verify_render",
    "warp_track",
]
