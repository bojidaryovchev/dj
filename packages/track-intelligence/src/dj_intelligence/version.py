"""
Versioning of the analysis itself.

Three different versions matter and they change for different reasons:

``ANALYSIS_VERSION``
    The semantics of the numbers we produce. Bump it whenever a change could
    make the same file analyse differently -- a new default key profile, a
    changed confidence formula, a fixed bug in segmentation. Stored results
    carrying an older value are stale and should be re-analysed.

``SCHEMA_VERSION``
    The *shape* of the result document. Bump it when fields move or change
    meaning, so consumers can branch on it.

package version (``pyproject.toml``)
    Distribution packaging. Irrelevant to whether a result is stale.

Keeping the first two out of the package version is deliberate: shipping a
new CLI flag should not invalidate a library of analysed tracks.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from typing import Final

ANALYSIS_VERSION: Final = "1.0.0"
SCHEMA_VERSION: Final = "1.0"


def package_version() -> str:
    try:
        return _package_version("dj-track-intelligence")
    except PackageNotFoundError:  # running from a source checkout
        return "0.0.0+unknown"
