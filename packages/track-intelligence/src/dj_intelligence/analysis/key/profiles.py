"""
Key profiles and the correlation that turns chroma into a key.

The idea is Krumhansl and Schmuckler's, from 1990, and it has outlived every
attempt to replace it for this particular job: take a 12-element vector of
how much energy sits on each pitch class, compare it against a template for
each of the 24 keys, and pick the best fit. What changed since 1990 is the
templates -- profiles derived from probe-tone experiments on Western
classical listeners do not describe a techno record.

Profiles here are all published:

``krumhansl``
    Krumhansl & Kessler (1982) probe-tone ratings. The original. Assumes
    classical voice leading.

``temperley``
    Temperley (2001), fitted to the Kostka-Payne corpus. Flatter than
    Krumhansl-Kessler, less confused by ornamentation.

``edma``
    Faraldo, Gómez, Jordà & Herrera (2016), fitted to a corpus of electronic
    dance music. The default here, because the corpus is the one this tool
    analyses: strong tonic and fifth, weak leading tone, and a minor profile
    that expects the natural minor rather than the harmonic minor.

Essentia exposes profiles under the same names, so switching backends
compares like with like at the profile level. The two will still not agree
exactly -- they build their chroma differently, and that front-end matters as
much as the template.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ...music.notes import Mode

__all__ = ["PROFILES", "KeyScores", "score_chroma"]

# Each profile is (major, minor), 12 weights starting at the tonic.
PROFILES: Final[dict[str, tuple[tuple[float, ...], tuple[float, ...]]]] = {
    "krumhansl": (
        (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88),
        (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17),
    ),
    "temperley": (
        (5.00, 2.00, 3.50, 2.00, 4.50, 4.00, 2.00, 4.50, 2.00, 3.50, 1.50, 4.00),
        (5.00, 2.00, 3.50, 4.50, 2.00, 4.00, 2.00, 4.50, 3.50, 2.00, 1.50, 4.00),
    ),
    "edma": (
        (
            0.16519551,
            0.04749026,
            0.08293076,
            0.06687112,
            0.09994645,
            0.09274123,
            0.05294487,
            0.13159476,
            0.05229297,
            0.07141712,
            0.06457878,
            0.07199619,
        ),
        (
            0.17235348,
            0.04429327,
            0.07575704,
            0.10396779,
            0.05586074,
            0.08849485,
            0.05942666,
            0.13979822,
            0.08966614,
            0.05036733,
            0.07374028,
            0.06070141,
        ),
    ),
}


class KeyScores:
    """
    Correlation of one chroma vector against all 24 keys.

    Built once per profile and reused, because the expensive part -- mean
    centring and normalising the 24 templates -- does not depend on the
    audio. Scoring a window is then a single matrix-vector product, which is
    what makes per-window segmentation cheap enough to run by default.
    """

    __slots__ = ("_modes", "_templates", "profile_name")

    def __init__(self, profile_name: str) -> None:
        try:
            major, minor = PROFILES[profile_name]
        except KeyError as exc:
            known = ", ".join(sorted(PROFILES))
            raise ValueError(f"unknown key profile {profile_name!r}; known: {known}") from exc

        self.profile_name = profile_name

        # Row i of `templates` is the profile rotated so that pitch class i is
        # the tonic; rows 0-11 major, 12-23 minor. Pre-centred and scaled to
        # unit norm so that a dot product with a centred chroma vector *is*
        # the Pearson correlation.
        rows = []
        modes = []
        for mode, profile in ((Mode.MAJOR, major), (Mode.MINOR, minor)):
            base = np.asarray(profile, dtype=np.float64)
            for tonic in range(12):
                rotated = np.roll(base, tonic)
                centred = rotated - rotated.mean()
                norm = np.linalg.norm(centred)
                rows.append(centred / norm if norm else centred)
                modes.append(mode)

        self._templates = np.vstack(rows)  # (24, 12)
        self._modes: tuple[Mode, ...] = tuple(modes)

    def correlate(self, chroma: np.ndarray) -> np.ndarray:
        """
        Pearson correlation of ``chroma`` (12,) with each of the 24 keys.

        Returns a (24,) array in the same row order as the templates: index
        ``i`` is major with tonic ``i`` for ``i < 12``, minor with tonic
        ``i - 12`` above that. A flat chroma vector (silence, white noise)
        correlates with nothing and returns all zeros rather than dividing by
        zero.
        """
        vector = np.asarray(chroma, dtype=np.float64).reshape(12)
        centred = vector - vector.mean()
        norm = np.linalg.norm(centred)
        if norm == 0 or not np.isfinite(norm):
            return np.zeros(24)
        return np.asarray(self._templates @ (centred / norm), dtype=np.float64)

    def mode_of(self, index: int) -> Mode:
        return self._modes[index]

    @staticmethod
    def pitch_class_of(index: int) -> int:
        return index % 12


def score_chroma(chroma: np.ndarray, profile_name: str) -> np.ndarray:
    """Convenience wrapper for one-off scoring; prefer reusing a
    :class:`KeyScores` when scoring many windows."""
    return KeyScores(profile_name).correlate(chroma)
