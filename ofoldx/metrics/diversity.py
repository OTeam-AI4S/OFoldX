# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
Vendi diversity scores over similarity matrices and sequences.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .sequence import sequence_alignment_score


def vendi_score(similarity: NDArray) -> float:
    r"""
    Return the Vendi effective count of a unit-diagonal PSD similarity kernel.

    Meaningful negative eigenvalues are outside this metric's domain. Only
    eigensolver roundoff is clipped. An empty collection has effective count zero.
    """

    similarity = np.asarray(similarity, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(similarity)
    if not len(similarity):
        return 0.0
    tolerance = 32 * np.finfo(np.float64).eps * len(similarity) * max(1.0, float(np.max(np.abs(similarity))))
    if not np.allclose(similarity, similarity.T, rtol=0.0, atol=tolerance):
        raise ValueError("Vendi score requires a symmetric similarity kernel")
    if not np.allclose(np.diag(similarity), 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("Vendi score requires a unit-diagonal similarity kernel")
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
        raise ValueError("Vendi score requires a positive semidefinite similarity kernel")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    eigenvalues /= eigenvalues.sum()
    nonzero = eigenvalues > 0
    return float(np.exp(-np.sum(eigenvalues[nonzero] * np.log(eigenvalues[nonzero]))))


def normalized_psd_projection(similarity: NDArray) -> NDArray:
    r"""Project symmetric similarities onto a normalized positive semidefinite kernel.

    For ``S = U diag(lambda) U.T``, construct ``P = U diag(max(lambda, 0)) U.T``
    and return ``D**(-1/2) P D**(-1/2)``, where ``D = diag(P)``. This explicitly
    defines a new kernel; it is not the Vendi score of the unprojected similarities.
    The alignment/TM callers provide a unit self-similarity, so ``diag(P)`` is positive.
    """

    similarity = np.asarray(similarity, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(similarity)
    if not len(similarity):
        return similarity.copy()
    tolerance = 32 * np.finfo(np.float64).eps * len(similarity) * max(1.0, float(np.max(np.abs(similarity))))
    if not np.allclose(similarity, similarity.T, rtol=0.0, atol=tolerance):
        raise ValueError("PSD projection requires symmetric similarities")
    projected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    diagonal = np.diag(projected)
    if not np.isfinite(projected).all() or np.any(diagonal <= 0):
        raise ValueError("normalized PSD projection requires positive projected self-similarity")
    scale = np.sqrt(diagonal)
    return projected / (scale[:, None] * scale[None, :])


def sequence_vendi_score(sequences: Sequence[str]) -> float:
    r"""
    Return Vendi diversity of the normalized PSD-projected alignment kernel.
    """

    similarity = np.eye(len(sequences), dtype=np.float64)
    for first in range(len(sequences)):
        for second in range(first + 1, len(sequences)):
            similarity[first, second] = similarity[second, first] = sequence_alignment_score(
                sequences[first], sequences[second]
            )
    return vendi_score(normalized_psd_projection(similarity))
