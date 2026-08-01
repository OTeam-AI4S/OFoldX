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

import numpy as np
import pytest

from ofoldx.metrics.diversity import normalized_psd_projection, sequence_vendi_score, vendi_score


def test_vendi_effective_counts_and_empty_collection() -> None:
    assert vendi_score(np.empty((0, 0))) == 0.0
    assert sequence_vendi_score(()) == 0.0
    assert vendi_score(np.ones((4, 4))) == pytest.approx(1.0)
    assert vendi_score(np.eye(4)) == pytest.approx(4.0)
    assert sequence_vendi_score(("AAAA", "AAAA", "AAAA")) == pytest.approx(1.0)


@pytest.mark.parametrize("matrix", [np.array([[1.0, 2.0], [2.0, 1.0]]), np.diag([2.0, 1.0])])
def test_vendi_rejects_an_invalid_kernel(matrix) -> None:
    with pytest.raises(ValueError, match="similarity kernel"):
        vendi_score(matrix)


def test_vendi_rejects_asymmetric_similarities() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        vendi_score(np.array([[1.0, 0.0], [0.5, 1.0]]))


def test_normalized_positive_projection_has_known_rank_one_kernel() -> None:
    # The eigenvectors (1, 1) and (1, -1) have eigenvalues 3 and -1.
    # Its positive part is 1.5 * ones, whose normalized kernel is ones.
    projected = normalized_psd_projection(np.array([[1.0, 2.0], [2.0, 1.0]]))
    np.testing.assert_allclose(projected, np.ones((2, 2)), atol=1e-14)
    assert vendi_score(projected) == pytest.approx(1.0)


def test_sequence_alignment_projection_retains_a_probability_spectrum() -> None:
    raw = np.array([[1.0, -0.8, -0.98], [-0.8, 1.0, -0.8], [-0.98, -0.8, 1.0]])
    assert np.linalg.eigvalsh(raw).min() < -0.7
    with pytest.raises(ValueError, match="positive semidefinite"):
        vendi_score(raw)
    projected = normalized_psd_projection(raw)
    np.testing.assert_allclose(np.diag(projected), 1.0, atol=1e-14)
    assert np.linalg.eigvalsh(projected).min() >= -1e-14
    value = sequence_vendi_score(("A", "A" * 10, "A" * 100))
    assert value == pytest.approx(1.9971016398323447)
    assert 1.0 <= value <= 3.0
    permutation = np.array([2, 0, 1])
    np.testing.assert_allclose(
        normalized_psd_projection(raw[np.ix_(permutation, permutation)]),
        projected[np.ix_(permutation, permutation)],
        atol=1e-14,
    )
