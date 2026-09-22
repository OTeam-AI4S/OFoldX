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

"""Behavior of ofoldx.data.residue_constants."""

from __future__ import annotations

import numpy as np

from ofoldx.data import residue_constants


class TestResidueConstants:
    def test_arg_atom14_geometry(self) -> None:
        arginine_index = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("R")
        np.testing.assert_array_equal(
            residue_constants.AMINO_ACID_ATOM14_MASK[arginine_index],
            [1.0] * 11 + [0.0] * 3,
        )

        chi1_frame = residue_constants.AMINO_ACID_RIGID_GROUP_DEFAULT_FRAME[arginine_index, 4]
        np.testing.assert_allclose(chi1_frame[:3, 3], [-0.524, -0.778, -1.209])
        np.testing.assert_allclose(np.linalg.det(chi1_frame[:3, :3]), 1.0)
