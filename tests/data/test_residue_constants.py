from __future__ import annotations

import numpy as np

from ofoldx.data import residue_constants


def test_arg_atom14_geometry() -> None:
    arginine_index = residue_constants.AMINO_ACID_ONE_LETTER_CODES.index("R")
    np.testing.assert_array_equal(
        residue_constants.AMINO_ACID_ATOM14_MASK[arginine_index],
        [1.0] * 11 + [0.0] * 3,
    )

    chi1_frame = residue_constants.AMINO_ACID_RIGID_GROUP_DEFAULT_FRAME[arginine_index, 4]
    np.testing.assert_allclose(chi1_frame[:3, 3], [-0.524, -0.778, -1.209])
    np.testing.assert_allclose(np.linalg.det(chi1_frame[:3, :3]), 1.0)
