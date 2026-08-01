r"""
Finite point-group rotations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


def point_group_rotations(
    kind: Literal["cyclic", "dihedral", "t3", "tetrahedral", "octahedral", "icosahedral"],
    order: int | None = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[Tensor, ...]:
    r"""
    Return the canonically ordered rotation matrices of a finite three-dimensional point group.
    """

    if kind in ("cyclic", "dihedral"):
        if order is None:
            raise ValueError(f"{kind} symmetry requires order")
        angles = np.arange(order, dtype=np.float64) * (2 * np.pi / order)
        rotations = [
            np.array(
                (
                    (np.cos(angle), -np.sin(angle), 0.0),
                    (np.sin(angle), np.cos(angle), 0.0),
                    (0.0, 0.0, 1.0),
                )
            )
            for angle in angles
        ]
        if kind == "dihedral":
            flip = np.diag((1.0, -1.0, -1.0))
            rotations = [matrix for rotation in rotations for matrix in (rotation, flip @ rotation)]
        matrices = np.stack(rotations)
    elif kind == "t3":
        matrices = np.asarray(
            (
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
                ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
                ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            )
        )
    else:
        group = {"tetrahedral": "T", "octahedral": "O", "icosahedral": "I"}[kind]
        matrices = Rotation.create_group(group).as_matrix()
        if kind == "icosahedral":
            orientation = np.array(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
            matrices = np.einsum("ij,njk,lk->nil", orientation, matrices, orientation)
        rounded = np.round(matrices, decimals=12)
        canonical_order = sorted(range(len(matrices)), key=lambda index: tuple(rounded[index].reshape(-1)))
        matrices = matrices[np.asarray(canonical_order)]
        permutation = {
            "tetrahedral": (11, 10, 7, 6, 5, 9, 4, 8, 3, 2, 1, 0),
            "octahedral": (23, 19, 18, 21, 14, 15, 11, 10, 16, 22, 20, 17, 3, 2, 13, 7, 5, 8, 12, 6, 9, 1, 0, 4),
            "icosahedral": (
                59,
                58,
                55,
                53,
                57,
                51,
                56,
                50,
                54,
                52,
                47,
                45,
                49,
                43,
                48,
                42,
                46,
                44,
                41,
                35,
                39,
                37,
                38,
                36,
                40,
                34,
                31,
                29,
                33,
                26,
                28,
                30,
                32,
                27,
                0,
                1,
                9,
                3,
                7,
                5,
                6,
                4,
                8,
                2,
                17,
                11,
                15,
                13,
                14,
                12,
                16,
                10,
                23,
                21,
                25,
                19,
                24,
                18,
                22,
                20,
            ),
        }[kind]
        matrices = matrices[np.asarray(permutation)]
    return tuple(torch.as_tensor(matrix, dtype=dtype, device=device) for matrix in matrices)
