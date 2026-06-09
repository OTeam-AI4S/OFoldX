r"""
LigandMPNN atom-featurization reference tables.

The periodic group/period lookup and the side-chain atom-type table used to build LigandMPNN's per-atom
input features. These are LigandMPNN-specific featurization data, so they live with the model family
rather than in the shared [`data`][ofoldx.data] package.
"""

from __future__ import annotations

import torch
from torch import Tensor


def periodic_group_period() -> tuple[Tensor, Tensor]:
    r"""
    Build the periodic-table group/period lookup used for atom element features.

    LigandMPNN one-hot-encodes each context atom by its periodic group and period; this returns the two
    lookup tables indexed by atomic number so ``group[z]`` / ``period[z]`` give the placement of element
    ``z``. Index ``0`` (no atom) and any atomic number past the tabulated main-group elements map to ``0``.

    Returns:
        Tuple ``(group, period)`` of two `[120]` long tensors indexed by atomic number.

    Examples:
        >>> group, period = periodic_group_period()
        >>> group.shape
        torch.Size([120])
        >>> (group[6].item(), period[6].item())  # carbon
        (14, 2)
        >>> (group[8].item(), period[8].item())  # oxygen
        (16, 2)
    """

    groups = [0] * 120
    periods = [0] * 120
    rows = [
        (1, [1, 18]),
        (2, [1, 2, 13, 14, 15, 16, 17, 18]),
        (3, [1, 2, 13, 14, 15, 16, 17, 18]),
        (4, list(range(1, 19))),
        (5, list(range(1, 19))),
        (6, [1, 2] + [3] * 15 + list(range(4, 19))),
        (7, [1, 2] + [3] * 15 + list(range(4, 19))),
    ]
    atomic_number = 1
    for period, row_groups in rows:
        for group in row_groups:
            groups[atomic_number] = group
            periods[atomic_number] = period
            atomic_number += 1
    return torch.tensor(groups, dtype=torch.long), torch.tensor(periods, dtype=torch.long)


def side_chain_atom_types() -> Tensor:
    r"""
    Return the element type of each atom14 side-chain slot, flattened across residue types.

    Used when side-chain context is enabled: the gathered neighbor side-chain atoms are tagged with these
    atomic numbers (6=carbon, 7=nitrogen, 8=oxygen, 16=sulfur) so they can be featurized like ligand atoms.

    Returns:
        A `[32]` long tensor of atomic numbers for the tabulated side-chain atom slots.

    Examples:
        >>> types = side_chain_atom_types()
        >>> types.shape
        torch.Size([32])
        >>> types[:5].tolist()
        [6, 6, 6, 8, 8]
    """

    return torch.tensor(
        [6, 6, 6, 8, 8, 16, 6, 6, 6, 7, 7, 8, 8, 16, 6, 6, 6, 6, 7, 7, 7, 8, 8, 6, 7, 7, 8, 6, 6, 6, 7, 8],
        dtype=torch.long,
    )
