r"""
Coordinates aligned to exact atom occurrences in a MolecularSystem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .molecular import AtomSite, ComponentReference, MolecularSystem


@dataclass(frozen=True)
class ResidueNumber:
    r"""
    Residue numbering retained from a structure source.

    Args:
        number: Integer residue number.
        insertion_code: Optional insertion code.
    """

    number: int
    insertion_code: str = ""


class Structure:
    r"""
    Coordinates for selected atom occurrences in a molecular system.

    ``sites[i]`` identifies the atom represented by ``coordinates[i]``,
    ``coordinate_mask[i]``, and, when present, ``occupancy[i]`` and
    ``b_factor[i]``.

    Args:
        system: Coordinate-free molecular system containing every referenced atom.
        sites: Unique atom occurrences in coordinate order.
        coordinates: Real array with shape ``(num_atoms, 3)``. Entries whose mask is
            false are not observed.
        coordinate_mask: Boolean array with shape ``(num_atoms,)`` indicating observed
            coordinates. At least one coordinate must be observed.
        occupancy: Optional real array with shape ``(num_atoms,)``.
        b_factor: Optional real array with shape ``(num_atoms,)`` containing isotropic
            B factors.
        residue_numbers: Optional source numbering keyed by exact component occurrence.

    Raises:
        ValueError: If aligned arrays have incompatible shapes, sites are duplicated, or
            no coordinate is observed.
        TypeError: If an aligned array has an incompatible dtype.
        KeyError: If a site does not occur in ``system``.
    """

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    coordinates: NDArray
    coordinate_mask: NDArray[np.bool_]
    occupancy: NDArray | None
    b_factor: NDArray | None
    residue_numbers: dict[ComponentReference, ResidueNumber] | None

    def __init__(
        self,
        system: MolecularSystem,
        sites: tuple[AtomSite, ...],
        coordinates: NDArray,
        coordinate_mask: NDArray[np.bool_],
        occupancy: NDArray | None = None,
        b_factor: NDArray | None = None,
        residue_numbers: dict[ComponentReference, ResidueNumber] | None = None,
    ) -> None:
        self.system = system
        self.sites = sites
        self.coordinates = coordinates
        self.coordinate_mask = coordinate_mask
        self.occupancy = occupancy
        self.b_factor = b_factor
        self.residue_numbers = residue_numbers

        count = len(self.sites)
        if self.coordinates.shape != (count, 3):
            raise ValueError(f"coordinates must have shape {(count, 3)}, got {self.coordinates.shape}")
        if self.coordinates.dtype.kind not in "iuf":
            raise TypeError("coordinates must have a real numerical dtype")
        if self.coordinate_mask.shape != (count,):
            raise ValueError(f"coordinate_mask must have shape {(count,)}, got {self.coordinate_mask.shape}")
        if self.coordinate_mask.dtype != np.bool_:
            raise TypeError("coordinate_mask must have boolean dtype")
        if self.occupancy is not None and self.occupancy.shape != (count,):
            raise ValueError(f"occupancy must have shape {(count,)}, got {self.occupancy.shape}")
        if self.occupancy is not None and self.occupancy.dtype.kind not in "iuf":
            raise TypeError("occupancy must have a real numerical dtype")
        if self.b_factor is not None and self.b_factor.shape != (count,):
            raise ValueError(f"b_factor must have shape {(count,)}, got {self.b_factor.shape}")
        if self.b_factor is not None and self.b_factor.dtype.kind not in "iuf":
            raise TypeError("b_factor must have a real numerical dtype")
        if len(self.sites) != len(set(self.sites)):
            raise ValueError("structure sites must be unique")
        for site in self.sites:
            self.system.atom(site)
        if not self.coordinate_mask.any():
            raise ValueError("structure must contain at least one observed coordinate")
