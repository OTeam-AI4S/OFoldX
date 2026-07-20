r"""Coordinates aligned to exact atom occurrences in a MolecularSystem."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .molecular import AtomSite, ComponentReference, MolecularSystem


@dataclass(frozen=True)
class ResidueNumber:
    r"""One source residue number and insertion code."""

    number: int
    insertion_code: str = ""


class Structure:
    r"""One coordinate realization of a molecular system."""

    system: MolecularSystem
    sites: tuple[AtomSite, ...]
    coordinates: NDArray
    coordinate_mask: NDArray[np.bool_]
    occupancy: NDArray | None
    residue_numbers: dict[ComponentReference, ResidueNumber] | None

    def __init__(
        self,
        system: MolecularSystem,
        sites: tuple[AtomSite, ...],
        coordinates: NDArray,
        coordinate_mask: NDArray[np.bool_],
        occupancy: NDArray | None = None,
        residue_numbers: dict[ComponentReference, ResidueNumber] | None = None,
    ) -> None:
        self.system = system
        self.sites = sites
        self.coordinates = coordinates
        self.coordinate_mask = coordinate_mask
        self.occupancy = occupancy
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
        if len(self.sites) != len(set(self.sites)):
            raise ValueError("structure sites must be unique")
        for site in self.sites:
            self.system.atom(site)
        if not self.coordinate_mask.any():
            raise ValueError("structure must contain at least one observed coordinate")
