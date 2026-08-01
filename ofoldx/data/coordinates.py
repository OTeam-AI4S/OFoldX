"""Polymer coordinate evidence, independent of connectivity and tensor storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .._arrays import immutable_array

if TYPE_CHECKING:
    from .molecular import PolymerType
    from .structure import Structure


@dataclass(frozen=True, eq=False, kw_only=True)
class ChainCoordinates:
    """One polymer instance, including declared positions with no observed atoms.

    component_index addresses zero-based sequence positions, never author residue
    numbers. Names identify atom rows; coordinate_mask marks which rows are observed.
    An absent name or false mask does not imply chemical absence.
    Coordinates share the containing set's Cartesian frame.
    Arrays are owned immutable snapshots; masks determine coordinate validity.

    Attributes:
        polymer_type: Polymer chemistry used by model consumers.
        component_ids: Deposited component IDs in sequence order.
        atom_names: Atom names aligned with the coordinate rows.
        component_index: Zero-based sequence position of each atom row.
        coordinates: Cartesian coordinates in angstroms, stored as float32 or float64.
        coordinate_mask: Whether each coordinate row is observed.
        parent_component_ids: Complete declared parent lists, one tuple per position.
            An empty tuple means no parent declaration; no preferred parent is implied.

    Examples:
        >>> xyz = np.array([[1., 2., 3.]])
        >>> chain = ChainCoordinates(
        ...     polymer_type="protein", component_ids=("ALA",), atom_names=("CA",),
        ...     component_index=np.array([0]), coordinates=xyz, coordinate_mask=np.array([True]),
        ... )
        >>> xyz[:] = 0
        >>> chain.coordinates.tolist()
        [[1.0, 2.0, 3.0]]
        >>> chain.parent_component_ids
        ((),)
    """

    polymer_type: PolymerType
    component_ids: tuple[str, ...]
    atom_names: tuple[str, ...]
    component_index: NDArray[np.int64]
    coordinates: NDArray[np.float32 | np.float64]
    coordinate_mask: NDArray[np.bool_]
    parent_component_ids: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for name in ("component_ids", "atom_names"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if any(isinstance(parents, str) for parents in self.parent_component_ids):
            raise TypeError("parent_component_ids must contain sequences of component IDs")
        parents = tuple(tuple(values) for values in self.parent_component_ids)
        object.__setattr__(self, "parent_component_ids", parents or ((),) * len(self.component_ids))
        if (
            not isinstance(self.polymer_type, str)
            or not self.polymer_type
            or any(not isinstance(name, str) or not name for name in (*self.component_ids, *self.atom_names))
        ):
            raise ValueError("polymer type, component IDs and atom names must be nonempty")
        indices = np.asarray(self.component_index)
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("component_index must contain integers")
        coordinate_dtype = np.asarray(self.coordinates).dtype
        if coordinate_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("coordinates must use float32 or float64 storage")
        count = len(self.atom_names)
        for name, dtype, shape in (
            ("component_index", np.int64, (count,)),
            ("coordinates", coordinate_dtype, (count, 3)),
            ("coordinate_mask", np.bool_, (count,)),
        ):
            array = immutable_array(getattr(self, name), dtype)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, array)
        if np.any(self.component_index < 0) or np.any(self.component_index >= len(self.component_ids)):
            raise ValueError("atom component_index is outside the polymer sequence")
        if not np.isfinite(self.coordinates[self.coordinate_mask]).all():
            raise ValueError("observed coordinates must be finite")
        if len(set(zip(self.component_index.tolist(), self.atom_names))) != count:
            raise ValueError("duplicate observed atom at a polymer position")
        if len(self.parent_component_ids) != len(self.component_ids):
            raise ValueError("parent_component_ids must match the sequence length")
        if any(not isinstance(value, str) or not value for parents in self.parent_component_ids for value in parents):
            raise ValueError("parent component identifiers must be nonempty strings")

    def __getstate__(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @cached_property
    def _atom_index(self) -> Mapping[tuple[int, str], int]:
        return MappingProxyType(
            {
                (int(component), name): index
                for index, (component, name) in enumerate(zip(self.component_index, self.atom_names, strict=True))
            }
        )

    def gather(
        self, positions: Sequence[int], atom_names: Sequence[str | None]
    ) -> tuple[NDArray[np.float32 | np.float64], NDArray[np.bool_]]:
        """Project named observations into independently writable arrays.

        Args:
            positions: Zero-based sequence positions, in output order; repeats are allowed.
            atom_names: Atom names in output order; ``None`` reserves an empty slot.

        Returns:
            coordinates: Array shaped ``(len(positions), len(atom_names), 3)`` in the
                source dtype. Missing and unobserved atoms have zero coordinates.
            present: Boolean array shaped ``(len(positions), len(atom_names))``.
                Missing and unobserved atoms have a false mask.

        Examples:
            >>> chain = ChainCoordinates(
            ...     polymer_type="rna", component_ids=("A", "C"), atom_names=("P",),
            ...     component_index=np.array([0]), coordinates=np.array([[1., 2., 3.]]),
            ...     coordinate_mask=np.array([True]),
            ... )
            >>> xyz, mask = chain.gather([1, 0], ["P", None])
            >>> mask.tolist()
            [[False, False], [True, False]]
            >>> xyz[1, 0].tolist()
            [1.0, 2.0, 3.0]
        """
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in positions):
            raise TypeError("coordinate positions must be integers")
        if any(index < 0 or index >= len(self.component_ids) for index in positions):
            raise IndexError("coordinate position is outside the polymer sequence")
        indices = np.asarray(
            [
                [self._atom_index.get((index, name), -1) if name is not None else -1 for name in atom_names]
                for index in positions
            ],
            dtype=np.int64,
        ).reshape(len(positions), len(atom_names))
        coordinates = np.zeros((*indices.shape, 3), dtype=self.coordinates.dtype)
        mask = indices >= 0
        if mask.any():
            mask[mask] &= self.coordinate_mask[indices[mask]]
            coordinates[mask] = self.coordinates[indices[mask]]
        return coordinates, mask


@dataclass(frozen=True, eq=False)
class PolymerCoordinates:
    """Polymer instances observed in one shared frame; no bond graph is implied.

    Attributes:
        chains: Instance IDs mapped to immutable [`ChainCoordinates`][ofoldx.data.coordinates.ChainCoordinates].
            The mapping is copied; immutable chain values may be shared.

    Examples:
        >>> chains = {}
        >>> evidence = PolymerCoordinates(chains)
        >>> evidence.chains is chains
        False
        >>> dict(evidence.chains)
        {}
    """

    chains: Mapping[str, ChainCoordinates]

    def __post_init__(self) -> None:
        if any(not isinstance(key, str) or not key for key in self.chains):
            raise ValueError("coordinate chain identifiers must be nonempty strings")
        if any(not isinstance(value, ChainCoordinates) for value in self.chains.values()):
            raise TypeError("chains must contain ChainCoordinates")
        object.__setattr__(self, "chains", MappingProxyType(dict(self.chains)))

    def __reduce__(self) -> tuple:
        return type(self), (dict(self.chains),)

    @classmethod
    def from_structure(cls, structure: Structure) -> PolymerCoordinates:
        """Take one owned CPU snapshot of a single molecular system's polymers.

        Args:
            structure: A completed coordinate realization. Finish mutations before
                taking the snapshot; subsequent edits do not affect the result.

        Returns:
            Coordinate evidence with all parent declarations and source precision.
                Float16 and bfloat16 coordinates are promoted to float32 before NumPy conversion.

        Raises:
            ValueError: If the structure contains multiple systems or unsupported coordinate storage.

        Examples:
            >>> from ofoldx.data import Structure
            >>> source = Structure.from_sequence("AG")
            >>> evidence = PolymerCoordinates.from_structure(source)
            >>> evidence.chains["A"].component_ids
            ('ALA', 'GLY')
            >>> source.coordinate_mask[:] = True
            >>> bool(evidence.chains["A"].coordinate_mask.any())
            False
        """
        import torch

        from .molecular import ENTITY_DNA, ENTITY_DNA_RNA_HYBRID, ENTITY_PNA, ENTITY_PROTEIN, ENTITY_RNA

        kinds: dict[int, PolymerType] = {
            ENTITY_PROTEIN: "protein",
            ENTITY_RNA: "rna",
            ENTITY_DNA: "dna",
            ENTITY_DNA_RNA_HYBRID: "dna_rna_hybrid",
            ENTITY_PNA: "pna",
        }
        system = structure.system
        if system.system_instance_ptr.numel() != 2:
            raise ValueError("coordinate evidence requires one molecular system; call Structure.unbatch() first")
        coordinates = structure.coordinates.detach().cpu()
        if coordinates.dtype in (torch.float16, torch.bfloat16):
            coordinates = coordinates.float()
        if coordinates.dtype not in (torch.float32, torch.float64):
            raise ValueError("structure coordinates must use real floating-point storage")
        positions = coordinates.numpy()
        mask = structure.coordinate_mask.detach().cpu().numpy()
        chains = {}
        component_ptr = system.instance_component_ptr.detach().cpu().tolist()
        atom_ptr = system.component_atom_ptr.detach().cpu().tolist()
        parent_ptr = structure.component_parent_ptr.detach().cpu().tolist()
        parent_ids = structure.parent_component_id.detach().cpu().tolist()
        component_ids = system.component_symbol.detach().cpu().tolist()
        atom_names = system.atom_symbol.detach().cpu().tolist()
        entity_types = system.entity_type.detach().cpu().tolist()
        for instance, entity in enumerate(system.instance_entity.detach().cpu().tolist()):
            kind = kinds.get(entity_types[entity])
            if kind is None:
                continue
            start, stop = component_ptr[instance : instance + 2]
            atom_start, atom_stop = atom_ptr[start], atom_ptr[stop]
            chains[system.instance_id(instance)] = ChainCoordinates(
                polymer_type=kind,
                component_ids=tuple(system.symbols[index] for index in component_ids[start:stop]),
                parent_component_ids=tuple(
                    tuple(structure.source_symbols[index] for index in parent_ids[parent_ptr[i] : parent_ptr[i + 1]])
                    for i in range(start, stop)
                ),
                atom_names=tuple(system.symbols[index] for index in atom_names[atom_start:atom_stop]),
                component_index=np.repeat(np.arange(stop - start, dtype=np.int64), np.diff(atom_ptr[start : stop + 1])),
                coordinates=positions[atom_start:atom_stop],
                coordinate_mask=mask[atom_start:atom_stop],
            )
        return cls(chains)
