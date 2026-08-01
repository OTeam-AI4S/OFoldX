r"""
Coordinates aligned to the canonical molecular-system atom axis.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import field
from os import PathLike
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from . import metadata as _metadata
from ._tensorclass import TensorClass
from .molecular import MolecularSystem, PolymerType, atom_subgraph, suppress_hydrogens

RESOLUTION_KIND_EM_RECONSTRUCTION = _metadata.RESOLUTION_KIND_EM_RECONSTRUCTION
RESOLUTION_KIND_PDB_REMARK_2 = _metadata.RESOLUTION_KIND_PDB_REMARK_2
RESOLUTION_KIND_REFINEMENT = _metadata.RESOLUTION_KIND_REFINEMENT
RESOLUTION_KIND_REFLECTIONS = _metadata.RESOLUTION_KIND_REFLECTIONS
RESOLUTION_KIND_UNKNOWN = _metadata.RESOLUTION_KIND_UNKNOWN

STRUCTURE_ARTIFACT_SUFFIX = ".structure.safetensors"

_STRUCTURE_ARTIFACT_FORMAT = "ofoldx.structure"


class Structure(TensorClass):
    r"""
    One coordinate realization of one or more ragged molecular systems.

    Every atom-aligned field uses the complete realized atom axis of ``system``.
    Every component-aligned source field uses its complete component axis.
    Experimental-resolution records form a ragged axis over molecular systems.
    Parent declarations form a ragged axis over components: ``component_parent_ptr``
    partitions ``parent_component_id``, whose values reference ``source_symbols``.
    Empty slices mean no declared parents; multiple entries are retained without
    implying a preferred parent or an atom correspondence.
    ``system_resolution_ptr`` owns the slices and ``resolution_kind`` distinguishes
    refinement, reflection-processing, EM-reconstruction, and legacy-PDB values.
    """

    system: MolecularSystem
    coordinates: Tensor
    coordinate_mask: Tensor
    occupancy: Tensor
    occupancy_mask: Tensor
    b_factor: Tensor
    b_factor_mask: Tensor
    source_symbols: tuple[str, ...]
    residue_id: Tensor
    insertion_code: Tensor
    component_parent_ptr: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    parent_component_id: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    source_formal_charge_mask: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.bool))
    system_resolution_ptr: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    resolution: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    resolution_kind: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))

    def __post_init__(self) -> None:
        if not isinstance(self.source_symbols, tuple):
            self.source_symbols = tuple(self.source_symbols)
        component_count = int(self.system.component_symbol.numel())
        if not self.component_parent_ptr.numel() and not self.parent_component_id.numel():
            self.component_parent_ptr = torch.zeros(
                component_count + 1, dtype=torch.int64, device=self.coordinates.device
            )
            self.parent_component_id = torch.empty(0, dtype=torch.int64, device=self.coordinates.device)
        if self.component_parent_ptr.dtype != torch.int64 or self.parent_component_id.dtype != torch.int64:
            raise ValueError("component parent pointers and identifiers must use int64")
        if self.component_parent_ptr.shape != (component_count + 1,) or self.parent_component_id.ndim != 1:
            raise ValueError("component_parent_ptr must have one boundary per component")
        if (
            int(self.component_parent_ptr[0]) != 0
            or bool((self.component_parent_ptr[1:] < self.component_parent_ptr[:-1]).any())
            or int(self.component_parent_ptr[-1]) != self.parent_component_id.numel()
        ):
            raise ValueError("component_parent_ptr must partition the parent component axis")
        if bool(((self.parent_component_id < 0) | (self.parent_component_id >= len(self.source_symbols))).any()):
            raise ValueError("parent_component_id must reference source_symbols")
        atom_count = int(self.system.atom_symbol.numel())
        if not self.source_formal_charge_mask.numel():
            self.source_formal_charge_mask = torch.zeros(
                atom_count,
                dtype=torch.bool,
                device=self.coordinates.device,
            )
        if self.source_formal_charge_mask.shape != (atom_count,):
            raise ValueError("source_formal_charge_mask must have one value per atom")
        system_count = int(self.system.system_entity_ptr.numel()) - 1
        if not self.system_resolution_ptr.numel() and not self.resolution.numel() and not self.resolution_kind.numel():
            self.system_resolution_ptr = torch.zeros(
                system_count + 1,
                dtype=torch.int64,
                device=self.coordinates.device,
            )
        if self.system_resolution_ptr.shape != (system_count + 1,):
            raise ValueError("system_resolution_ptr must have one boundary per molecular system")
        if self.resolution.ndim != 1 or self.resolution_kind.shape != self.resolution.shape:
            raise ValueError("resolution and resolution_kind must be aligned one-dimensional tensors")
        if self.resolution.numel() and (
            not bool(torch.isfinite(self.resolution).all()) or not bool((self.resolution > 0).all())
        ):
            raise ValueError("experimental-resolution values must be finite and positive")
        if self.resolution_kind.numel() and not bool((self.resolution_kind <= RESOLUTION_KIND_PDB_REMARK_2).all()):
            raise ValueError("resolution_kind contains an unknown code")
        if (
            int(self.system_resolution_ptr[0]) != 0
            or bool((self.system_resolution_ptr[1:] < self.system_resolution_ptr[:-1]).any())
            or int(self.system_resolution_ptr[-1]) != self.resolution.numel()
        ):
            raise ValueError("system_resolution_ptr must partition the resolution axis")

    @classmethod
    def from_sequence(
        cls,
        sequence: str,
        *,
        polymer_type: PolymerType = "protein",
        instance_id: str = "A",
    ) -> Structure:
        r"""
        Build one polymer's chemistry with no coordinates.

        The result carries the complete component and bond graph the sequence implies and
        nothing else: every atom is unresolved. A reference conformer places a component's
        atoms relative to each other only, so laying those coordinates out chain-wide
        would put every residue on top of its neighbours and call the overlap observed.
        Read a real structure, or generate a placed one, when coordinates are needed.

        Args:
            sequence: One-letter polymer sequence.
            polymer_type: Polymer alphabet in which to read ``sequence``.
            instance_id: Instance ID given to the single realized chain.

        Returns:
            A realization whose ``coordinate_mask`` is unset for every atom.

        Examples:
            >>> structure = Structure.from_sequence("ACDEF")
            >>> int(structure.system.component_symbol.numel()), bool(structure.coordinate_mask.any())
            (5, False)
        """

        from .molecular import PolymerSequence

        polymer = PolymerSequence.from_one_letter(sequence, polymer_type=polymer_type)
        system = MolecularSystem.from_entities({"polymer": polymer}, {instance_id: "polymer"})
        atom_count = int(system.atom_symbol.numel())
        component_count = int(system.component_symbol.numel())
        coordinates = torch.zeros(atom_count, 3)
        coordinate_mask = torch.zeros(atom_count, dtype=torch.bool)
        return cls(
            system=system,
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
            occupancy=torch.zeros(atom_count),
            occupancy_mask=torch.zeros(atom_count, dtype=torch.bool),
            b_factor=torch.zeros(atom_count),
            b_factor_mask=torch.zeros(atom_count, dtype=torch.bool),
            source_symbols=(),
            residue_id=torch.full((component_count,), -1, dtype=torch.int64),
            insertion_code=torch.full((component_count,), -1, dtype=torch.int64),
            batch_size=[],
        )

    def to(
        self,
        device: torch.device | str | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Structure:
        r"""
        Move the realization while casting only floating-point values.
        """

        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype was specified twice")
            dtype = device
            device = None
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("Structure.to only accepts floating or complex dtypes")

        def move(value: Tensor) -> Tensor:
            return value.to(
                device=device,
                dtype=(dtype if value.is_floating_point() or value.is_complex() else None),
                non_blocking=non_blocking,
            )

        return Structure(
            system=self.system.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            coordinates=move(self.coordinates),
            coordinate_mask=move(self.coordinate_mask),
            occupancy=move(self.occupancy),
            occupancy_mask=move(self.occupancy_mask),
            b_factor=move(self.b_factor),
            b_factor_mask=move(self.b_factor_mask),
            source_symbols=self.source_symbols,
            residue_id=move(self.residue_id),
            insertion_code=move(self.insertion_code),
            component_parent_ptr=move(self.component_parent_ptr),
            parent_component_id=move(self.parent_component_id),
            source_formal_charge_mask=move(self.source_formal_charge_mask),
            system_resolution_ptr=move(self.system_resolution_ptr),
            resolution=move(self.resolution),
            resolution_kind=move(self.resolution_kind),
            batch_size=[],
        )

    def component_parents(self, component: int) -> tuple[str, ...]:
        r"""Return all declared parent IDs for one component in source order.

        Args:
            component: Zero-based index on the complete component axis.

        Returns:
            Parent identifiers, or an empty tuple if none were declared.

        Examples:
            >>> structure = Structure.from_sequence("A")
            >>> structure.component_parents(0)
            ()
        """
        if not 0 <= component < self.system.component_symbol.numel():
            raise IndexError("component index is outside the molecular system")
        start, stop = self.component_parent_ptr[component : component + 2].detach().cpu().tolist()
        return tuple(
            self.source_symbols[index] for index in self.parent_component_id[start:stop].detach().cpu().tolist()
        )

    def select_component_parents(self, indices: Tensor | Sequence[int]) -> tuple[Tensor, Tensor]:
        r"""Gather complete parent lists in output-component order.

        Args:
            indices: One-dimensional integer source indices. Repetition copies a
                complete list; ``-1`` represents a new component without parents.

        Returns:
            pointers: Rebased component pointers with independent output storage.
            parents: Parent IDs referencing the same ``source_symbols`` table, with
                independent output storage.

        Examples:
            >>> structure = Structure.from_sequence("AC")
            >>> pointers, parents = structure.select_component_parents([1, -1, 0])
            >>> pointers.tolist(), parents.tolist()
            ([0, 0, 0, 0], [])
        """
        indices = torch.as_tensor(indices, device=self.component_parent_ptr.device)
        if indices.ndim != 1 or (indices.numel() and indices.dtype not in (torch.int32, torch.int64)):
            raise ValueError("parent selection requires one-dimensional integer component indices")
        indices = indices.to(torch.int64)
        component_count = self.component_parent_ptr.numel() - 1
        if bool(((indices < -1) | (indices >= component_count)).any()):
            raise IndexError("parent selection contains an invalid component index")
        retained = indices >= 0
        starts = torch.zeros_like(indices)
        counts = torch.zeros_like(indices)
        starts[retained] = self.component_parent_ptr[indices[retained]]
        counts[retained] = self.component_parent_ptr[indices[retained] + 1] - starts[retained]
        pointers = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        rows = torch.repeat_interleave(starts - pointers[:-1], counts)
        rows += torch.arange(rows.numel(), device=rows.device)
        return pointers, self.parent_component_id[rows]

    def atom_subgraph(self, atom_mask: Tensor) -> tuple[Structure, Tensor]:
        r"""Keep selected atoms while preserving every atom-aligned field."""

        system, old_to_new = atom_subgraph(self.system, atom_mask)
        return (
            self.replace(
                system=system,
                coordinates=self.coordinates[atom_mask],
                coordinate_mask=self.coordinate_mask[atom_mask],
                occupancy=self.occupancy[atom_mask],
                occupancy_mask=self.occupancy_mask[atom_mask],
                b_factor=self.b_factor[atom_mask],
                b_factor_mask=self.b_factor_mask[atom_mask],
                source_formal_charge_mask=self.source_formal_charge_mask[atom_mask],
            ),
            old_to_new,
        )

    def suppress_hydrogens(self) -> Structure:
        r"""Return the same realization with hydrogen vertices folded into heavy atoms."""

        system, old_to_new = suppress_hydrogens(self.system)
        if system is self.system:
            return self
        keep = old_to_new >= 0
        return self.replace(
            system=system,
            coordinates=self.coordinates[keep],
            coordinate_mask=self.coordinate_mask[keep],
            occupancy=self.occupancy[keep],
            occupancy_mask=self.occupancy_mask[keep],
            b_factor=self.b_factor[keep],
            b_factor_mask=self.b_factor_mask[keep],
            source_formal_charge_mask=self.source_formal_charge_mask[keep],
        )

    def float(self) -> Structure:
        return self.to(dtype=torch.float32)

    def double(self) -> Structure:
        return self.to(dtype=torch.float64)

    def half(self) -> Structure:
        return self.to(dtype=torch.float16)

    def bfloat16(self) -> Structure:
        return self.to(dtype=torch.bfloat16)

    @classmethod
    def batch(cls, structures: Sequence[Structure]) -> Structure:
        r"""
        Concatenate coordinate realizations as a ragged disjoint union.
        """

        source_symbols: list[str] = []
        source_offset = 0
        residue_ids = []
        insertion_codes = []
        parent_ids = []
        parent_counts = []
        for structure in structures:
            source_symbols.extend(structure.source_symbols)
            residue_ids.append(
                torch.where(
                    structure.residue_id < 0,
                    structure.residue_id,
                    structure.residue_id + source_offset,
                )
            )
            insertion_codes.append(
                torch.where(
                    structure.insertion_code < 0,
                    structure.insertion_code,
                    structure.insertion_code + source_offset,
                )
            )
            parent_ids.append(structure.parent_component_id + source_offset)
            parent_counts.append(structure.component_parent_ptr[1:] - structure.component_parent_ptr[:-1])
            source_offset += len(structure.source_symbols)

        resolution_counts = torch.cat(
            [structure.system_resolution_ptr[1:] - structure.system_resolution_ptr[:-1] for structure in structures]
        )
        return cls(
            system=MolecularSystem.batch([structure.system for structure in structures]),
            coordinates=torch.cat([structure.coordinates for structure in structures]),
            coordinate_mask=torch.cat([structure.coordinate_mask for structure in structures]),
            occupancy=torch.cat([structure.occupancy for structure in structures]),
            occupancy_mask=torch.cat([structure.occupancy_mask for structure in structures]),
            b_factor=torch.cat([structure.b_factor for structure in structures]),
            b_factor_mask=torch.cat([structure.b_factor_mask for structure in structures]),
            source_symbols=tuple(source_symbols),
            residue_id=torch.cat(residue_ids),
            insertion_code=torch.cat(insertion_codes),
            component_parent_ptr=torch.cat((parent_counts[0].new_zeros(1), torch.cat(parent_counts).cumsum(0))),
            parent_component_id=torch.cat(parent_ids),
            source_formal_charge_mask=torch.cat([structure.source_formal_charge_mask for structure in structures]),
            system_resolution_ptr=torch.cat(
                [
                    resolution_counts.new_zeros(1),
                    resolution_counts.cumsum(0),
                ]
            ),
            resolution=torch.cat([structure.resolution for structure in structures]),
            resolution_kind=torch.cat([structure.resolution_kind for structure in structures]),
            batch_size=[],
        )

    def unbatch(self) -> tuple[Structure, ...]:
        r"""
        Split a ragged coordinate union into single-system realizations.
        """

        systems = self.system.unbatch()
        outputs = []
        for system_index, system in enumerate(systems):
            instance_start = int(self.system.system_instance_ptr[system_index].item())
            instance_stop = int(self.system.system_instance_ptr[system_index + 1].item())
            component_start = int(self.system.instance_component_ptr[instance_start].item())
            component_stop = int(self.system.instance_component_ptr[instance_stop].item())
            atom_start = int(self.system.component_atom_ptr[component_start].item())
            atom_stop = int(self.system.component_atom_ptr[component_stop].item())
            residue_id = self.residue_id[component_start:component_stop]
            insertion_code = self.insertion_code[component_start:component_stop]
            parent_start = int(self.component_parent_ptr[component_start])
            parent_stop = int(self.component_parent_ptr[component_stop])
            component_parent_ptr = self.component_parent_ptr[component_start : component_stop + 1] - parent_start
            parent_component_id = self.parent_component_id[parent_start:parent_stop]
            resolution_start = int(self.system_resolution_ptr[system_index])
            resolution_stop = int(self.system_resolution_ptr[system_index + 1])
            used_symbols = torch.unique(
                torch.cat(
                    (
                        residue_id[residue_id >= 0],
                        insertion_code[insertion_code >= 0],
                        parent_component_id,
                    )
                ),
                sorted=True,
            )
            symbol_map = torch.full(
                (len(self.source_symbols),),
                -1,
                dtype=torch.int64,
                device=residue_id.device,
            )
            symbol_map[used_symbols] = torch.arange(
                used_symbols.numel(),
                dtype=torch.int64,
                device=residue_id.device,
            )
            source_symbols = tuple(self.source_symbols[index] for index in used_symbols.detach().cpu().tolist())
            remapped_residue_id = residue_id.clone()
            present_residue_id = residue_id >= 0
            remapped_residue_id[present_residue_id] = symbol_map[residue_id[present_residue_id]]
            remapped_insertion_code = insertion_code.clone()
            present_insertion_code = insertion_code >= 0
            remapped_insertion_code[present_insertion_code] = symbol_map[insertion_code[present_insertion_code]]
            outputs.append(
                Structure(
                    system=system,
                    coordinates=self.coordinates[atom_start:atom_stop],
                    coordinate_mask=self.coordinate_mask[atom_start:atom_stop],
                    occupancy=self.occupancy[atom_start:atom_stop],
                    occupancy_mask=self.occupancy_mask[atom_start:atom_stop],
                    b_factor=self.b_factor[atom_start:atom_stop],
                    b_factor_mask=self.b_factor_mask[atom_start:atom_stop],
                    source_symbols=source_symbols,
                    residue_id=remapped_residue_id,
                    insertion_code=remapped_insertion_code,
                    component_parent_ptr=component_parent_ptr,
                    parent_component_id=symbol_map[parent_component_id],
                    source_formal_charge_mask=self.source_formal_charge_mask[atom_start:atom_stop],
                    system_resolution_ptr=self.system_resolution_ptr.new_tensor(
                        [0, resolution_stop - resolution_start]
                    ),
                    resolution=self.resolution[resolution_start:resolution_stop],
                    resolution_kind=self.resolution_kind[resolution_start:resolution_stop],
                    batch_size=[],
                )
            )
        return tuple(outputs)


def save_structure(structure: Structure, path: str | PathLike[str]) -> None:
    r"""
    Save a complete tensor-native structure as one OFoldX artifact.

    Unlike PDB and mmCIF, this format preserves the canonical molecular graph
    without projecting it back into a source interchange schema.

    Args:
        structure: Molecular graph, coordinates, and source metadata to preserve,
            including complete parent declarations and floating-point precision.
        path: Destination ending in ``.structure.safetensors``.

    Raises:
        ValueError: The destination does not have the required artifact suffix.
    """

    destination = _artifact_path(path)
    state = {key: value.detach().cpu().contiguous() for key, value in structure.state_dict().items()}
    save_file(
        state,
        str(destination),
        metadata={
            "format": _STRUCTURE_ARTIFACT_FORMAT,
            "system_symbols": json.dumps(structure.system.symbols, ensure_ascii=False),
            "source_symbols": json.dumps(structure.source_symbols, ensure_ascii=False),
        },
    )


def load_structure(path: str | PathLike[str]) -> Structure:
    r"""Load the complete tensor schema written by [`save_structure`][ofoldx.data.structure.save_structure].

    Args:
        path: Source ending in ``.structure.safetensors``.

    Returns:
        A Structure on CPU with its molecular graph, source symbol tables,
            ragged parent declarations, and stored tensor dtypes preserved.

    Raises:
        ValueError: The artifact suffix, format, symbol tables, or tensor schema
            is invalid.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:
        ...     path = Path(directory) / "example.structure.safetensors"
        ...     save_structure(Structure.from_sequence("A"), path)
        ...     restored = load_structure(path)
        >>> restored.component_parent_ptr.tolist(), restored.component_parents(0)
        ([0, 0], ())
    """

    source = _artifact_path(path)
    with safe_open(str(source), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        if metadata.get("format") != _STRUCTURE_ARTIFACT_FORMAT:
            raise ValueError(f"{source} is not an OFoldX structure artifact")
        state = {key: artifact.get_tensor(key) for key in artifact.keys()}

    system_symbols = _symbols(metadata, "system_symbols", source)
    source_symbols = _symbols(metadata, "source_symbols", source)
    system_fields = tuple(name for name in MolecularSystem.__annotations__ if name != "symbols")
    structure_fields = tuple(name for name in Structure.__annotations__ if name not in {"system", "source_symbols"})
    expected_keys = {
        *(f"system.{name}" for name in system_fields),
        *structure_fields,
    }
    if state.keys() != expected_keys:
        raise ValueError(f"{source} does not contain the complete Structure tensor schema")

    system = MolecularSystem(
        symbols=system_symbols,
        **{name: state[f"system.{name}"] for name in system_fields},
        batch_size=[],
    )
    return Structure(
        system=system,
        source_symbols=source_symbols,
        **{name: state[name] for name in structure_fields},
        batch_size=[],
    )


def _artifact_path(path: str | PathLike[str]) -> Path:
    artifact = Path(path)
    if not artifact.name.endswith(STRUCTURE_ARTIFACT_SUFFIX):
        raise ValueError(f"structure artifacts must end with {STRUCTURE_ARTIFACT_SUFFIX!r}")
    return artifact


def _symbols(metadata: dict[str, str], key: str, path: Path) -> tuple[str, ...]:
    values = json.loads(metadata[key])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"invalid {key!r} metadata in {path}")
    return tuple(values)
