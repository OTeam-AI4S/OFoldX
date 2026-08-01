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
from tensordict import tensorclass
from torch import Tensor

from .molecular import MolecularSystem, PolymerType

STRUCTURE_ARTIFACT_SUFFIX = ".structure.safetensors"

_STRUCTURE_ARTIFACT_FORMAT = "ofoldx.structure"
_STRUCTURE_ARTIFACT_VERSION = "3"

RESOLUTION_KIND_UNKNOWN = 0
RESOLUTION_KIND_REFINEMENT = 1
RESOLUTION_KIND_REFLECTIONS = 2
RESOLUTION_KIND_EM_RECONSTRUCTION = 3
RESOLUTION_KIND_PDB_REMARK_2 = 4


@tensorclass
class Structure:
    r"""
    One coordinate realization of one or more ragged molecular systems.

    Every atom-aligned field uses the complete realized atom axis of ``system``.
    Every component-aligned source field uses its complete component axis.
    Experimental-resolution records form a ragged axis over molecular systems.
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
    standard_component_id: Tensor
    system_resolution_ptr: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    resolution: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.float64))
    resolution_kind: Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.uint8))

    def __post_init__(self) -> None:
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
        return cls(  # type: ignore[call-arg]
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
            standard_component_id=torch.full((component_count,), -1, dtype=torch.int64),
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

        return Structure(  # type: ignore[call-arg]
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
            standard_component_id=move(self.standard_component_id),
            system_resolution_ptr=move(self.system_resolution_ptr),
            resolution=move(self.resolution),
            resolution_kind=move(self.resolution_kind),
            batch_size=[],
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
        standard_component_ids = []
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
            standard_component_ids.append(
                torch.where(
                    structure.standard_component_id < 0,
                    structure.standard_component_id,
                    structure.standard_component_id + source_offset,
                )
            )
            source_offset += len(structure.source_symbols)

        resolution_counts = torch.cat(
            [structure.system_resolution_ptr[1:] - structure.system_resolution_ptr[:-1] for structure in structures]
        )
        return cls(  # type: ignore[call-arg]
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
            standard_component_id=torch.cat(standard_component_ids),
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
            standard_component_id = self.standard_component_id[component_start:component_stop]
            resolution_start = int(self.system_resolution_ptr[system_index])
            resolution_stop = int(self.system_resolution_ptr[system_index + 1])
            used_symbols = torch.unique(
                torch.cat(
                    (
                        residue_id[residue_id >= 0],
                        insertion_code[insertion_code >= 0],
                        standard_component_id[standard_component_id >= 0],
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
            remapped_standard_component_id = standard_component_id.clone()
            present_standard_component_id = standard_component_id >= 0
            remapped_standard_component_id[present_standard_component_id] = symbol_map[
                standard_component_id[present_standard_component_id]
            ]
            outputs.append(
                Structure(  # type: ignore[call-arg]
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
                    standard_component_id=remapped_standard_component_id,
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
    """

    destination = _artifact_path(path)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in structure.state_dict().items()  # type: ignore[attr-defined]
    }
    save_file(
        state,
        str(destination),
        metadata={
            "format": _STRUCTURE_ARTIFACT_FORMAT,
            "version": _STRUCTURE_ARTIFACT_VERSION,
            "system_symbols": json.dumps(structure.system.symbols, ensure_ascii=False),
            "source_symbols": json.dumps(structure.source_symbols, ensure_ascii=False),
        },
    )


def load_structure(path: str | PathLike[str]) -> Structure:
    r"""Load a tensor-native structure saved by :func:`save_structure`."""

    source = _artifact_path(path)
    with safe_open(str(source), framework="pt", device="cpu") as artifact:
        metadata = artifact.metadata()
        if metadata.get("format") != _STRUCTURE_ARTIFACT_FORMAT:
            raise ValueError(f"{source} is not an OFoldX structure artifact")
        if metadata.get("version") != _STRUCTURE_ARTIFACT_VERSION:
            raise ValueError(f"unsupported OFoldX structure artifact version in {source}")
        state = {key: artifact.get_tensor(key) for key in artifact.keys()}

    system_fields = tuple(name for name in MolecularSystem.__annotations__ if name != "symbols")
    structure_fields = tuple(name for name in Structure.__annotations__ if name not in {"system", "source_symbols"})
    expected_keys = {
        *(f"system.{name}" for name in system_fields),
        *structure_fields,
    }
    if state.keys() != expected_keys:
        raise ValueError(f"{source} does not contain the complete Structure tensor schema")

    system = MolecularSystem(  # type: ignore[call-arg]
        symbols=_symbols(metadata, "system_symbols", source),
        **{name: state[f"system.{name}"] for name in system_fields},
        batch_size=[],
    )
    return Structure(  # type: ignore[call-arg]
        system=system,
        source_symbols=_symbols(metadata, "source_symbols", source),
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
