r"""
Coordinates aligned to the canonical molecular-system atom axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ._tensorclass import frozen_tensorclass
from .molecular import MolecularSystem


@frozen_tensorclass
class Structure:
    r"""
    One coordinate realization of one or more ragged molecular systems.

    Every atom-aligned field uses the complete realized atom axis of ``system``.
    Every component-aligned source field uses its complete component axis.
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
    resolution: Tensor | None = None
    resolution_is_known: Tensor | None = None

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
            resolution=None if self.resolution is None else move(self.resolution),
            resolution_is_known=(
                None if self.resolution_is_known is None else move(self.resolution_is_known)
            ),
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
        resolution_values = []
        resolution_known_values = []
        has_resolution = any(
            structure.resolution is not None or structure.resolution_is_known is not None
            for structure in structures
        )
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
            source_offset += len(structure.source_symbols)
            if has_resolution:
                system_count = structure.system.system_entity_ptr.numel() - 1
                if structure.resolution is None and structure.resolution_is_known is None:
                    resolution_values.append(structure.coordinates.new_full((system_count,), -1.0))
                    resolution_known_values.append(
                        torch.zeros(system_count, dtype=torch.bool, device=structure.coordinates.device)
                    )
                elif structure.resolution is None or structure.resolution_is_known is None:
                    raise ValueError("resolution and resolution_is_known must either both be present or both be absent")
                else:
                    resolution = structure.resolution.reshape(-1)
                    resolution_is_known = structure.resolution_is_known.reshape(-1)
                    if resolution.numel() != system_count or resolution_is_known.numel() != system_count:
                        raise ValueError("resolution fields must have one value per molecular system")
                    resolution_values.append(resolution)
                    resolution_known_values.append(resolution_is_known)

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
            resolution=torch.cat(resolution_values) if has_resolution else None,
            resolution_is_known=torch.cat(resolution_known_values) if has_resolution else None,
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
            used_symbols = torch.unique(
                torch.cat(
                    (
                        residue_id[residue_id >= 0],
                        insertion_code[insertion_code >= 0],
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
                    resolution=(
                        None if self.resolution is None else self.resolution.reshape(-1)[system_index]
                    ),
                    resolution_is_known=(
                        None
                        if self.resolution_is_known is None
                        else self.resolution_is_known.reshape(-1)[system_index]
                    ),
                    batch_size=[],
                )
            )
        return tuple(outputs)
