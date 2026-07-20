r"""Canonical Structure lowering for LigandMPNN."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from os import PathLike

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from ...data import (
    is_water_component,
)
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    AtomSite,
    ComponentReference,
    MolecularSystem,
    PolymerSequence,
)
from ...data.structure import Structure
from ...io import read_pdb
from ...io.pdb import ATOMIC_NUMBER_BY_ELEMENT
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
    _virtual_beta_carbon,
)
from .config import LigandMpnnProcessorConfig
from .model import LigandMpnnInput

_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_ATOM37 = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)
_BACKBONE_ATOM37_INDICES = tuple(_ATOM37.index(name) for name in _BACKBONE_ATOMS)
_RESIDUE_CONTEXT_START = 5
_GENERATED_IDENTITIES = len(AMINO_ACID_COMPONENT_IDS)
_VOCABULARY = _GENERATED_IDENTITIES + 1

_LoweredRows = tuple[
    list[ComponentReference],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[bool],
    list[int],
    list[int],
    list[str],
]


class LigandMpnnProcessor:
    r"""Lower one canonical protein/context Structure to the released LigandMPNN tensors."""

    config_class = LigandMpnnProcessorConfig

    def __init__(
        self,
        config: LigandMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = LigandMpnnProcessorConfig() if config is None else config
        num_context_atoms = int(config.num_context_atoms)
        if num_context_atoms < 1:
            raise ValueError("num_context_atoms must be a positive integer")
        self.config = config
        self.device = torch.device("cpu" if device is None else device)
        self.dtype = torch.float32 if dtype is None else dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Self:
        r"""Update runtime placement in place and return this Processor."""

        if device is not None:
            self.device = torch.device(device)
        if dtype is not None:
            self.dtype = dtype
        return self

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Select the arm from the expression: a path reads first, then one lowering runs."""

        if isinstance(input, (str, PathLike)):
            return self.process_pdb(
                input,
                positions=positions,
                conditions=conditions,
                tied_weights=tied_weights,
                logit_bias=logit_bias,
                residue_logit_bias=residue_logit_bias,
                use_atom_context=use_atom_context,
                ligand_interface_cutoff=ligand_interface_cutoff,
                filter_by_occupancy=filter_by_occupancy,
            )
        if not isinstance(input, Structure):
            raise TypeError("LigandMPNN requires coordinates; pass a Structure or local PDB/mmCIF path")
        return self.process_structure(
            input,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )

    def process_pdb(
        self,
        path: str | PathLike[str],
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Lower a PDB path exactly as ``processor(read_pdb(path), ...)``.

        Chain filtering stays on the shared reader's ``chains`` argument rather than a
        path-only Processor option; a caller needing it invokes ``read_pdb`` directly
        and passes the resulting Structure.
        """

        return self.process_structure(
            read_pdb(path),
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
            filter_by_occupancy=filter_by_occupancy,
        )

    def process_structure(
        self,
        structure: Structure,
        *,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
    ) -> LigandMpnnInput:
        r"""Lower one canonical Structure to the released protein/context tensors."""

        if not isinstance(structure, Structure):
            raise TypeError("LigandMpnnProcessor.process_structure requires a Structure")
        cutoff = float(ligand_interface_cutoff)
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("ligand_interface_cutoff must be a finite positive number")

        system = structure.system
        for condition in conditions:
            if type(condition) not in (AllowedResidueIdentities, TiedResidueIdentities):
                raise TypeError(f"LigandMPNN cannot consume {type(condition).__name__}")

        observed = structure.coordinate_mask
        if filter_by_occupancy and structure.occupancy is not None:
            observed = observed & (np.isnan(structure.occupancy) | (structure.occupancy > 0))

        rows = self._lower_structure_rows(structure, system, observed)
        return self._finalize(
            structure,
            system,
            rows,
            observed=observed,
            positions=positions,
            conditions=conditions,
            tied_weights=tied_weights,
            logit_bias=logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=cutoff,
        )

    def _lower_structure_rows(
        self,
        structure: Structure,
        system: MolecularSystem,
        observed: np.ndarray,
    ) -> _LoweredRows:
        r"""Build rows: observed-CA protein occurrences in CA-site order, cumulative positions."""

        coordinate_index = {site: index for index, site in enumerate(structure.sites)}

        row_references: list[ComponentReference] = []
        row_backbone: list[np.ndarray] = []
        row_residue_context_coordinates: list[np.ndarray] = []
        row_residue_context_mask: list[np.ndarray] = []
        row_valid: list[bool] = []
        row_component_ids: list[str] = []
        row_chain_indices: list[int] = []
        row_source_numbers: list[int | None] = []
        chain_label_by_instance: dict[str, int] = {}
        for index, site in enumerate(structure.sites):
            if site.atom_id != "CA" or not bool(observed[index]):
                continue
            entity = system.entities[system.instances[site.instance_id]]
            if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                continue
            reference = (site.instance_id, site.component_index)
            if site.instance_id not in chain_label_by_instance:
                chain_label_by_instance[site.instance_id] = len(chain_label_by_instance)

            atom37 = np.zeros((len(_ATOM37), 3), dtype=structure.coordinates.dtype)
            atom37_mask = np.zeros(len(_ATOM37), dtype=np.bool_)
            for atom_index, atom_id in enumerate(_ATOM37):
                if atom_id == "OXT":
                    # The released feature build reserves the OXT slot but never fills
                    # it; leaving it empty keeps the 32-slot residue context in parity.
                    continue
                position = coordinate_index.get(AtomSite(site.instance_id, site.component_index, atom_id))
                if position is not None and bool(observed[position]):
                    atom37[atom_index] = structure.coordinates[position]
                    atom37_mask[atom_index] = True

            backbone = atom37[list(_BACKBONE_ATOM37_INDICES)]
            valid = bool(atom37_mask[list(_BACKBONE_ATOM37_INDICES)].all())
            row_references.append(reference)
            row_backbone.append(backbone)
            row_residue_context_coordinates.append(atom37[_RESIDUE_CONTEXT_START:])
            row_residue_context_mask.append(atom37_mask[_RESIDUE_CONTEXT_START:])
            row_valid.append(valid)
            row_component_ids.append(system.component(reference).component_id)
            row_chain_indices.append(chain_label_by_instance[site.instance_id])
            row_source_numbers.append(
                None if structure.residue_numbers is None else structure.residue_numbers[reference].number
            )

        if not row_references:
            raise ValueError("LigandMPNN requires at least one protein residue with an observed CA")
        if not any(row_valid):
            raise ValueError("LigandMPNN requires at least one residue with complete N/CA/C/O coordinates")

        row_residue_indices: list[int] = []
        if structure.residue_numbers is None:
            for reference in row_references:
                row_residue_indices.append(reference[1])
        else:
            duplicate_count = 0
            previous_number: int | None = None
            for source_number in row_source_numbers:
                assert source_number is not None
                if source_number == previous_number:
                    duplicate_count += 1
                row_residue_indices.append(source_number + duplicate_count)
                previous_number = source_number

        row_tokens = [
            AMINO_ACID_TOKEN_BY_COMPONENT.get(component_id, AMINO_ACID_UNKNOWN_TOKEN)
            for component_id in row_component_ids
        ]

        return (
            row_references,
            row_backbone,
            row_residue_context_coordinates,
            row_residue_context_mask,
            row_tokens,
            row_valid,
            row_residue_indices,
            row_chain_indices,
            row_component_ids,
        )

    def _lower_context_atoms(
        self,
        structure: Structure,
        system: MolecularSystem,
        observed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Collect observed nonprotein, nonwater, nonhydrogen atoms as ordinary context."""

        context_coordinates: list[np.ndarray] = []
        context_atomic_numbers: list[int] = []
        for index, site in enumerate(structure.sites):
            if not bool(observed[index]):
                continue
            entity = system.entities[system.instances[site.instance_id]]
            if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                continue
            if is_water_component(entity.components[site.component_index]):
                continue
            element = system.atom(site).element
            if element == "H":
                continue
            context_coordinates.append(structure.coordinates[index])
            context_atomic_numbers.append(ATOMIC_NUMBER_BY_ELEMENT[element])
        if not context_coordinates:
            context_coordinates.append(np.zeros(3, dtype=np.float32))
            context_atomic_numbers.append(0)
        return (
            np.stack(context_coordinates).astype(np.float32),
            np.asarray(context_atomic_numbers, dtype=np.int64),
        )

    def _select_context(
        self,
        backbone: Tensor,
        valid_mask: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        *,
        use_atom_context: bool,
        ligand_interface_cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Select the nearest padded context atoms per residue and the interface mask."""

        num_context_atoms = int(self.config.num_context_atoms)
        length = backbone.shape[0]
        cb = _virtual_beta_carbon(backbone[:, 0], backbone[:, 1], backbone[:, 2])
        context_mask = context_atomic_numbers != 0
        squared_distance = ((cb[:, None] - context_coordinates[None]) ** 2).sum(dim=-1)
        pair_mask = valid_mask[:, None] & context_mask[None]
        adjusted = squared_distance * pair_mask + (~pair_mask) * 1000.0
        nearest = adjusted.argsort(dim=-1)[:, :num_context_atoms]
        nearest_distance = adjusted.gather(1, nearest)
        selected_coordinates = context_coordinates[nearest]
        selected_atomic_numbers = context_atomic_numbers[nearest]
        selected_mask = context_mask[nearest]

        atom_count = selected_coordinates.shape[1]
        if atom_count < num_context_atoms:
            pad = num_context_atoms - atom_count
            selected_coordinates = torch.cat(
                (selected_coordinates, torch.zeros(length, pad, 3, device=self.device, dtype=self.dtype)),
                dim=1,
            )
            selected_atomic_numbers = torch.cat(
                (selected_atomic_numbers, torch.zeros(length, pad, device=self.device, dtype=torch.int64)),
                dim=1,
            )
            selected_mask = torch.cat(
                (selected_mask, torch.zeros(length, pad, device=self.device, dtype=torch.bool)),
                dim=1,
            )

        interface = valid_mask & selected_mask[:, 0] & (nearest_distance[:, 0] < ligand_interface_cutoff**2)
        if not use_atom_context:
            selected_mask = torch.zeros_like(selected_mask)
        return selected_coordinates, selected_atomic_numbers, selected_mask, interface

    def _finalize(
        self,
        structure: Structure,
        system: MolecularSystem,
        rows: _LoweredRows,
        *,
        observed: np.ndarray,
        positions: Sequence[ComponentReference] | None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        tied_weights: Mapping[ComponentReference, float] | None,
        logit_bias: Tensor | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        use_atom_context: bool,
        ligand_interface_cutoff: float,
    ) -> LigandMpnnInput:
        r"""Apply selection, shared conditions, tie closure, and context to build the input."""

        (
            row_references,
            row_backbone,
            row_residue_context_coordinates,
            row_residue_context_mask,
            row_tokens,
            row_valid,
            row_residue_indices,
            row_chain_indices,
            row_component_ids,
        ) = rows
        length = len(row_references)
        row_by_reference = {reference: index for index, reference in enumerate(row_references)}

        row_design = self._lower_design(row_references, row_valid, positions)
        tied_group_id_values, tied_groups, restrictions = self._close_conditions(conditions, row_by_reference, length)
        allowed_token_mask = self._lower_allowed(tied_groups, restrictions, row_design, row_component_ids)
        logit_bias_values = self._lower_logit_bias(
            logit_bias, residue_logit_bias, row_by_reference, length, tied_groups
        )
        tied_weight_values = self._lower_tied_weights(tied_weights, row_by_reference, tied_groups)

        backbone = torch.as_tensor(np.stack(row_backbone), device=self.device, dtype=self.dtype)
        valid_mask = torch.tensor(row_valid, device=self.device, dtype=torch.bool)
        context_coordinates_array, context_atomic_numbers_array = self._lower_context_atoms(structure, system, observed)
        context_coordinates = torch.as_tensor(context_coordinates_array, device=self.device, dtype=self.dtype)
        context_atomic_numbers = torch.as_tensor(context_atomic_numbers_array, device=self.device, dtype=torch.int64)
        (
            selected_coordinates,
            selected_atomic_numbers,
            selected_mask,
            interface,
        ) = self._select_context(
            backbone,
            valid_mask,
            context_coordinates,
            context_atomic_numbers,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
        )
        residue_context_coordinates = torch.as_tensor(
            np.stack(row_residue_context_coordinates), device=self.device, dtype=self.dtype
        )
        residue_context_mask = torch.as_tensor(np.stack(row_residue_context_mask), device=self.device)

        return LigandMpnnInput(
            backbone_coordinates=backbone.unsqueeze(0),
            sequence_tokens=torch.tensor(row_tokens, device=self.device, dtype=torch.int64).unsqueeze(0),
            valid_mask=valid_mask.unsqueeze(0),
            residue_indices=torch.tensor(row_residue_indices, device=self.device, dtype=torch.int64).unsqueeze(0),
            chain_indices=torch.tensor(row_chain_indices, device=self.device, dtype=torch.int64).unsqueeze(0),
            context_coordinates=selected_coordinates.unsqueeze(0),
            context_atomic_numbers=selected_atomic_numbers.unsqueeze(0),
            context_mask=selected_mask.unsqueeze(0),
            residue_context_coordinates=residue_context_coordinates.unsqueeze(0),
            residue_context_mask=residue_context_mask.unsqueeze(0),
            design_mask=torch.tensor(row_design, device=self.device, dtype=torch.bool).unsqueeze(0),
            ligand_interface_mask=interface.unsqueeze(0),
            structure=structure,
            component_references=tuple(row_references),
            tied_group_ids=torch.tensor(tied_group_id_values, device=self.device, dtype=torch.int64).unsqueeze(0),
            tied_weights=tied_weight_values,
            allowed_token_mask=allowed_token_mask,
            logit_bias=logit_bias_values,
        )

    def _lower_design(
        self,
        row_references: list[ComponentReference],
        row_valid: list[bool],
        positions: Sequence[ComponentReference] | None,
    ) -> list[bool]:
        r"""Resolve the designable rows from the shared ``positions`` selection meaning."""

        valid_by_reference = dict(zip(row_references, row_valid))
        if positions is None:
            selected = {reference for reference, valid in zip(row_references, row_valid) if valid}
        else:
            selected = set(positions)
            for reference in selected:
                if reference not in valid_by_reference:
                    raise ValueError(f"LigandMPNN position {reference!r} is not a protein occurrence")
                if not valid_by_reference[reference]:
                    raise ValueError(f"LigandMPNN position {reference!r} lacks its required backbone coordinates")
        return [reference in selected and valid for reference, valid in zip(row_references, row_valid)]

    def _close_conditions(
        self,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        row_by_reference: Mapping[ComponentReference, int],
        length: int,
    ) -> tuple[list[int], tuple[tuple[int, ...], ...], dict[int, set[str]]]:
        r"""Close transitive ties once and collect explicit allowed-identity restrictions."""

        parent = list(range(length))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        restrictions: dict[int, set[str]] = {}
        tied_rows: set[int] = set()
        for condition in conditions:
            if isinstance(condition, TiedResidueIdentities):
                for group in condition.groups:
                    members = [self._resolve(reference, row_by_reference) for reference in group]
                    unique_members = tuple(dict.fromkeys(members))
                    if len(unique_members) < 2:
                        continue
                    tied_rows.update(unique_members)
                    root = find(unique_members[0])
                    for member in unique_members[1:]:
                        member_root = find(member)
                        if root != member_root:
                            parent[member_root] = root
            else:
                for reference in condition.positions:
                    row = self._resolve(reference, row_by_reference)
                    allowed = set(condition.component_ids)
                    if row in restrictions:
                        restrictions[row].intersection_update(allowed)
                    else:
                        restrictions[row] = allowed

        groups_by_root: dict[int, list[int]] = {}
        for row in sorted(tied_rows):
            groups_by_root.setdefault(find(row), []).append(row)
        tied_groups = tuple(tuple(members) for members in groups_by_root.values() if len(members) > 1)

        group_id_by_root: dict[int, int] = {}
        tied_group_id_values: list[int] = []
        for row in range(length):
            root = find(row)
            if root not in group_id_by_root:
                group_id_by_root[root] = len(group_id_by_root)
            tied_group_id_values.append(group_id_by_root[root])

        return tied_group_id_values, tied_groups, restrictions

    @staticmethod
    def _resolve(
        reference: ComponentReference,
        row_by_reference: Mapping[ComponentReference, int],
    ) -> int:
        r"""Map one condition reference to its Model row or fail with a clear message."""

        try:
            return row_by_reference[reference]
        except KeyError:
            raise ValueError(
                f"LigandMPNN condition references a residue outside the protein layout: {reference!r}"
            ) from None

    def _lower_allowed(
        self,
        tied_groups: tuple[tuple[int, ...], ...],
        restrictions: dict[int, set[str]],
        row_design: list[bool],
        row_component_ids: list[str],
    ) -> Tensor | None:
        r"""Intersect explicit allowed identities and fixed anchors into a support mask.

        Restrictions come only from explicit ``AllowedResidueIdentities`` on those
        rows; a designable tied member without an explicit restriction inherits none.
        A fixed tied member (excluded by ``positions``) anchors its exact identity.
        Unknown is always outside generated support.
        """

        generatable_ids = set(AMINO_ACID_COMPONENT_IDS)
        support_by_row: dict[int, set[str]] = {}
        grouped_rows: set[int] = set()
        for group in tied_groups:
            grouped_rows.update(group)
            group_restrictions = [restrictions[row] for row in group if row in restrictions]
            anchors = {row_component_ids[row] for row in group if not row_design[row]}
            if len(anchors) > 1:
                raise ValueError("a tied LigandMPNN group has conflicting fixed identities")
            support = set(generatable_ids)
            for restriction in group_restrictions:
                support.intersection_update(restriction)
            if anchors:
                support.intersection_update(anchors)
            if not support:
                raise ValueError("a tied LigandMPNN group has no generatable identity")
            if group_restrictions or anchors:
                for row in group:
                    support_by_row[row] = support

        for row, restriction in restrictions.items():
            if row in grouped_rows:
                continue
            support = generatable_ids.intersection(restriction)
            if not row_design[row]:
                support.intersection_update({row_component_ids[row]})
            if not support:
                raise ValueError("LigandMPNN allowed identities contain no generatable identity")
            support_by_row[row] = support

        if not support_by_row:
            return None
        allowed_token_mask = torch.ones(1, len(row_component_ids), _VOCABULARY, dtype=torch.bool, device=self.device)
        for row, support in support_by_row.items():
            allowed_token_mask[0, row] = False
            for component_id in support:
                allowed_token_mask[0, row, AMINO_ACID_TOKEN_BY_COMPONENT[component_id]] = True
        return allowed_token_mask

    def _lower_logit_bias(
        self,
        logit_bias: Tensor | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        row_by_reference: Mapping[ComponentReference, int],
        length: int,
        tied_groups: tuple[tuple[int, ...], ...],
    ) -> Tensor | None:
        r"""Sum the broadcast global vector and sparse per-residue additions once.

        The public bias axis carries only the twenty generatable identities; the
        inert unknown slot stays zero. The aligned per-member bias must be identical
        across a tied group after sparse defaults.
        """

        if logit_bias is None and not residue_logit_bias:
            return None
        bias_values = torch.zeros(1, length, _VOCABULARY, device=self.device, dtype=self.dtype)
        if logit_bias is not None:
            converted = logit_bias.to(device=self.device, dtype=self.dtype)
            if converted.shape != (_GENERATED_IDENTITIES,):
                raise ValueError(f"LigandMPNN logit_bias must have shape ({_GENERATED_IDENTITIES},)")
            bias_values[0, :, :_GENERATED_IDENTITIES] = converted
        if residue_logit_bias:
            for reference, value in residue_logit_bias.items():
                row = self._resolve(reference, row_by_reference)
                converted = value.to(device=self.device, dtype=self.dtype)
                if converted.shape != (_GENERATED_IDENTITIES,):
                    raise ValueError(f"LigandMPNN residue logit bias must have shape ({_GENERATED_IDENTITIES},)")
                bias_values[0, row, :_GENERATED_IDENTITIES] += converted
        for group in tied_groups:
            members = torch.tensor(group, device=self.device)
            group_values = bias_values[0, members]
            if not torch.equal(group_values, group_values[:1].expand_as(group_values)):
                raise ValueError("tied LigandMPNN residues require identical logit bias")
        return bias_values

    def _lower_tied_weights(
        self,
        tied_weights: Mapping[ComponentReference, float] | None,
        row_by_reference: Mapping[ComponentReference, int],
        tied_groups: tuple[tuple[int, ...], ...],
    ) -> Tensor | None:
        r"""Lower final signed tie coefficients; missing members default to one."""

        if tied_weights is None:
            return None
        grouped_rows = {row for group in tied_groups for row in group}
        length = len(row_by_reference)
        tied_weight_values = torch.ones(1, length, device=self.device, dtype=self.dtype)
        for reference, value in tied_weights.items():
            row = self._resolve(reference, row_by_reference)
            if row not in grouped_rows:
                raise ValueError("LigandMPNN tied weights require a tied residue")
            tied_weight_values[0, row] = value
        return tied_weight_values
