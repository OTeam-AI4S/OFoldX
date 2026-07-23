r"""Canonical Structure lowering for ProteinMPNN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from ...data.components import get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    AtomSite,
    ChemicalAtom,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    Molecule,
    PolymerSequence,
)
from ...data.structure import ResidueNumber, Structure
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...processor import Processor
from .config import ProteinMpnnProcessorConfig
from .model import ProteinMpnnInput

_BACKBONE_ATOMS = ("N", "CA", "C", "O")


class _LoweredRows(NamedTuple):
    references: tuple[ComponentReference | None, ...]
    coordinates: tuple[np.ndarray, ...]
    tokens: tuple[int, ...]
    valid: tuple[bool, ...]
    residue_index: tuple[int, ...]
    chain_index: tuple[int, ...]
    design: tuple[bool, ...]
    component_ids: tuple[str | None, ...]


class ProteinMpnnProcessor(Processor):
    r"""Lower one canonical protein structure to ProteinMPNN tensors.

    Args:
        config: ProteinMPNN preprocessing options. Defaults to ``ProteinMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    def __init__(
        self,
        config: ProteinMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = ProteinMpnnProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        input: Structure | str | PathLike[str],
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
    ) -> ProteinMpnnInput:
        r"""Lower a PDB path or canonical ``Structure`` to ``ProteinMpnnInput``.

        Args:
            input: Canonical ``Structure`` or local PDB path.
            sequence: Optional replacement sequence. A bare one-letter string requires one protein
                instance; a mapping may provide one-letter strings or component-ID sequences by
                instance.
            design_positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with the required backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 21-token vocabulary.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires matching
                ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 21-token vocabulary.
            pssm_log_odds: Per-residue log-odds vectors over the 21-token vocabulary.

        Returns:
            A single-example ``ProteinMpnnInput`` carrying aligned tensors, source structure, and
            component references.

        Notes:
            Paths use ``process_pdb`` fixed-column row mapping; ``Structure`` values use
            ``process_structure`` observed-CA row mapping.
        """

        if isinstance(input, (str, PathLike)):
            return self.process_pdb(
                input,
                chains=None,
                sequence=sequence,
                design_positions=design_positions,
                conditions=conditions,
                tied_weights=tied_weights,
                residue_logit_bias=residue_logit_bias,
                pssm_coefficients=pssm_coefficients,
                pssm_probabilities=pssm_probabilities,
                pssm_log_odds=pssm_log_odds,
            )
        return self.process_structure(
            input,
            sequence=sequence,
            design_positions=design_positions,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
            filter_by_occupancy=True,
        )

    def process_pdb(
        self,
        path: str | PathLike[str],
        *,
        chains: Sequence[str] | None = None,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
    ) -> ProteinMpnnInput:
        r"""Lower a PDB path with ProteinMPNN fixed-column semantics.

        Args:
            path: Local PDB path.
            chains: Optional chain identifiers retained from the PDB.
            sequence: Optional replacement sequence. A bare one-letter string requires one protein
                instance; a mapping may provide one-letter strings or component-ID sequences by
                instance.
            design_positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with the required backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 21-token vocabulary.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires matching
                ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 21-token vocabulary.
            pssm_log_odds: Per-residue log-odds vectors over the 21-token vocabulary.

        Returns:
            A single-example ``ProteinMpnnInput`` lowered from the selected PDB chains.

        Notes:
            Rows order designed chains before remaining chains, with each group sorted by chain
            identifier. Integer residue-number gaps become masked rows with ``None`` component
            references; ``residue_index`` follows ``100 * chain_index + global_row_index``.
        """

        structure = _read_proteinmpnn_pdb(path, chains)
        structure, system = self._resolve_target(structure, sequence)
        rows = self._lower_pdb_rows(structure, system, design_positions)
        return self._finalize(
            structure,
            system,
            rows,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
        )

    def process_structure(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> ProteinMpnnInput:
        r"""Lower a canonical ``Structure`` to ``ProteinMpnnInput``.

        Args:
            structure: Coordinate-bearing canonical protein structure.
            sequence: Optional replacement sequence. A bare one-letter string requires one protein
                instance; a mapping may provide one-letter strings or component-ID sequences by
                instance.
            design_positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with the required backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 21-token vocabulary.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires matching
                ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 21-token vocabulary.
            pssm_log_odds: Per-residue log-odds vectors over the 21-token vocabulary.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.

        Returns:
            A single-example ``ProteinMpnnInput`` lowered from the canonical structure.

        Notes:
            Rows follow observed CA sites in ``structure.sites`` order, and
            ``component_references`` uses the same axis. ``residue_index`` uses component indices
            when source residue numbers are absent and source numbers with duplicate offsets when
            present.
        """

        structure, system = self._resolve_target(structure, sequence)
        rows = self._lower_structure_rows(structure, system, design_positions, filter_by_occupancy)
        return self._finalize(
            structure,
            system,
            rows,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
        )

    def _resolve_target(
        self,
        structure: Structure,
        sequence: str | Mapping[str, str | Sequence[str]] | None,
    ) -> tuple[Structure, MolecularSystem]:
        r"""Apply an optional target sequence, rebuilding chemistry when identities change."""

        system = structure.system
        protein_instance_list: list[tuple[str, PolymerSequence]] = []
        for instance_id, entity_id in system.instances.items():
            entity = system.entities[entity_id]
            if isinstance(entity, PolymerSequence) and entity.polymer_type == "protein":
                protein_instance_list.append((instance_id, entity))
        protein_instances = tuple(protein_instance_list)
        if not protein_instances:
            raise ValueError("ProteinMPNN requires at least one protein instance")

        if isinstance(sequence, str):
            if len(protein_instances) != 1:
                raise ValueError("a bare ProteinMPNN sequence requires exactly one protein instance")
            sequence = {protein_instances[0][0]: sequence}

        target_components = {instance_id: entity.components for instance_id, entity in protein_instances}
        if sequence is not None:
            for instance_id, value in sequence.items():
                entity = system.entities[system.instances[instance_id]]
                if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                    raise ValueError(f"ProteinMPNN sequence instance {instance_id!r} is not a protein")
                if isinstance(value, str):
                    requested = PolymerSequence.from_one_letter(value).components
                else:
                    requested = tuple(get_component(component_id) for component_id in value)
                if len(requested) != len(entity):
                    raise ValueError(
                        f"ProteinMPNN sequence for instance {instance_id!r} has length "
                        f"{len(requested)}, expected {len(entity)}"
                    )
                target_components[instance_id] = tuple(
                    original if original.component_id == replacement.component_id else replacement
                    for original, replacement in zip(entity.components, requested)
                )

        changed_references = {
            (instance_id, component_index)
            for instance_id, entity in protein_instances
            for component_index, (original, target) in enumerate(zip(entity.components, target_components[instance_id]))
            if original.component_id != target.component_id
        }
        if not changed_references:
            return structure, system

        replacements = {reference: target_components[reference[0]][reference[1]] for reference in changed_references}
        system = system.replace_components(replacements)

        consumed_atoms = ("CA",) if self.config.ca_only else _BACKBONE_ATOMS
        indices: list[int] = []
        for index, site in enumerate(structure.sites):
            reference = (site.instance_id, site.component_index)
            if reference in changed_references and site.atom_id not in consumed_atoms:
                continue
            system.atom(site)
            indices.append(index)
        index_array = np.asarray(indices, dtype=np.int64)
        structure = Structure(
            system,
            tuple(structure.sites[index] for index in indices),
            structure.coordinates[index_array],
            structure.coordinate_mask[index_array],
            None if structure.occupancy is None else structure.occupancy[index_array],
            structure.residue_numbers,
        )
        return structure, system

    def _lower_pdb_rows(
        self,
        structure: Structure,
        system: MolecularSystem,
        design_positions: Sequence[ComponentReference] | None,
    ) -> _LoweredRows:
        r"""Build fixed-column rows: designed chains first, numeric gaps, ``100 * c + i`` positions."""

        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        atom_names = ("CA",) if self.config.ca_only else _BACKBONE_ATOMS
        coordinates_by_reference: dict[ComponentReference, np.ndarray] = {}
        valid_by_reference: dict[ComponentReference, bool] = {}
        protein_by_instance: dict[str, PolymerSequence] = {}
        for instance_id, entity_id in system.instances.items():
            entity = system.entities[entity_id]
            if not isinstance(entity, PolymerSequence) or entity.polymer_type != "protein":
                continue
            protein_by_instance[instance_id] = entity
            for component_index in range(len(entity)):
                reference = (instance_id, component_index)
                atom_coordinates = np.zeros(
                    (len(atom_names), 3),
                    dtype=structure.coordinates.dtype,
                )
                valid = True
                for atom_index, atom_id in enumerate(atom_names):
                    index = coordinate_index.get(AtomSite(instance_id, component_index, atom_id))
                    observed = (
                        index is not None
                        and bool(structure.coordinate_mask[index])
                        and bool(np.isfinite(structure.coordinates[index]).all())
                    )
                    if observed:
                        atom_coordinates[atom_index] = structure.coordinates[index]
                    else:
                        valid = False
                coordinates_by_reference[reference] = atom_coordinates[0] if self.config.ca_only else atom_coordinates
                valid_by_reference[reference] = valid

        if design_positions is None:
            selected_references = {reference for reference, valid in valid_by_reference.items() if valid}
        else:
            selected_references = set(design_positions)
            for selected_reference in selected_references:
                if selected_reference not in valid_by_reference:
                    raise ValueError(f"ProteinMPNN position {selected_reference!r} is not a protein occurrence")
                if not valid_by_reference[selected_reference]:
                    raise ValueError(
                        f"ProteinMPNN position {selected_reference!r} lacks its required backbone coordinates"
                    )

        designed_instances = {instance_id for instance_id, _component_index in selected_references}
        chain_order = (
            *sorted(designed_instances),
            *sorted(set(protein_by_instance) - designed_instances),
        )
        row_coordinates: list[np.ndarray] = []
        row_tokens: list[int] = []
        row_valid: list[bool] = []
        row_residue_index: list[int] = []
        row_chain_index: list[int] = []
        row_design: list[bool] = []
        row_references: list[ComponentReference | None] = []
        row_component_ids: list[str | None] = []
        for chain_index, instance_id in enumerate(chain_order):
            entity = protein_by_instance[instance_id]
            references = tuple((instance_id, component_index) for component_index in range(len(entity)))
            if structure.residue_numbers is None:
                chain_rows: tuple[ComponentReference | None, ...] = references
            else:
                numbered = [(reference, structure.residue_numbers[reference]) for reference in references]
                numbered.sort(key=lambda item: (item[1].number, item[1].insertion_code))
                references_by_number: dict[int, list[ComponentReference]] = {}
                for numbered_reference, residue_number in numbered:
                    references_by_number.setdefault(residue_number.number, []).append(numbered_reference)
                mutable_chain_rows: list[ComponentReference | None] = []
                if numbered:
                    for number in range(numbered[0][1].number, numbered[-1][1].number + 1):
                        occurrences = references_by_number.get(number)
                        if occurrences is None:
                            mutable_chain_rows.append(None)
                        else:
                            mutable_chain_rows.extend(occurrences)
                chain_rows = tuple(mutable_chain_rows)

            for row_reference in chain_rows:
                global_row_index = len(row_references)
                row_references.append(row_reference)
                row_residue_index.append(100 * chain_index + global_row_index)
                row_chain_index.append(chain_index)
                if row_reference is None:
                    coordinate_shape = (3,) if self.config.ca_only else (4, 3)
                    row_coordinates.append(np.zeros(coordinate_shape, dtype=structure.coordinates.dtype))
                    row_tokens.append(AMINO_ACID_UNKNOWN_TOKEN)
                    row_valid.append(False)
                    row_design.append(False)
                    row_component_ids.append(None)
                    continue

                component_id = system.component(row_reference).component_id
                token_component_id = "MET" if component_id == "MSE" else component_id
                row_coordinates.append(coordinates_by_reference[row_reference])
                row_tokens.append(
                    AMINO_ACID_TOKEN_BY_COMPONENT.get(
                        token_component_id,
                        AMINO_ACID_UNKNOWN_TOKEN,
                    )
                )
                row_valid.append(valid_by_reference[row_reference])
                row_design.append(row_reference in selected_references and valid_by_reference[row_reference])
                row_component_ids.append(component_id)

        return _LoweredRows(
            tuple(row_references),
            tuple(row_coordinates),
            tuple(row_tokens),
            tuple(row_valid),
            tuple(row_residue_index),
            tuple(row_chain_index),
            tuple(row_design),
            tuple(row_component_ids),
        )

    def _lower_structure_rows(
        self,
        structure: Structure,
        system: MolecularSystem,
        design_positions: Sequence[ComponentReference] | None,
        filter_by_occupancy: bool,
    ) -> _LoweredRows:
        r"""Build updated rows: observed-CA occurrences in source order, cumulative positions."""

        observed = structure.coordinate_mask
        if filter_by_occupancy and structure.occupancy is not None:
            observed = observed & (np.isnan(structure.occupancy) | (structure.occupancy > 0))

        coordinate_index = {site: index for index, site in enumerate(structure.sites)}
        atom_names = ("CA",) if self.config.ca_only else _BACKBONE_ATOMS

        row_references: list[ComponentReference | None] = []
        row_coordinates: list[np.ndarray] = []
        row_valid: list[bool] = []
        row_component_ids: list[str | None] = []
        row_chain_index: list[int] = []
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

            atom_coordinates = np.zeros(
                (len(atom_names), 3),
                dtype=structure.coordinates.dtype,
            )
            valid = True
            for atom_index, atom_id in enumerate(atom_names):
                atom_position = coordinate_index.get(AtomSite(site.instance_id, site.component_index, atom_id))
                if atom_position is not None and bool(observed[atom_position]):
                    atom_coordinates[atom_index] = structure.coordinates[atom_position]
                else:
                    valid = False

            row_references.append(reference)
            row_coordinates.append(atom_coordinates[0] if self.config.ca_only else atom_coordinates)
            row_valid.append(valid)
            row_component_ids.append(system.component(reference).component_id)
            row_chain_index.append(chain_label_by_instance[site.instance_id])
            row_source_numbers.append(
                None if structure.residue_numbers is None else structure.residue_numbers[reference].number
            )

        row_residue_index: list[int] = []
        if structure.residue_numbers is None:
            for row_reference in row_references:
                assert row_reference is not None
                row_residue_index.append(row_reference[1])
        else:
            duplicate_count = 0
            previous_number: int | None = None
            for source_number in row_source_numbers:
                assert source_number is not None
                if source_number == previous_number:
                    duplicate_count += 1
                row_residue_index.append(source_number + duplicate_count)
                previous_number = source_number

        row_tokens = [
            (
                AMINO_ACID_UNKNOWN_TOKEN
                if component_id is None
                else AMINO_ACID_TOKEN_BY_COMPONENT.get(component_id, AMINO_ACID_UNKNOWN_TOKEN)
            )
            for component_id in row_component_ids
        ]

        valid_by_reference = dict(zip(row_references, row_valid))
        if design_positions is None:
            selected_references = {reference for reference, valid in zip(row_references, row_valid) if valid}
        else:
            selected_references = set(design_positions)
            for selected_reference in selected_references:
                if selected_reference not in valid_by_reference:
                    raise ValueError(f"ProteinMPNN position {selected_reference!r} is not a protein occurrence")
                if not valid_by_reference[selected_reference]:
                    raise ValueError(
                        f"ProteinMPNN position {selected_reference!r} lacks its required backbone coordinates"
                    )
        row_design = [reference in selected_references and valid for reference, valid in zip(row_references, row_valid)]

        return _LoweredRows(
            tuple(row_references),
            tuple(row_coordinates),
            tuple(row_tokens),
            tuple(row_valid),
            tuple(row_residue_index),
            tuple(row_chain_index),
            tuple(row_design),
            tuple(row_component_ids),
        )

    def _finalize(
        self,
        structure: Structure,
        system: MolecularSystem,
        rows: _LoweredRows,
        *,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        tied_weights: Mapping[ComponentReference, float] | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        pssm_coefficients: Mapping[ComponentReference, float] | None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None,
    ) -> ProteinMpnnInput:
        r"""Apply shared conditions, tie closure, and evidence to build the arm-agnostic input."""

        (
            row_references,
            row_coordinates,
            row_tokens,
            row_valid,
            row_residue_index,
            row_chain_index,
            row_design,
            row_component_ids,
        ) = rows

        length = len(row_references)
        row_by_reference = {reference: index for index, reference in enumerate(row_references) if reference is not None}
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
                for reference_group in condition.groups:
                    member_rows = [row_by_reference[reference] for reference in reference_group]
                    unique_members = tuple(dict.fromkeys(member_rows))
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
                    row = row_by_reference[reference]
                    allowed = set(condition.component_ids)
                    if row in restrictions:
                        restrictions[row].intersection_update(allowed)
                    else:
                        restrictions[row] = allowed

        groups_by_root: dict[int, list[int]] = {}
        for row in sorted(tied_rows):
            groups_by_root.setdefault(find(row), []).append(row)
        tied_groups = tuple(tuple(row_members) for row_members in groups_by_root.values() if len(row_members) > 1)

        group_id_by_root: dict[int, int] = {}
        tied_group_id_values: list[int] = []
        for row in range(length):
            root = find(row)
            if root not in group_id_by_root:
                group_id_by_root[root] = len(group_id_by_root)
            tied_group_id_values.append(group_id_by_root[root])

        generatable_ids = set(AMINO_ACID_COMPONENT_IDS)
        support_by_row: dict[int, set[str]] = {}
        grouped_rows: set[int] = set()
        for row_group in tied_groups:
            grouped_rows.update(row_group)
            group_restrictions = [restrictions[row] for row in row_group if row in restrictions]
            anchors = {
                component_id
                for row in row_group
                if not row_design[row] and (component_id := row_component_ids[row]) is not None
            }
            if len(anchors) > 1:
                raise ValueError("a tied ProteinMPNN group has conflicting fixed identities")
            support = set(generatable_ids)
            for restriction in group_restrictions:
                support.intersection_update(restriction)
            if anchors:
                support.intersection_update(anchors)
            if not support:
                raise ValueError("a tied ProteinMPNN group has no generatable identity")
            if group_restrictions or anchors:
                for row in row_group:
                    support_by_row[row] = support

        for row, restriction in restrictions.items():
            if row in grouped_rows:
                continue
            support = generatable_ids.intersection(restriction)
            if not row_design[row]:
                support.intersection_update({row_component_ids[row]})
            if not support:
                raise ValueError("ProteinMPNN allowed identities contain no generatable identity")
            support_by_row[row] = support

        if support_by_row:
            allowed_token_mask = torch.ones(
                1,
                length,
                len(AMINO_ACID_COMPONENT_IDS) + 1,
                dtype=torch.bool,
                device=self.device,
            )
            for row, support in support_by_row.items():
                allowed_token_mask[0, row] = False
                for component_id in support:
                    token = AMINO_ACID_TOKEN_BY_COMPONENT[component_id]
                    allowed_token_mask[0, row, token] = True
        else:
            allowed_token_mask = None

        if tied_weights is None:
            tied_weight_values = None
        else:
            # Stored values are the final signed tied coefficients consumed by the
            # Model. Each tied member defaults to coefficient 1/len(G), reproducing
            # the equal mean; untied singleton rows keep coefficient one. An explicit
            # assignment overrides the default, so all-ones yields the member sum.
            tied_weight_values = torch.ones(
                1,
                length,
                device=self.device,
                dtype=self.dtype,
            )
            for row_group in tied_groups:
                default_coefficient = 1.0 / len(row_group)
                for row in row_group:
                    tied_weight_values[0, row] = default_coefficient
            for reference, value in tied_weights.items():
                row = row_by_reference[reference]
                if row not in grouped_rows:
                    raise ValueError("ProteinMPNN tied weights require a tied residue")
                tied_weight_values[0, row] = value

        if residue_logit_bias:
            residue_bias_values = torch.zeros(
                1,
                length,
                len(AMINO_ACID_COMPONENT_IDS) + 1,
                device=self.device,
                dtype=self.dtype,
            )
            for reference, value in residue_logit_bias.items():
                row = row_by_reference[reference]
                converted = value.to(device=self.device, dtype=self.dtype)
                if converted.shape != (len(AMINO_ACID_COMPONENT_IDS) + 1,):
                    raise ValueError("ProteinMPNN residue logit bias must have shape (21,)")
                residue_bias_values[0, row] = converted
        else:
            residue_bias_values = None

        if pssm_probabilities:
            pssm_probability_values = torch.zeros(
                1,
                length,
                len(AMINO_ACID_COMPONENT_IDS) + 1,
                device=self.device,
                dtype=self.dtype,
            )
            pssm_coefficient_values = torch.zeros(
                1,
                length,
                device=self.device,
                dtype=self.dtype,
            )
            for reference, value in pssm_probabilities.items():
                row = row_by_reference[reference]
                converted = value.to(device=self.device, dtype=self.dtype)
                if converted.shape != (len(AMINO_ACID_COMPONENT_IDS) + 1,):
                    raise ValueError("ProteinMPNN PSSM probabilities must have shape (21,)")
                pssm_probability_values[0, row] = converted
                pssm_coefficient_values[0, row] = 1
            if pssm_coefficients:
                for reference, value in pssm_coefficients.items():
                    if reference not in pssm_probabilities:
                        raise ValueError("ProteinMPNN PSSM coefficients require a probability at the same residue")
                    pssm_coefficient_values[0, row_by_reference[reference]] = value
        else:
            pssm_probability_values = None
            pssm_coefficient_values = None
            if pssm_coefficients:
                raise ValueError("ProteinMPNN PSSM coefficients require PSSM probabilities")

        if pssm_log_odds:
            pssm_log_odds_values = torch.full(
                (1, length, len(AMINO_ACID_COMPONENT_IDS) + 1),
                float("inf"),
                device=self.device,
                dtype=self.dtype,
            )
            for reference, value in pssm_log_odds.items():
                row = row_by_reference[reference]
                converted = value.to(device=self.device, dtype=self.dtype)
                if converted.shape != (len(AMINO_ACID_COMPONENT_IDS) + 1,):
                    raise ValueError("ProteinMPNN PSSM log odds must have shape (21,)")
                pssm_log_odds_values[0, row] = converted
        else:
            pssm_log_odds_values = None

        for row_group in tied_groups:
            member_indices = torch.tensor(row_group, device=self.device)
            if residue_bias_values is not None:
                group_values = residue_bias_values[0, member_indices]
                if not torch.equal(group_values, group_values[:1].expand_as(group_values)):
                    raise ValueError("tied ProteinMPNN residues require identical residue logit bias")
            if pssm_coefficient_values is not None:
                group_values = pssm_coefficient_values[0, member_indices]
                if not torch.equal(group_values, group_values[:1].expand_as(group_values)):
                    raise ValueError("tied ProteinMPNN residues require identical PSSM coefficients")
            if pssm_probability_values is not None:
                group_values = pssm_probability_values[0, member_indices]
                if not torch.equal(group_values, group_values[:1].expand_as(group_values)):
                    raise ValueError("tied ProteinMPNN residues require identical PSSM probabilities")

        return ProteinMpnnInput(
            backbone_coordinates=torch.as_tensor(
                np.stack(row_coordinates),
                device=self.device,
                dtype=self.dtype,
            ).unsqueeze(0),
            sequence_tokens=torch.tensor(
                row_tokens,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            residue_mask=torch.tensor(
                row_valid,
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(0),
            residue_index=torch.tensor(
                row_residue_index,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            chain_index=torch.tensor(
                row_chain_index,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            design_positions=torch.tensor(
                row_design,
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(0),
            structures=(structure,),
            component_references=(tuple(row_references),),
            tied_group_ids=torch.tensor(
                tied_group_id_values,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
            tied_weights=tied_weight_values,
            allowed_token_mask=allowed_token_mask,
            residue_logit_bias=residue_bias_values,
            pssm_coefficients=pssm_coefficient_values,
            pssm_probabilities=pssm_probability_values,
            pssm_log_odds=pssm_log_odds_values,
        )


def _read_proteinmpnn_pdb(
    path: str | PathLike[str],
    chains: Sequence[str] | None = None,
) -> Structure:
    r"""Read the ProteinMPNN fixed-column PDB slice."""

    residue_keys_by_chain: dict[str, list[tuple[str, int, str]]] = {}
    component_id_by_residue: dict[tuple[str, int, str], str] = {}
    atoms_by_residue: dict[
        tuple[str, int, str],
        dict[str, tuple[np.ndarray, str]],
    ] = {}
    with Path(path).expanduser().open() as handle:
        for line in handle:
            record = line[:6]
            component_id = line[17:20].strip()
            if line[:4] != "ATOM" and not (record == "HETATM" and component_id == "MSE"):
                continue

            instance_id = line[21:22]
            if chains is not None and instance_id not in chains:
                continue
            number = int(line[22:26])
            insertion_code = line[26:27].strip()
            residue_key = (instance_id, number, insertion_code)
            if residue_key not in component_id_by_residue:
                component_id_by_residue[residue_key] = component_id
                atoms_by_residue[residue_key] = {}
                residue_keys_by_chain.setdefault(instance_id, []).append(residue_key)

            atom_id = line[12:16].strip()
            parsed_coordinates = np.array([float(line[start : start + 8]) for start in (30, 38, 46)])
            residue_atoms = atoms_by_residue[residue_key]
            if atom_id not in residue_atoms:
                residue_atoms[atom_id] = (
                    parsed_coordinates,
                    line[76:78].strip().title(),
                )

    if not component_id_by_residue:
        raise ValueError("ProteinMPNN requires at least one PDB ATOM residue or HETATM MSE residue")

    entities: dict[str, PolymerSequence | Molecule] = {}
    instances: dict[str, str] = {}
    sites: list[AtomSite] = []
    coordinates: list[np.ndarray] = []
    residue_numbers: dict[ComponentReference, ResidueNumber] = {}
    for instance_id, residue_keys in residue_keys_by_chain.items():
        components: list[ChemicalComponent] = []
        for component_index, residue_key in enumerate(residue_keys):
            residue_atoms = atoms_by_residue[residue_key]
            components.append(
                ChemicalComponent(
                    component_id=component_id_by_residue[residue_key],
                    atoms=tuple(
                        ChemicalAtom(atom_id=atom_id, element=element)
                        for atom_id, (_coordinates, element) in residue_atoms.items()
                    ),
                    bonds=None,
                )
            )
            residue_numbers[(instance_id, component_index)] = ResidueNumber(
                residue_key[1],
                residue_key[2],
            )
            for atom_id, (atom_coordinates, _element) in residue_atoms.items():
                sites.append(AtomSite(instance_id, component_index, atom_id))
                coordinates.append(atom_coordinates)
        entities[instance_id] = PolymerSequence("protein", tuple(components))
        instances[instance_id] = instance_id

    return Structure(
        MolecularSystem(entities, instances),
        tuple(sites),
        np.stack(coordinates),
        np.ones(len(sites), dtype=np.bool_),
        residue_numbers=residue_numbers,
    )
