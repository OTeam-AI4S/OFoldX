r"""
ProteinMPNN input processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast

import torch
from torch import Tensor

from ...data.components import get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    AMINO_ACID_ONE_TO_THREE,
    ENTITY_PROTEIN,
    ComponentReference,
    instance_subgraph,
    replace_components,
)
from ...data.structure import Structure
from ...model import Model
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...processor import Processor
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import ProteinMpnnInput, _repartition_entities_by_sequence

_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_SEQUENCE_COMPONENT_ID_BY_LETTER = {
    **AMINO_ACID_ONE_TO_THREE,
    "X": "UNK",
}


class _LoweredRows(NamedTuple):
    component_index: tuple[int, ...]
    coordinates: tuple[Tensor, ...]
    tokens: tuple[int, ...]
    valid: tuple[bool, ...]
    residue_index: tuple[int, ...]
    chain_index: tuple[int, ...]
    design: tuple[bool, ...]
    model_component_ids: tuple[str | None, ...]


class ProteinMpnnProcessor(Processor):
    r"""
    Prepare protein backbones and sequence constraints for ProteinMPNN.

    Args:
        config: ProteinMPNN input options. Defaults to ``ProteinMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = ProteinMpnnProcessorConfig
    config: ProteinMpnnProcessorConfig

    def __init__(
        self,
        config: ProteinMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = ProteinMpnnProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def validate_model_input_layout(self, model: Model) -> None:
        model_config = cast(ProteinMpnnConfig, model.config)
        if self.config.ca_only != model_config.ca_only:
            raise ValueError(
                "ProteinMPNN Model and Processor must use the same ca_only layout; "
                f"got model={model_config.ca_only}, processor={self.config.ca_only}"
            )

    def __call__(
        self,
        structure: Structure,
        *,
        protein_instances: Sequence[str] | None = None,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentReference, float] | None = None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None = None,
        filter_by_occupancy: bool = True,
    ) -> ProteinMpnnInput:
        r"""
        Prepare a ``Structure`` for ProteinMPNN.

        Args:
            structure: Coordinate-bearing molecular structure.
            protein_instances: Optional protein instance IDs retained as the design target.
            sequence: Optional replacement sequence. A bare one-letter string requires one protein
                instance; a mapping may provide one-letter strings or component-ID sequences by
                instance.
            positions: Protein residue occurrences to design. ``None`` selects every
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
            A single-example ``ProteinMpnnInput`` whose residue axes align with
            ``component_index``.

        Notes:
            Without ``protein_instances``, the complete input structure is retained while only
            protein occurrences become model rows.
            Rows order designed protein instances before fixed instances. Complete source numbering
            inserts numeric gaps and retains insertion occurrences; an unnumbered instance follows
            component order. ``residue_index`` is ``100 * chain_index + global_row_index``.
        """

        if protein_instances is not None:
            structure = _select_protein_instances(structure, protein_instances)
        structure = self._resolve_target(structure, sequence)
        rows = self._lower_rows(
            structure,
            positions,
            filter_by_occupancy,
        )
        return self._finalize(
            structure,
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
    ) -> Structure:
        r"""
        Apply an optional target sequence, rebuilding chemistry when identities change.
        """

        original = structure.system
        protein_instances = torch.nonzero(
            original.entity_type[original.instance_entity] == ENTITY_PROTEIN,
            as_tuple=False,
        ).flatten()
        if not protein_instances.numel():
            raise ValueError("ProteinMPNN requires at least one protein instance")

        protein_instance_ids = tuple(original.instance_id(int(instance)) for instance in protein_instances)
        if sequence is None:
            return structure
        if isinstance(sequence, str):
            if protein_instances.numel() != 1:
                raise ValueError("a bare ProteinMPNN sequence requires exactly one protein instance")
            sequence = {
                protein_instance_ids[0]: sequence,
            }
        if set(sequence) != set(protein_instance_ids):
            raise ValueError("ProteinMPNN sequence mapping must exactly cover the selected protein instances")

        changed_components: list[int] = []
        replacements = []
        for instance_id, value in sequence.items():
            instance = original.instance_index(instance_id)
            entity = int(original.instance_entity[instance])
            if int(original.entity_type[entity]) != ENTITY_PROTEIN:
                raise ValueError(f"ProteinMPNN sequence instance {instance_id!r} is not a protein")
            start = int(original.instance_component_ptr[instance])
            stop = int(original.instance_component_ptr[instance + 1])
            if isinstance(value, str):
                try:
                    requested = tuple(get_component(_SEQUENCE_COMPONENT_ID_BY_LETTER[symbol]) for symbol in value)
                except KeyError as error:
                    raise ValueError(f"unsupported ProteinMPNN sequence symbol {error.args[0]!r}") from None
            else:
                requested = tuple(get_component(component_id) for component_id in value)
            if len(requested) != stop - start:
                raise ValueError(
                    f"ProteinMPNN sequence for instance {instance_id!r} has length "
                    f"{len(requested)}, expected {stop - start}"
                )
            for component, replacement_component in zip(
                range(start, stop),
                requested,
                strict=True,
            ):
                component_id = original.symbols[int(original.component_symbol[component])]
                if component_id != replacement_component.component_id:
                    changed_components.append(component)
                    replacements.append(replacement_component)

        if not changed_components:
            return structure

        system, old_to_new, _ = replace_components(
            original,
            torch.tensor(
                changed_components,
                dtype=torch.long,
                device=original.component_symbol.device,
            ),
            replacements,
        )
        system = _repartition_entities_by_sequence(original, system)

        num_atoms = system.atom_symbol.numel()
        coordinates = structure.coordinates.new_zeros(num_atoms, 3)
        coordinate_mask = structure.coordinate_mask.new_zeros(num_atoms)
        occupancy = structure.occupancy.new_zeros(num_atoms)
        occupancy_mask = structure.occupancy_mask.new_zeros(num_atoms)
        b_factor = structure.b_factor.new_zeros(num_atoms)
        b_factor_mask = structure.b_factor_mask.new_zeros(num_atoms)

        atom_component = torch.repeat_interleave(
            torch.arange(
                original.component_symbol.numel(),
                dtype=torch.long,
                device=original.atom_symbol.device,
            ),
            original.component_atom_ptr[1:] - original.component_atom_ptr[:-1],
        )
        changed_mask = torch.zeros_like(original.component_symbol, dtype=torch.bool)
        changed_mask[changed_components] = True
        consumed_atoms = {"CA"} if self.config.ca_only else set(_BACKBONE_ATOMS)
        consumed_mask = original.atom_symbol.new_tensor(
            [original.symbols[int(symbol)] in consumed_atoms for symbol in original.atom_symbol.tolist()],
            dtype=torch.bool,
        )
        retained = (old_to_new >= 0) & (~changed_mask[atom_component] | consumed_mask)
        destination = old_to_new[retained]
        coordinates[destination] = structure.coordinates[retained]
        coordinate_mask[destination] = structure.coordinate_mask[retained]
        occupancy[destination] = structure.occupancy[retained]
        occupancy_mask[destination] = structure.occupancy_mask[retained]
        b_factor[destination] = structure.b_factor[retained]
        b_factor_mask[destination] = structure.b_factor_mask[retained]
        standard_component_id = structure.standard_component_id.clone()
        standard_component_id[changed_components] = -1
        structure = Structure(  # type: ignore[call-arg]
            system=system,
            coordinates=coordinates,
            coordinate_mask=coordinate_mask,
            occupancy=occupancy,
            occupancy_mask=occupancy_mask,
            b_factor=b_factor,
            b_factor_mask=b_factor_mask,
            source_symbols=structure.source_symbols,
            residue_id=structure.residue_id,
            insertion_code=structure.insertion_code,
            standard_component_id=standard_component_id,
            batch_size=[],
        )
        return structure

    def _lower_rows(
        self,
        structure: Structure,
        design_positions: Sequence[ComponentReference] | None,
        filter_by_occupancy: bool,
    ) -> _LoweredRows:
        r"""
        Lower canonical protein occurrences with numbered gaps and released tensor ordering.
        """

        system = structure.system
        observed = structure.coordinate_mask & torch.isfinite(structure.coordinates).all(dim=-1)
        if filter_by_occupancy:
            observed = observed & (~structure.occupancy_mask | (structure.occupancy > 0))

        atom_names = ("CA",) if self.config.ca_only else _BACKBONE_ATOMS
        coordinates_by_component: dict[int, Tensor] = {}
        valid_by_component: dict[int, bool] = {}
        component_instance: dict[int, int] = {}
        protein_instances: list[int] = []
        for instance in range(system.instance_symbol.numel()):
            entity = int(system.instance_entity[instance])
            if int(system.entity_type[entity]) != ENTITY_PROTEIN:
                continue
            protein_instances.append(instance)
            start = int(system.instance_component_ptr[instance])
            stop = int(system.instance_component_ptr[instance + 1])
            for component in range(start, stop):
                component_instance[component] = instance
                atom_start = int(system.component_atom_ptr[component])
                atom_stop = int(system.component_atom_ptr[component + 1])
                atom_by_name = {
                    system.symbols[int(system.atom_symbol[atom])]: atom for atom in range(atom_start, atom_stop)
                }
                atom_coordinates = structure.coordinates.new_zeros(
                    len(atom_names),
                    3,
                )
                valid = True
                for feature_index, atom_id in enumerate(atom_names):
                    atom = atom_by_name.get(atom_id)
                    if atom is not None and bool(observed[atom]):
                        atom_coordinates[feature_index] = structure.coordinates[atom]
                    else:
                        valid = False
                coordinates_by_component[component] = atom_coordinates[0] if self.config.ca_only else atom_coordinates
                valid_by_component[component] = valid

        if design_positions is None:
            selected_components = {component for component, valid in valid_by_component.items() if valid}
        else:
            selected_components = {system.component_index(reference) for reference in design_positions}
            for reference in design_positions:
                component = system.component_index(reference)
                if component not in valid_by_component:
                    raise ValueError(f"ProteinMPNN position {reference!r} is not a protein occurrence")
                if not valid_by_component[component]:
                    raise ValueError(f"ProteinMPNN position {reference!r} lacks its required backbone coordinates")

        designed_instances = {component_instance[component] for component in selected_components}
        instance_order = (
            *sorted(designed_instances, key=system.instance_id),
            *sorted(
                set(protein_instances) - designed_instances,
                key=system.instance_id,
            ),
        )
        row_coordinates: list[Tensor] = []
        row_tokens: list[int] = []
        row_valid: list[bool] = []
        row_residue_index: list[int] = []
        row_chain_index: list[int] = []
        row_design: list[bool] = []
        row_component_index: list[int] = []
        row_model_component_ids: list[str | None] = []
        for chain_index, instance in enumerate(instance_order):
            component_start = int(system.instance_component_ptr[instance])
            component_stop = int(system.instance_component_ptr[instance + 1])
            components = tuple(range(component_start, component_stop))
            numbered = structure.residue_id[component_start:component_stop] >= 0
            if bool(numbered.any()) and not bool(numbered.all()):
                raise ValueError(
                    f"ProteinMPNN residue numbering for protein instance "
                    f"{system.instance_id(instance)!r} "
                    "must cover every component occurrence or none"
                )
            if not bool(numbered.any()):
                chain_rows: tuple[int | None, ...] = components
            else:
                positions = []
                for component in components:
                    residue_symbol = int(structure.residue_id[component])
                    insertion_symbol = int(structure.insertion_code[component])
                    positions.append(
                        (
                            component,
                            int(structure.source_symbols[residue_symbol]),
                            ("" if insertion_symbol < 0 else structure.source_symbols[insertion_symbol]),
                        )
                    )
                positions.sort(key=lambda item: (item[1], item[2]))
                components_by_number: dict[int, list[int]] = {}
                for component, number, _insertion_code in positions:
                    components_by_number.setdefault(number, []).append(component)
                mutable_chain_rows: list[int | None] = []
                for number in range(positions[0][1], positions[-1][1] + 1):
                    occurrences = components_by_number.get(number)
                    if occurrences is None:
                        mutable_chain_rows.append(None)
                    else:
                        mutable_chain_rows.extend(occurrences)
                chain_rows = tuple(mutable_chain_rows)

            for row_component in chain_rows:
                global_row_index = len(row_component_index)
                row_component_index.append(-1 if row_component is None else row_component)
                row_residue_index.append(100 * chain_index + global_row_index)
                row_chain_index.append(chain_index)
                if row_component is None:
                    coordinate_shape = (3,) if self.config.ca_only else (4, 3)
                    row_coordinates.append(structure.coordinates.new_zeros(coordinate_shape))
                    row_tokens.append(AMINO_ACID_UNKNOWN_TOKEN)
                    row_valid.append(False)
                    row_design.append(False)
                    row_model_component_ids.append(None)
                    continue

                component_id = system.symbols[int(system.component_symbol[row_component])]
                token_component_id = "MET" if component_id == "MSE" else component_id
                token = AMINO_ACID_TOKEN_BY_COMPONENT.get(
                    token_component_id,
                    AMINO_ACID_UNKNOWN_TOKEN,
                )
                row_coordinates.append(coordinates_by_component[row_component])
                row_tokens.append(token)
                row_model_component_ids.append(
                    "UNK" if token == AMINO_ACID_UNKNOWN_TOKEN else AMINO_ACID_COMPONENT_IDS[token]
                )
                row_valid.append(valid_by_component[row_component])
                row_design.append(row_component in selected_components and valid_by_component[row_component])

        return _LoweredRows(
            tuple(row_component_index),
            tuple(row_coordinates),
            tuple(row_tokens),
            tuple(row_valid),
            tuple(row_residue_index),
            tuple(row_chain_index),
            tuple(row_design),
            tuple(row_model_component_ids),
        )

    def _finalize(
        self,
        structure: Structure,
        rows: _LoweredRows,
        *,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        tied_weights: Mapping[ComponentReference, float] | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        pssm_coefficients: Mapping[ComponentReference, float] | None,
        pssm_probabilities: Mapping[ComponentReference, Tensor] | None,
        pssm_log_odds: Mapping[ComponentReference, Tensor] | None,
    ) -> ProteinMpnnInput:
        r"""
        Apply conditions, tie closure, and evidence to the canonical rows.
        """

        system = structure.system
        (
            row_component_index,
            row_coordinates,
            row_tokens,
            row_valid,
            row_residue_index,
            row_chain_index,
            row_design,
            row_model_component_ids,
        ) = rows

        length = len(row_component_index)
        row_by_component = {component: row for row, component in enumerate(row_component_index) if component >= 0}

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
                    member_rows = [row_by_component[system.component_index(reference)] for reference in reference_group]
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
                    row = row_by_component[system.component_index(reference)]
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

        generatable_ids = {*AMINO_ACID_COMPONENT_IDS, "UNK"}
        support_by_row: dict[int, set[str]] = {}
        grouped_rows: set[int] = set()
        for row_group in tied_groups:
            grouped_rows.update(row_group)
            group_restrictions = [restrictions[row] for row in row_group if row in restrictions]
            anchors = {
                component_id
                for row in row_group
                if not row_design[row] and (component_id := row_model_component_ids[row]) is not None
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
                support.intersection_update({row_model_component_ids[row]})
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
                    token = (
                        AMINO_ACID_UNKNOWN_TOKEN
                        if component_id == "UNK"
                        else AMINO_ACID_TOKEN_BY_COMPONENT[component_id]
                    )
                    allowed_token_mask[0, row, token] = True
        else:
            allowed_token_mask = None

        if not tied_weights:
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
            for reference, weight in tied_weights.items():
                row = row_by_component[system.component_index(reference)]
                if row not in grouped_rows:
                    raise ValueError("ProteinMPNN tied weights require a tied residue")
                tied_weight_values[0, row] = weight

        if residue_logit_bias:
            residue_bias_values = torch.zeros(
                1,
                length,
                len(AMINO_ACID_COMPONENT_IDS) + 1,
                device=self.device,
                dtype=self.dtype,
            )
            for reference, bias in residue_logit_bias.items():
                row = row_by_component[system.component_index(reference)]
                converted = bias.to(device=self.device, dtype=self.dtype)
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
            for reference, probabilities in pssm_probabilities.items():
                row = row_by_component[system.component_index(reference)]
                converted = probabilities.to(device=self.device, dtype=self.dtype)
                if converted.shape != (len(AMINO_ACID_COMPONENT_IDS) + 1,):
                    raise ValueError("ProteinMPNN PSSM probabilities must have shape (21,)")
                pssm_probability_values[0, row] = converted
                pssm_coefficient_values[0, row] = 1
            if pssm_coefficients:
                for reference, coefficient in pssm_coefficients.items():
                    if reference not in pssm_probabilities:
                        raise ValueError("ProteinMPNN PSSM coefficients require a probability at the same residue")
                    row = row_by_component[system.component_index(reference)]
                    pssm_coefficient_values[0, row] = coefficient
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
            for reference, log_odds in pssm_log_odds.items():
                row = row_by_component[system.component_index(reference)]
                converted = log_odds.to(device=self.device, dtype=self.dtype)
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

        return ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=torch.stack(row_coordinates).to(device=self.device, dtype=self.dtype).unsqueeze(0),
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
            structure=structure,
            component_index=torch.tensor(
                row_component_index,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0),
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
            batch_size=[],
        )


def _select_protein_instances(
    structure: Structure,
    protein_instances: Sequence[str],
) -> Structure:
    r"""
    Retain the exact protein instances selected as one ProteinMPNN target.
    """

    system = structure.system
    instance_mask = torch.zeros_like(system.instance_symbol, dtype=torch.bool)
    for instance_id in protein_instances:
        instance = system.instance_index(instance_id)
        entity = int(system.instance_entity[instance])
        if int(system.entity_type[entity]) != ENTITY_PROTEIN:
            raise ValueError(f"ProteinMPNN instance {instance_id!r} is not a protein")
        instance_mask[instance] = True
    if not bool(instance_mask.any()):
        raise ValueError("ProteinMPNN requires at least one selected protein instance")

    selected_system, component_to_new, atom_to_new = instance_subgraph(
        system,
        instance_mask,
    )
    component_mask = component_to_new >= 0
    atom_mask = atom_to_new >= 0
    return Structure(  # type: ignore[call-arg]
        system=selected_system,
        coordinates=structure.coordinates[atom_mask],
        coordinate_mask=structure.coordinate_mask[atom_mask],
        occupancy=structure.occupancy[atom_mask],
        occupancy_mask=structure.occupancy_mask[atom_mask],
        b_factor=structure.b_factor[atom_mask],
        b_factor_mask=structure.b_factor_mask[atom_mask],
        source_symbols=structure.source_symbols,
        residue_id=structure.residue_id[component_mask],
        insertion_code=structure.insertion_code[component_mask],
        standard_component_id=structure.standard_component_id[component_mask],
        batch_size=[],
    )
