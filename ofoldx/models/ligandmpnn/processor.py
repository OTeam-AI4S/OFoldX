r"""
LigandMPNN input processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from ...data import ENTITY_PROTEIN, water_component_mask
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ComponentReference, MolecularSystem
from ...data.structure import Structure
from ...data.tokenization import TokenLayout, select_tokens, tokenize_molecular_system
from ...modules.geometry import virtual_beta_carbon
from ...modules.mpnn import (
    AMINO_ACID_COMPONENT_IDS,
    AMINO_ACID_TOKEN_BY_COMPONENT,
    AMINO_ACID_UNKNOWN_TOKEN,
)
from ...processor import Processor
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
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    Tensor,
    list[str],
]


class LigandMpnnProcessor(Processor):
    r"""
    Prepare protein-ligand structures and sequence constraints for LigandMPNN.

    Args:
        config: LigandMPNN context options. Defaults to ``LigandMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = LigandMpnnProcessorConfig
    config: LigandMpnnProcessorConfig

    def __init__(
        self,
        config: LigandMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        config = LigandMpnnProcessorConfig() if config is None else config
        if config.num_context_atoms < 1:
            raise ValueError("num_context_atoms must be a positive integer")
        self.config = config
        super().__init__(device=device, dtype=dtype)

    def __call__(
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
        cropper: Cropper | None = None,
        generator: torch.Generator | None = None,
    ) -> LigandMpnnInput:
        r"""
        Prepare a ``Structure`` for LigandMPNN.

        Args:
            structure: Coordinate-bearing protein and context structure.
            positions: Protein residue occurrences to design. ``None`` selects every
                occurrence with complete backbone coordinates.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            logit_bias: Global bias vector over the 20 generatable identities.
            residue_logit_bias: Per-residue bias vectors over the 20 generatable identities.
            use_atom_context: Whether nonwater, nonprotein heavy-atom context is visible to the model.
            ligand_interface_cutoff: Distance cutoff in angstroms used to mark interface residues.
            filter_by_occupancy: Whether zero-occupancy atoms are treated as unobserved.
            cropper: Optional training-time crop policy over protein residue tokens.
            generator: Optional crop random generator.

        Returns:
            A single-example ``LigandMpnnInput`` whose residue axes align with
            ``component_index``.

        Notes:
            Protein rows follow observed CA atoms in canonical atom order.
            ``component_index`` uses the same axis. Each row's context axis
            contains its nearest nonwater, nonprotein heavy atoms, padded to
            ``config.num_context_atoms``.
        """

        system = structure.system
        observed = structure.coordinate_mask
        if filter_by_occupancy:
            observed = observed & (~structure.occupancy_mask | (structure.occupancy > 0))
        rows = self._lower_structure_rows(structure, system, observed)
        row_components = rows[0].detach().cpu().tolist()
        row_design = self._lower_design(
            system,
            row_components,
            rows[5].detach().cpu().tolist(),
            positions,
        )
        if cropper is not None:
            layout = _ligand_protein_token_layout(system, rows[0])
            design_by_component = dict(zip(row_components, row_design, strict=True))
            layout = crop_token_layout(
                structure,
                layout,
                cropper,
                required_token_mask=_ligand_required_token_mask(
                    system,
                    layout,
                    conditions=conditions,
                    tied_weights=tied_weights,
                    residue_logit_bias=residue_logit_bias,
                ),
                anchor_token_mask=torch.tensor(
                    [
                        design_by_component[component]
                        for component in layout.token_component_index.detach().cpu().tolist()
                    ],
                    dtype=torch.bool,
                    device=layout.token_component_index.device,
                ),
                generator=generator,
            )
            retained_components = set(layout.token_component_index.detach().cpu().tolist())
            retained_rows = [
                index for index, component in enumerate(row_components) if component in retained_components
            ]
            rows = _retain_ligand_rows(rows, retained_rows)
            row_design = [row_design[index] for index in retained_rows]

        (
            row_component_index,
            row_backbone,
            row_residue_context_coordinates,
            row_residue_context_mask,
            row_tokens,
            row_valid,
            row_residue_index,
            row_chain_index,
            row_component_ids,
        ) = rows
        length = row_component_index.numel()
        row_components = row_component_index.detach().cpu().tolist()
        row_by_component = {component: row for row, component in enumerate(row_components)}
        tied_group_ids, tied_groups, restrictions = self._close_conditions(
            conditions,
            system,
            row_by_component,
            length,
        )
        allowed_token_mask = self._lower_allowed(tied_groups, restrictions, row_design, row_component_ids)
        logit_bias_values = self._lower_logit_bias(
            logit_bias,
            residue_logit_bias,
            system,
            row_by_component,
            length,
            tied_groups,
        )
        tied_weight_values = self._lower_tied_weights(
            tied_weights,
            system,
            row_by_component,
            tied_groups,
        )

        backbone = row_backbone.to(device=self.device, dtype=self.dtype)
        residue_mask = row_valid.to(device=self.device)
        context_coordinates, context_atomic_numbers = self._lower_context_atoms(
            structure,
            system,
            observed,
        )
        context_coordinates = context_coordinates.to(device=self.device, dtype=self.dtype)
        context_atomic_numbers = context_atomic_numbers.to(device=self.device)
        selected_coordinates, selected_atomic_numbers, selected_mask, interface = self._select_context(
            backbone,
            residue_mask,
            context_coordinates,
            context_atomic_numbers,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
        )
        residue_context_coordinates = row_residue_context_coordinates.to(device=self.device, dtype=self.dtype)
        residue_context_mask = row_residue_context_mask.to(device=self.device)

        return LigandMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=backbone.unsqueeze(0),
            sequence_tokens=row_tokens.to(device=self.device).unsqueeze(0),
            residue_mask=residue_mask.unsqueeze(0),
            residue_index=row_residue_index.to(device=self.device).unsqueeze(0),
            chain_index=row_chain_index.to(device=self.device).unsqueeze(0),
            context_coordinates=selected_coordinates.unsqueeze(0),
            context_atomic_numbers=selected_atomic_numbers.unsqueeze(0),
            context_mask=selected_mask.unsqueeze(0),
            residue_context_coordinates=residue_context_coordinates.unsqueeze(0),
            residue_context_mask=residue_context_mask.unsqueeze(0),
            design_positions=torch.tensor(row_design, device=self.device, dtype=torch.bool).unsqueeze(0),
            ligand_interface_mask=interface.unsqueeze(0),
            structure=structure,
            component_index=row_component_index.to(device=self.device).unsqueeze(0),
            tied_group_ids=torch.tensor(tied_group_ids, device=self.device, dtype=torch.int64).unsqueeze(0),
            tied_weights=tied_weight_values,
            allowed_token_mask=allowed_token_mask,
            logit_bias=logit_bias_values,
            batch_size=[],
        )

    def _lower_structure_rows(
        self,
        structure: Structure,
        system: MolecularSystem,
        observed: Tensor,
    ) -> _LoweredRows:
        r"""
        Build rows: observed-CA protein occurrences in CA-site order, cumulative positions.
        """

        device = system.atom_symbol.device
        component_count = system.component_symbol.numel()
        component_instance = torch.repeat_interleave(
            torch.arange(
                system.instance_symbol.numel(),
                dtype=torch.int64,
                device=device,
            ),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        atom_component = torch.repeat_interleave(
            torch.arange(component_count, dtype=torch.int64, device=device),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        atom_instance = component_instance[atom_component]
        protein_instance = system.entity_type[system.instance_entity] == ENTITY_PROTEIN
        symbol_index = {symbol: index for index, symbol in enumerate(system.symbols)}
        ca_symbol = symbol_index.get("CA")
        if ca_symbol is None:
            row_atom = system.atom_symbol.new_empty((0,))
        else:
            row_atom = torch.nonzero(
                observed & protein_instance[atom_instance] & (system.atom_symbol == ca_symbol),
                as_tuple=False,
            ).flatten()
        row_component = atom_component[row_atom]
        if not row_component.numel():
            raise ValueError("LigandMPNN requires at least one protein residue with an observed CA")

        length = row_component.numel()
        row_by_component = torch.full(
            (component_count,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        row_by_component[row_component] = torch.arange(
            length,
            dtype=torch.int64,
            device=device,
        )
        atom37 = structure.coordinates.new_zeros((length, len(_ATOM37), 3))
        atom37_mask = torch.zeros(
            (length, len(_ATOM37)),
            dtype=torch.bool,
            device=observed.device,
        )
        for atom37_index, atom_id in enumerate(_ATOM37):
            if atom_id == "OXT" or atom_id not in symbol_index:
                continue
            matching_atom = torch.nonzero(
                (system.atom_symbol == symbol_index[atom_id]) & (row_by_component[atom_component] >= 0),
                as_tuple=False,
            ).flatten()
            matching_row = row_by_component[atom_component[matching_atom]]
            atom37[matching_row, atom37_index] = structure.coordinates[matching_atom]
            atom37_mask[matching_row, atom37_index] = observed[matching_atom]

        backbone_indices = torch.tensor(
            _BACKBONE_ATOM37_INDICES,
            dtype=torch.int64,
            device=device,
        )
        row_backbone = atom37[:, backbone_indices]
        row_valid = atom37_mask[:, backbone_indices].all(dim=-1)
        if not bool(row_valid.any()):
            raise ValueError("LigandMPNN requires at least one residue with complete N/CA/C/O coordinates")

        row_instance = component_instance[row_component]
        row_instance_host = row_instance.detach().cpu().tolist()
        chain_by_instance: dict[int, int] = {}
        row_chain_index = []
        for instance in row_instance_host:
            if instance not in chain_by_instance:
                chain_by_instance[instance] = len(chain_by_instance)
            row_chain_index.append(chain_by_instance[instance])

        row_source_id = structure.residue_id[row_component]
        has_source_id = row_source_id >= 0
        if not bool(has_source_id.any()):
            row_residue_index = (row_component - system.instance_component_ptr[row_instance]).detach().cpu().tolist()
        elif bool(has_source_id.all()):
            row_source_numbers = [
                int(structure.source_symbols[index]) for index in row_source_id.detach().cpu().tolist()
            ]
            row_residue_index = []
            duplicate_count = 0
            previous_number: int | None = None
            for source_number in row_source_numbers:
                if source_number == previous_number:
                    duplicate_count += 1
                row_residue_index.append(source_number + duplicate_count)
                previous_number = source_number
        else:
            raise ValueError("LigandMPNN requires source residue IDs for every retained row or none")

        row_component_ids = [
            system.symbols[index] for index in system.component_symbol[row_component].detach().cpu().tolist()
        ]
        row_tokens = torch.tensor(
            [
                AMINO_ACID_TOKEN_BY_COMPONENT.get(
                    component_id,
                    AMINO_ACID_UNKNOWN_TOKEN,
                )
                for component_id in row_component_ids
            ],
            dtype=torch.int64,
            device=device,
        )

        return (
            row_component,
            row_backbone,
            atom37[:, _RESIDUE_CONTEXT_START:],
            atom37_mask[:, _RESIDUE_CONTEXT_START:],
            row_tokens,
            row_valid,
            torch.tensor(row_residue_index, dtype=torch.int64, device=device),
            torch.tensor(row_chain_index, dtype=torch.int64, device=device),
            row_component_ids,
        )

    def _lower_context_atoms(
        self,
        structure: Structure,
        system: MolecularSystem,
        observed: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""
        Collect observed nonprotein, nonwater, nonhydrogen atoms as ordinary context.
        """

        device = system.atom_symbol.device
        component_instance = torch.repeat_interleave(
            torch.arange(
                system.instance_symbol.numel(),
                dtype=torch.int64,
                device=device,
            ),
            system.instance_component_ptr[1:] - system.instance_component_ptr[:-1],
        )
        atom_component = torch.repeat_interleave(
            torch.arange(
                system.component_symbol.numel(),
                dtype=torch.int64,
                device=device,
            ),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        atom_instance = component_instance[atom_component]
        protein_atom = system.entity_type[system.instance_entity[atom_instance]] == ENTITY_PROTEIN
        retained = (
            observed & ~protein_atom & ~water_component_mask(system)[atom_component] & (system.atomic_number != 1)
        )
        if not bool(retained.any()):
            return (
                structure.coordinates.new_zeros((1, 3)),
                torch.zeros(1, dtype=torch.int64, device=device),
            )
        return structure.coordinates[retained], system.atomic_number[retained].to(dtype=torch.int64)

    def _select_context(
        self,
        backbone: Tensor,
        residue_mask: Tensor,
        context_coordinates: Tensor,
        context_atomic_numbers: Tensor,
        *,
        use_atom_context: bool,
        ligand_interface_cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""
        Select the nearest padded context atoms per residue and the interface mask.
        """

        num_context_atoms = self.config.num_context_atoms
        length = backbone.shape[0]
        cb = virtual_beta_carbon(backbone[:, 0], backbone[:, 1], backbone[:, 2])
        context_mask = context_atomic_numbers != 0
        squared_distance = ((cb[:, None] - context_coordinates[None]) ** 2).sum(dim=-1)
        pair_mask = residue_mask[:, None] & context_mask[None]
        maximum = torch.where(pair_mask, squared_distance, 0).amax(dim=-1, keepdim=True)
        adjusted = torch.where(pair_mask, squared_distance, maximum + 1)
        atom_count = min(num_context_atoms, context_coordinates.shape[0])
        nearest = torch.topk(adjusted, atom_count, dim=-1, largest=False).indices
        nearest_distance = adjusted.gather(1, nearest)
        selected_coordinates = context_coordinates[nearest]
        selected_atomic_numbers = context_atomic_numbers[nearest]
        selected_mask = pair_mask.gather(1, nearest)

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

        interface = residue_mask & selected_mask[:, 0] & (nearest_distance[:, 0] < ligand_interface_cutoff**2)
        if not use_atom_context:
            selected_mask = torch.zeros_like(selected_mask)
        return selected_coordinates, selected_atomic_numbers, selected_mask, interface

    def _lower_design(
        self,
        system: MolecularSystem,
        row_components: list[int],
        row_valid: list[bool],
        positions: Sequence[ComponentReference] | None,
    ) -> list[bool]:
        r"""
        Resolve the designable rows from the shared ``positions`` selection meaning.
        """

        valid_by_component = dict(zip(row_components, row_valid, strict=True))
        if positions is None:
            selected = {component for component, valid in valid_by_component.items() if valid}
        else:
            selected = {system.component_index(reference) for reference in positions}
            for component in selected:
                reference = system.component_reference(component)
                if component not in valid_by_component:
                    raise ValueError(f"LigandMPNN position {reference!r} is not a protein occurrence")
                if not valid_by_component[component]:
                    raise ValueError(f"LigandMPNN position {reference!r} lacks its required backbone coordinates")
        return [component in selected and valid for component, valid in zip(row_components, row_valid, strict=True)]

    def _close_conditions(
        self,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        system: MolecularSystem,
        row_by_component: Mapping[int, int],
        length: int,
    ) -> tuple[list[int], tuple[tuple[int, ...], ...], dict[int, set[str]]]:
        r"""
        Close transitive ties once and collect explicit allowed-identity restrictions.
        """

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
                    members = [self._resolve(reference, system, row_by_component) for reference in group]
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
                    row = self._resolve(reference, system, row_by_component)
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
        system: MolecularSystem,
        row_by_component: Mapping[int, int],
    ) -> int:
        r"""
        Map one condition reference to its Model row or fail with a clear message.
        """

        try:
            return row_by_component[system.component_index(reference)]
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
        r"""
        Intersect explicit allowed identities and fixed anchors into a support mask.

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
        system: MolecularSystem,
        row_by_component: Mapping[int, int],
        length: int,
        tied_groups: tuple[tuple[int, ...], ...],
    ) -> Tensor | None:
        r"""
        Sum the broadcast global vector and sparse per-residue additions once.

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
                row = self._resolve(reference, system, row_by_component)
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
        system: MolecularSystem,
        row_by_component: Mapping[int, int],
        tied_groups: tuple[tuple[int, ...], ...],
    ) -> Tensor | None:
        r"""
        Lower final signed tie coefficients; missing members default to one.
        """

        if not tied_weights:
            return None
        grouped_rows = {row for group in tied_groups for row in group}
        length = len(row_by_component)
        tied_weight_values = torch.ones(1, length, device=self.device, dtype=self.dtype)
        for reference, value in tied_weights.items():
            row = self._resolve(reference, system, row_by_component)
            if row not in grouped_rows:
                raise ValueError("LigandMPNN tied weights require a tied residue")
            tied_weight_values[0, row] = value
        return tied_weight_values


def _ligand_protein_token_layout(system: MolecularSystem, row_component: Tensor) -> TokenLayout:
    r"""Build the canonical LigandMPNN protein residue axis without ligand atom tokens."""

    component_mask = torch.zeros_like(system.component_symbol, dtype=torch.bool)
    component_mask[row_component] = True
    layout = tokenize_molecular_system(
        system,
        atomize_nonstandard_polymers=False,
        component_mask=component_mask,
    )
    token_by_component = {
        component: token for token, component in enumerate(layout.token_component_index.detach().cpu().tolist())
    }
    return select_tokens(
        layout,
        layout.token_component_index.new_tensor(
            [token_by_component[component] for component in row_component.detach().cpu().tolist()]
        ),
    )


def _ligand_required_token_mask(
    system: MolecularSystem,
    layout: TokenLayout,
    *,
    conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
    tied_weights: Mapping[ComponentReference, float] | None,
    residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
) -> Tensor:
    r"""Retain every protein residue named by a LigandMPNN training constraint."""

    references: list[ComponentReference] = []
    for condition in conditions:
        if isinstance(condition, TiedResidueIdentities):
            references.extend(reference for group in condition.groups for reference in group)
        else:
            references.extend(condition.positions)
    if tied_weights:
        references.extend(tied_weights)
    if residue_logit_bias:
        references.extend(residue_logit_bias)

    required = {system.component_index(reference) for reference in references}
    layout_components = layout.token_component_index.detach().cpu().tolist()
    missing = required - set(layout_components)
    if missing:
        reference = system.component_reference(next(iter(missing)))
        raise ValueError(f"LigandMPNN constraint references a residue outside the protein layout: {reference!r}")
    return torch.tensor(
        [component in required for component in layout_components],
        dtype=torch.bool,
        device=layout.token_component_index.device,
    )


def _retain_ligand_rows(rows: _LoweredRows, retained: list[int]) -> _LoweredRows:
    r"""Filter the LigandMPNN protein axis before context-neighbour selection."""

    index = torch.tensor(retained, dtype=torch.int64, device=rows[0].device)
    return (
        rows[0][index],
        rows[1][index],
        rows[2][index],
        rows[3][index],
        rows[4][index],
        rows[5][index],
        rows[6][index],
        rows[7][index],
        [rows[8][row] for row in retained],
    )
