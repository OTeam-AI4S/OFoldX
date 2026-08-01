r"""
NA-MPNN input processing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import torch
from torch import Tensor

from ...data.components import get_component
from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import (
    BOND_ROLE_COMPONENT,
    BOND_ROLE_POLYMER,
    ENTITY_DNA,
    ENTITY_DNA_RNA_HYBRID,
    ENTITY_MOLECULE,
    ENTITY_PROTEIN,
    ENTITY_RNA,
    ChemicalComponent,
    ComponentReference,
    MolecularSystem,
    one_letter_to_component,
    replace_components,
)
from ...data.structure import Structure
from ...processor import Processor
from .config import NaMpnnProcessorConfig
from .model import (
    NA_MPNN_ATOM_NAMES,
    NA_MPNN_DNA_GENERATABLE_TOKEN_IDS,
    NA_MPNN_PROTEIN_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS,
    NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS,
    NA_MPNN_TOKEN_BY_COMPONENT,
    NaMpnnInput,
    _repartition_entities_by_sequence,
)

_VOCAB_SIZE = 33
_REQUIRED_ATOMS = {
    "protein": ("N", "CA", "C", "O"),
    "dna": ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"),
    "rna": ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"),
}
_GENERATED_COMPONENTS = {
    "protein": tuple(NA_MPNN_TOKEN_BY_COMPONENT)[:20],
    "dna": ("DA", "DC", "DG", "DT"),
    "rna": ("A", "C", "G", "U"),
}
_DNA_COMPONENTS = frozenset({"DA", "DC", "DG", "DT", "DX"})
_RNA_COMPONENTS = frozenset({"A", "C", "G", "U", "RX"})
_SHARED_NUCLEIC_TOKENS = {
    "A": NA_MPNN_TOKEN_BY_COMPONENT["DA"],
    "C": NA_MPNN_TOKEN_BY_COMPONENT["DC"],
    "G": NA_MPNN_TOKEN_BY_COMPONENT["DG"],
    "U": NA_MPNN_TOKEN_BY_COMPONENT["DT"],
}
_POLYMER_TYPE_INDEX = {"protein": 0, "dna": 1, "rna": 2}
_POLYMER_TYPE_NAME = ("protein", "dna", "rna")
_TOKEN_BY_ONE_LETTER = {symbol: token for token, symbol in enumerate("ARNDCQEGHILKMFPSTWYVXacgtxbdhuy-+")}


class _Row(NamedTuple):
    r"""One aligned polymer residue consumed by NA-MPNN."""

    component_index: int
    component_id: str
    polymer_type_index: int
    chain_index: int
    residue_index: int
    atom_coordinates: Tensor
    atom_mask: Tensor
    backbone_complete: bool


class NaMpnnProcessor(Processor):
    r"""
    Prepare mixed protein and nucleic-acid structures for NA-MPNN.

    Args:
        config: NA-MPNN token options. Defaults to ``NaMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = NaMpnnProcessorConfig
    config: NaMpnnProcessorConfig

    def __init__(
        self,
        config: NaMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = NaMpnnProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def __call__(
        self,
        structure: Structure,
        *,
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        positions: Sequence[ComponentReference] | None = None,
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
    ) -> NaMpnnInput:
        r"""
        Prepare one mixed-polymer structure for scoring or sequence design.

        Args:
            structure: Coordinate-bearing mixed-polymer structure.
            sequence: Optional replacement sequence selecting the scoring arm. A bare string
                requires one polymer instance; a mapping addresses instances explicitly.
            positions: Residue occurrences to evaluate or design. ``None`` selects every
                valid, eligible residue.
            nucleic_acid_only: Whether to design only DNA and RNA while retaining protein context.
            conditions: Allowed-identity and tied-identity conditions.
            symmetry_groups: Residue groups sharing one decoding position.
            tied_weights: Optional sampling coefficient for each tied residue.
            residue_logit_bias: Per-residue bias vectors over the NA-MPNN vocabulary.
            pair_logit_bias: Optional adjacent-token bias matrix with shape ``(33, 33)``.
            global_omit_tokens: Case-sensitive official one-letter tokens excluded from generation:
                uppercase protein, lowercase nucleic acid, ``-`` for mask, and ``+`` for padding.

        Returns:
            A single-example ``NaMpnnInput`` with aligned structure, residue, and condition axes.
        """

        if sequence is not None:
            structure = self._resolve_score_target(structure, sequence)
            return self._process_score(
                structure,
                self._lower_rows(structure),
                positions=positions,
            )
        rows = self._lower_rows(structure)
        return self._process_design(
            structure,
            rows,
            positions=positions,
            nucleic_acid_only=nucleic_acid_only,
            conditions=tuple(conditions),
            symmetry_groups=tuple(tuple(group) for group in symmetry_groups),
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=tuple(global_omit_tokens),
        )

    # ------------------------------------------------------------------
    # Shared lowering.
    # ------------------------------------------------------------------
    @staticmethod
    def _polymer_type(entity_type: int, component_id: str) -> int:
        if entity_type == ENTITY_PROTEIN:
            return _POLYMER_TYPE_INDEX["protein"]
        if entity_type == ENTITY_DNA:
            return _POLYMER_TYPE_INDEX["dna"]
        if entity_type == ENTITY_RNA:
            return _POLYMER_TYPE_INDEX["rna"]
        if entity_type == ENTITY_DNA_RNA_HYBRID:
            if component_id in _DNA_COMPONENTS:
                return _POLYMER_TYPE_INDEX["dna"]
            if component_id in _RNA_COMPONENTS:
                return _POLYMER_TYPE_INDEX["rna"]
        raise ValueError("NA-MPNN cannot classify this polymer occurrence as protein, DNA, or RNA")

    def _lower_rows(self, structure: Structure) -> list[_Row]:
        r"""
        Lower every polymer residue to aligned coordinates, masks, and identity.
        """

        system = structure.system
        component_indices: list[int] = []
        component_ids: list[str] = []
        polymer_types: list[int] = []
        chain_indices: list[int] = []
        source_numbers: list[int] = []
        chain_index = 0
        component_ptr = system.instance_component_ptr.detach().cpu().tolist()
        instance_entities = system.instance_entity.detach().cpu().tolist()
        entity_types = system.entity_type.detach().cpu().tolist()
        component_symbols = system.component_symbol.detach().cpu().tolist()
        residue_ids = structure.residue_id.detach().cpu().tolist()
        for instance, entity in enumerate(instance_entities):
            entity_type = entity_types[entity]
            if entity_type == ENTITY_MOLECULE:
                continue
            start = component_ptr[instance]
            stop = component_ptr[instance + 1]
            for component in range(start, stop):
                component_id = system.symbols[component_symbols[component]]
                polymer_type = self._polymer_type(entity_type, component_id)
                component_indices.append(component)
                component_ids.append(component_id)
                polymer_types.append(polymer_type)
                chain_indices.append(chain_index)
                source_id = residue_ids[component]
                if source_id >= 0:
                    source_numbers.append(int(structure.source_symbols[source_id]))
                else:
                    source_numbers.append(component - start)
            chain_index += 1

        if not component_indices:
            raise ValueError("NA-MPNN requires at least one protein, DNA, or RNA residue")

        device = system.component_symbol.device
        component_index_tensor = torch.tensor(
            component_indices,
            dtype=torch.int64,
            device=device,
        )
        length = component_index_tensor.numel()
        atom_coordinates = structure.coordinates.new_zeros((length, len(NA_MPNN_ATOM_NAMES), 3))
        atom_mask = torch.zeros(
            (length, len(NA_MPNN_ATOM_NAMES)),
            dtype=torch.bool,
            device=structure.coordinate_mask.device,
        )
        atom_component = torch.repeat_interleave(
            torch.arange(
                system.component_symbol.numel(),
                dtype=torch.int64,
                device=device,
            ),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        row_by_component = torch.full(
            (system.component_symbol.numel(),),
            -1,
            dtype=torch.int64,
            device=device,
        )
        row_by_component[component_index_tensor] = torch.arange(
            length,
            dtype=torch.int64,
            device=device,
        )
        symbol_index = {symbol: index for index, symbol in enumerate(system.symbols)}
        slot_by_symbol = torch.full(
            (len(system.symbols),),
            -1,
            dtype=torch.int64,
            device=system.atom_symbol.device,
        )
        for slot, atom_name in enumerate(NA_MPNN_ATOM_NAMES):
            symbol = symbol_index.get(atom_name)
            if symbol is not None:
                slot_by_symbol[symbol] = slot
        atom_row = row_by_component[atom_component]
        atom_slot = slot_by_symbol[system.atom_symbol]
        retained = torch.nonzero(
            (atom_row >= 0) & (atom_slot >= 0),
            as_tuple=False,
        ).flatten()
        retained_row = atom_row[retained]
        retained_slot = atom_slot[retained]
        observed = structure.coordinate_mask & (~structure.occupancy_mask | (structure.occupancy > 0))
        retained_observed = observed[retained]
        atom_coordinates[retained_row, retained_slot] = torch.where(
            retained_observed[:, None],
            structure.coordinates[retained],
            0,
        )
        atom_mask[retained_row, retained_slot] = retained_observed

        required_slots = {
            polymer_type: torch.tensor(
                [NA_MPNN_ATOM_NAMES.index(name) for name in required],
                dtype=torch.int64,
                device=device,
            )
            for polymer_type, required in _REQUIRED_ATOMS.items()
        }
        polymer_type_tensor = torch.tensor(
            polymer_types,
            dtype=torch.int64,
            device=device,
        )
        valid_tensor = torch.zeros(length, dtype=torch.bool, device=device)
        for polymer_name, polymer_type in _POLYMER_TYPE_INDEX.items():
            selected_type = polymer_type_tensor == polymer_type
            valid_tensor[selected_type] = atom_mask[selected_type][:, required_slots[polymer_name]].all(dim=-1)
        valids = valid_tensor.detach().cpu().tolist()
        residue_index = self._renumber(source_numbers)
        rows = [
            _Row(
                component,
                component_id,
                polymer_type,
                chain,
                residue_index[position],
                atom_coordinates[position],
                atom_mask[position],
                valids[position],
            )
            for position, (component, component_id, polymer_type, chain) in enumerate(
                zip(
                    component_indices,
                    component_ids,
                    polymer_types,
                    chain_indices,
                    strict=True,
                )
            )
        ]
        if not any(row.backbone_complete for row in rows):
            raise ValueError("NA-MPNN requires at least one residue with a complete polymer backbone")
        return rows

    @staticmethod
    def _renumber(source_numbers: Sequence[int]) -> list[int]:
        r"""
        Apply running-count renumbering for repeats and insertion codes.
        """

        renumbered: list[int] = []
        count = 0
        previous: int | None = None
        for source in source_numbers:
            if source == previous:
                count += 1
            renumbered.append(source + count)
            previous = source
        return renumbered

    def _component_token(self, component_id: str, polymer_type: int) -> int:
        polymer_name = _POLYMER_TYPE_NAME[polymer_type]
        if polymer_name == "rna" and self.config.shared_nucleic_tokens:
            return _SHARED_NUCLEIC_TOKENS.get(component_id, NA_MPNN_TOKEN_BY_COMPONENT["DX"])
        unknown = {"protein": "UNK", "dna": "DX", "rna": "RX"}[polymer_name]
        return NA_MPNN_TOKEN_BY_COMPONENT.get(component_id, NA_MPNN_TOKEN_BY_COMPONENT[unknown])

    def _generatable_tokens(self, polymer_type: int) -> tuple[int, ...]:
        if polymer_type == _POLYMER_TYPE_INDEX["protein"]:
            return NA_MPNN_PROTEIN_GENERATABLE_TOKEN_IDS
        if polymer_type == _POLYMER_TYPE_INDEX["dna"]:
            return NA_MPNN_DNA_GENERATABLE_TOKEN_IDS
        return (
            NA_MPNN_RNA_SHARED_GENERATABLE_TOKEN_IDS
            if self.config.shared_nucleic_tokens
            else NA_MPNN_RNA_SEPARATE_GENERATABLE_TOKEN_IDS
        )

    def _tensor(self, values: list, dtype: torch.dtype) -> Tensor:
        return torch.tensor(values, device=self.device, dtype=dtype).unsqueeze(0)

    def _base_input(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        design_positions: list[bool],
        token_group_ids: list[int],
        sequence_tokens: list[int],
        tied_weights: Tensor,
        residue_logit_bias: Tensor,
        allowed_token_mask: Tensor,
        pair_logit_bias: Tensor | None,
    ) -> NaMpnnInput:
        return NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=torch.stack([row.atom_coordinates for row in rows])
            .to(device=self.device, dtype=self.dtype)
            .unsqueeze(0),
            atom_mask=torch.stack([row.atom_mask for row in rows]).to(device=self.device).unsqueeze(0),
            sequence_tokens=self._tensor(sequence_tokens, torch.int64),
            residue_mask=self._tensor([row.backbone_complete for row in rows], torch.bool),
            residue_index=self._tensor([row.residue_index for row in rows], torch.int64),
            chain_index=self._tensor([row.chain_index for row in rows], torch.int64),
            polymer_type_index=self._tensor(
                [row.polymer_type_index if row.backbone_complete else 3 for row in rows],
                torch.int64,
            ),
            design_positions=self._tensor(design_positions, torch.bool),
            token_group_ids=self._tensor(token_group_ids, torch.int64),
            structure=structure,
            component_index=self._tensor([row.component_index for row in rows], torch.int64),
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            allowed_token_mask=allowed_token_mask,
            pair_logit_bias=pair_logit_bias,
            batch_size=[],
        )

    # ------------------------------------------------------------------
    # Score arm.
    # ------------------------------------------------------------------
    def _process_score(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        positions: Sequence[ComponentReference] | None,
    ) -> NaMpnnInput:
        r"""
        Lower an explicit observed sequence for autoregressive scoring.
        """

        sequence_tokens = [
            (
                self._component_token(row.component_id, row.polymer_type_index)
                if row.backbone_complete
                else NA_MPNN_TOKEN_BY_COMPONENT[
                    {"protein": "UNK", "dna": "DX", "rna": "RX"}[_POLYMER_TYPE_NAME[row.polymer_type_index]]
                ]
            )
            for row in rows
        ]
        selected = (
            None if positions is None else {structure.system.component_index(reference) for reference in positions}
        )
        if selected is not None:
            components = {row.component_index for row in rows}
            missing = selected - components
            if missing:
                references = [structure.system.component_reference(component) for component in missing]
                raise ValueError(f"NA-MPNN positions are not polymer residues: {references!r}")
        lowered_design_positions = [
            bool(row.backbone_complete and (selected is None or row.component_index in selected)) for row in rows
        ]
        token_group_ids = list(range(len(rows)))
        return self._base_input(
            structure,
            rows,
            design_positions=lowered_design_positions,
            token_group_ids=token_group_ids,
            sequence_tokens=sequence_tokens,
            tied_weights=torch.ones(1, len(rows), device=self.device, dtype=self.dtype),
            residue_logit_bias=torch.zeros(
                1,
                len(rows),
                _VOCAB_SIZE,
                device=self.device,
                dtype=self.dtype,
            ),
            allowed_token_mask=self._allowed_token_mask(rows, ()),
            pair_logit_bias=None,
        )

    def _resolve_score_target(
        self,
        structure: Structure,
        sequence: str | Mapping[str, str | Sequence[str]],
    ) -> Structure:
        r"""
        Apply an explicit scoring sequence and retain its compatible coordinates.
        """

        system = structure.system
        if system.system_entity_ptr.numel() != 2:
            raise ValueError("an NA-MPNN sequence target must contain one molecular system")
        instance_entities = system.instance_entity.detach().cpu().tolist()
        entity_types = system.entity_type.detach().cpu().tolist()
        component_ptr = system.instance_component_ptr.detach().cpu().tolist()
        component_symbols = system.component_symbol.detach().cpu().tolist()
        polymer_instances = [
            instance
            for instance, entity in enumerate(instance_entities)
            if entity_types[entity] in (ENTITY_PROTEIN, ENTITY_DNA, ENTITY_RNA, ENTITY_DNA_RNA_HYBRID)
        ]
        if isinstance(sequence, str):
            if len(polymer_instances) != 1:
                raise ValueError("a bare NA-MPNN sequence requires exactly one polymer instance")
            sequence = {system.instance_id(polymer_instances[0]): sequence}

        requested_by_instance: dict[int, tuple[ChemicalComponent, ...]] = {}
        for instance_id, value in sequence.items():
            instance = system.instance_index(instance_id)
            entity = instance_entities[instance]
            entity_type = entity_types[entity]
            if entity_type not in (
                ENTITY_PROTEIN,
                ENTITY_DNA,
                ENTITY_RNA,
                ENTITY_DNA_RNA_HYBRID,
            ):
                raise ValueError(f"NA-MPNN sequence instance {instance_id!r} is not a polymer")
            start = component_ptr[instance]
            stop = component_ptr[instance + 1]
            if isinstance(value, str):
                polymer_type = {
                    ENTITY_PROTEIN: "protein",
                    ENTITY_DNA: "dna",
                    ENTITY_RNA: "rna",
                }.get(entity_type)
                if polymer_type is None:
                    raise ValueError("a DNA/RNA hybrid requires explicit component occurrences")
                requested_components = []
                for index, symbol in enumerate(value):
                    try:
                        component_id = one_letter_to_component(
                            symbol,
                            polymer_type,
                            policy="strict",
                        )
                    except KeyError:
                        raise ValueError(
                            f"unsupported or ambiguous {polymer_type} one-letter symbol {symbol!r} at index {index}"
                        ) from None
                    requested_components.append(get_component(component_id))
                requested = tuple(requested_components)
            else:
                requested = tuple(get_component(component_id) for component_id in value)
            if len(requested) != stop - start:
                raise ValueError(
                    f"NA-MPNN sequence for instance {instance_id!r} has length "
                    f"{len(requested)}, expected {stop - start}"
                )
            requested_by_instance[instance] = requested

        changed_components = []
        replacements = []
        for instance, requested in requested_by_instance.items():
            start = component_ptr[instance]
            for component, replacement in zip(
                range(start, component_ptr[instance + 1]),
                requested,
                strict=True,
            ):
                if system.symbols[component_symbols[component]] != replacement.component_id:
                    changed_components.append(component)
                    replacements.append(replacement)
        if not changed_components:
            return structure

        realized, old_to_new, _ = replace_components(
            system,
            torch.tensor(
                changed_components,
                dtype=torch.int64,
                device=system.component_symbol.device,
            ),
            replacements,
        )
        realized = _repartition_entities_by_sequence(system, realized)

        atom_count = realized.atom_symbol.numel()
        coordinates = structure.coordinates.new_zeros(atom_count, 3)
        coordinate_mask = structure.coordinate_mask.new_zeros(atom_count)
        occupancy = structure.occupancy.new_zeros(atom_count)
        occupancy_mask = structure.occupancy_mask.new_zeros(atom_count)
        b_factor = structure.b_factor.new_zeros(atom_count)
        b_factor_mask = structure.b_factor_mask.new_zeros(atom_count)
        atom_component = torch.repeat_interleave(
            torch.arange(
                system.component_symbol.numel(),
                dtype=torch.int64,
                device=system.atom_symbol.device,
            ),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        changed_mask = torch.zeros_like(system.component_symbol, dtype=torch.bool)
        changed_mask[changed_components] = True
        consumed_symbols = {index for index, symbol in enumerate(system.symbols) if symbol in NA_MPNN_ATOM_NAMES}
        consumed_atom = torch.tensor(
            [symbol in consumed_symbols for symbol in system.atom_symbol.detach().cpu().tolist()],
            dtype=torch.bool,
            device=system.atom_symbol.device,
        )
        retained = (old_to_new >= 0) & (~changed_mask[atom_component] | consumed_atom)
        destination = old_to_new[retained]
        coordinates[destination] = structure.coordinates[retained]
        coordinate_mask[destination] = structure.coordinate_mask[retained]
        occupancy[destination] = structure.occupancy[retained]
        occupancy_mask[destination] = structure.occupancy_mask[retained]
        b_factor[destination] = structure.b_factor[retained]
        b_factor_mask[destination] = structure.b_factor_mask[retained]
        standard_component_id = structure.standard_component_id.clone()
        standard_component_id[changed_components] = -1
        return Structure(  # type: ignore[call-arg]
            system=realized,
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

    # ------------------------------------------------------------------
    # Design / probability arm.
    # ------------------------------------------------------------------
    def _process_design(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        positions: Sequence[ComponentReference] | None,
        nucleic_acid_only: bool,
        conditions: tuple[AllowedResidueIdentities | TiedResidueIdentities, ...],
        symmetry_groups: tuple[tuple[ComponentReference, ...], ...],
        tied_weights: Mapping[ComponentReference, float] | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        pair_logit_bias: Tensor | None,
        global_omit_tokens: tuple[str, ...],
    ) -> NaMpnnInput:
        r"""
        Lower conditions, token symmetry, and bias tensors for design and probabilities.
        """

        system = structure.system
        length = len(rows)
        index_by_component = {row.component_index: index for index, row in enumerate(rows)}

        def row_index(reference: ComponentReference) -> int:
            return index_by_component[system.component_index(reference)]

        sequence_tokens = [
            (
                self._component_token(row.component_id, row.polymer_type_index)
                if row.backbone_complete
                else NA_MPNN_TOKEN_BY_COMPONENT[
                    {"protein": "UNK", "dna": "DX", "rna": "RX"}[_POLYMER_TYPE_NAME[row.polymer_type_index]]
                ]
            )
            for row in rows
        ]

        bonded = self._bonded_rows(system)
        selected = None if positions is None else {system.component_index(reference) for reference in positions}
        if selected is not None:
            missing = selected - index_by_component.keys()
            if missing:
                references = [system.component_reference(component) for component in missing]
                raise ValueError(f"NA-MPNN positions are not polymer residues: {references!r}")
        lowered_design_positions = [
            bool(
                row.backbone_complete
                and row.component_index not in bonded
                and (selected is None or row.component_index in selected)
                and (not nucleic_acid_only or row.polymer_type_index != _POLYMER_TYPE_INDEX["protein"])
            )
            for row in rows
        ]

        parent = list(range(length))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(first: int, second: int) -> None:
            first, second = find(first), find(second)
            if first != second:
                parent[second] = first

        # Exact identity ties may not cross polymer types; token symmetry may.
        for group in symmetry_groups:
            members = [row_index(reference) for reference in group]
            for member in members[1:]:
                union(members[0], member)
        restrictions: list[set[str] | None] = [None] * length
        for condition in conditions:
            if isinstance(condition, TiedResidueIdentities):
                for group in condition.groups:
                    members = [row_index(reference) for reference in group]
                    if len({rows[member].polymer_type_index for member in members}) != 1:
                        raise ValueError("NA-MPNN cannot tie exact residue identities across different polymer types")
                    for member in members[1:]:
                        union(members[0], member)
            elif isinstance(condition, AllowedResidueIdentities):
                for target in condition.positions:
                    index = row_index(target)
                    allowed = set(condition.component_ids)
                    current = restrictions[index]
                    restrictions[index] = allowed if current is None else current & allowed
            else:
                raise TypeError("NA-MPNN conditions must be AllowedResidueIdentities or TiedResidueIdentities")

        groups: dict[int, list[int]] = {}
        for index in range(length):
            groups.setdefault(find(index), []).append(index)

        allowed_token_mask = self._allowed_token_mask(rows, global_omit_tokens)
        for members in groups.values():
            # A tie shares one decoding position, so a fixed member pins the whole unit to its identity.
            # Two fixed members disagreeing over that identity is unsatisfiable; fail loud rather than let
            # the sampler silently keep each member's own token.
            fixed_identities = {sequence_tokens[index] for index in members if not lowered_design_positions[index]}
            if len(fixed_identities) > 1:
                raise ValueError("NA-MPNN cannot tie residues fixed to different identities")
            # A tie shares one decoding position, so allowed identities intersect in token space:
            # token-symmetric nucleic identities (A/DA, ...) stay equivalent across a DNA/RNA tie
            # instead of being rejected as different component IDs.
            declared: list[set[int]] = []
            for index in members:
                restriction = restrictions[index]
                if restriction is not None:
                    polymer_type = rows[index].polymer_type_index
                    declared.append({self._component_token(component_id, polymer_type) for component_id in restriction})
            if declared:
                allowed_tokens = declared[0].intersection(*declared[1:])
                for index in members:
                    row = rows[index]
                    supported = {
                        component_id
                        for component_id in _GENERATED_COMPONENTS[_POLYMER_TYPE_NAME[row.polymer_type_index]]
                        if self._component_token(component_id, row.polymer_type_index) in allowed_tokens
                    }
                    if not supported:
                        if self._component_token(row.component_id, row.polymer_type_index) in allowed_tokens:
                            lowered_design_positions[index] = False
                            continue
                        raise ValueError("NA-MPNN allowed identities contain no residue it can generate")
                    mask = [False] * _VOCAB_SIZE
                    for component_id in supported:
                        mask[self._component_token(component_id, row.polymer_type_index)] = True
                    allowed_token_mask[0, index] = torch.tensor(mask, device=self.device, dtype=torch.bool)

            if fixed_identities and any(lowered_design_positions[index] for index in members):
                anchor = next(iter(fixed_identities))
                if not allowed_token_mask[0, members, anchor].all():
                    raise ValueError("NA-MPNN tied fixed identity is outside the common token support")
                anchor_support = torch.zeros(_VOCAB_SIZE, dtype=torch.bool, device=self.device)
                anchor_support[anchor] = True
                allowed_token_mask[0, members] &= anchor_support

            if (
                any(lowered_design_positions[index] for index in members)
                and not allowed_token_mask[0, members].all(dim=0).any()
            ):
                raise ValueError("NA-MPNN tied residues have no common token support")

        token_group_ids = self._dense_ids([find(index) for index in range(length)])

        tied_weight_tensor = self._tied_weights_tensor(
            rows,
            system,
            index_by_component,
            groups,
            tied_weights,
        )
        residue_bias_tensor = self._residue_bias_tensor(
            system,
            index_by_component,
            residue_logit_bias,
        )
        pair_logit_bias_tensor = self._pair_logit_bias(rows, pair_logit_bias)

        return self._base_input(
            structure,
            rows,
            design_positions=lowered_design_positions,
            token_group_ids=token_group_ids,
            sequence_tokens=sequence_tokens,
            tied_weights=tied_weight_tensor,
            residue_logit_bias=residue_bias_tensor,
            allowed_token_mask=allowed_token_mask,
            pair_logit_bias=pair_logit_bias_tensor,
        )

    @staticmethod
    def _bonded_rows(system: MolecularSystem) -> set[int]:
        r"""
        Return polymer occurrences participating in noncanonical external bonds.
        """

        external = (system.bond_role != BOND_ROLE_COMPONENT) & (system.bond_role != BOND_ROLE_POLYMER)
        atoms = system.edge_index[:, external].reshape(-1)
        if not atoms.numel():
            return set()
        atom_component = torch.repeat_interleave(
            torch.arange(
                system.component_symbol.numel(),
                dtype=torch.int64,
                device=system.atom_symbol.device,
            ),
            system.component_atom_ptr[1:] - system.component_atom_ptr[:-1],
        )
        return set(atom_component[atoms].detach().cpu().tolist())

    def _allowed_token_mask(self, rows: list[_Row], global_omit_tokens: tuple[str, ...]) -> Tensor:
        omit_tokens: set[int] = set()
        for symbol in global_omit_tokens:
            try:
                omit_tokens.add(_TOKEN_BY_ONE_LETTER[symbol])
            except KeyError:
                raise ValueError(f"unknown NA-MPNN omit token {symbol!r}") from None
        mask = torch.zeros(1, len(rows), _VOCAB_SIZE, dtype=torch.bool, device=self.device)
        for index, row in enumerate(rows):
            for generated_token in self._generatable_tokens(row.polymer_type_index):
                if generated_token not in omit_tokens:
                    mask[0, index, generated_token] = True
        return mask

    def _dense_ids(self, roots: list[int]) -> list[int]:
        dense: dict[int, int] = {}
        result: list[int] = []
        for root in roots:
            if root not in dense:
                dense[root] = len(dense)
            result.append(dense[root])
        return result

    def _tied_weights_tensor(
        self,
        rows: list[_Row],
        system: MolecularSystem,
        index_by_component: dict[int, int],
        groups: dict[int, list[int]],
        tied_weights: Mapping[ComponentReference, float] | None,
    ) -> Tensor:
        grouped = {index for members in groups.values() if len(members) > 1 for index in members}
        values = torch.ones(1, len(rows), device=self.device, dtype=self.dtype)
        if tied_weights is None:
            return values
        for reference, weight in tied_weights.items():
            index = index_by_component[system.component_index(reference)]
            if index not in grouped:
                raise ValueError("NA-MPNN tied weights require a tied or symmetric residue")
            values[0, index] = weight
        return values

    def _residue_bias_tensor(
        self,
        system: MolecularSystem,
        index_by_component: dict[int, int],
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
    ) -> Tensor:
        values = torch.zeros(
            1,
            len(index_by_component),
            _VOCAB_SIZE,
            device=self.device,
            dtype=self.dtype,
        )
        if not residue_logit_bias:
            return values
        for reference, value in residue_logit_bias.items():
            converted = value.to(device=self.device, dtype=self.dtype)
            if converted.shape != (_VOCAB_SIZE,):
                raise ValueError("NA-MPNN residue logit bias must have shape (33,)")
            values[
                0,
                index_by_component[system.component_index(reference)],
            ] = converted
        return values

    def _pair_logit_bias(self, rows: list[_Row], pair_logit_bias: Tensor | None) -> Tensor | None:
        r"""
        Build the directed adjacent-token pair bias from a [33, 33] matrix.
        """

        if pair_logit_bias is None:
            return None
        matrix = pair_logit_bias.to(device=self.device, dtype=self.dtype)
        if matrix.shape != (_VOCAB_SIZE, _VOCAB_SIZE):
            raise ValueError("NA-MPNN pair bias must have shape (33, 33)")
        length = len(rows)
        residue_index = torch.tensor([row.residue_index for row in rows], device=self.device)
        chain_index = torch.tensor([row.chain_index for row in rows], device=self.device)
        same_chain = (chain_index[:, None] == chain_index[None, :]).to(self.dtype)
        upper = (residue_index[1:] - residue_index[:-1] == 1).to(self.dtype)
        lower = (residue_index[:-1] - residue_index[1:] == -1).to(self.dtype)
        upper_mask = torch.diag(upper, 1) * same_chain
        lower_mask = torch.diag(lower, -1) * same_chain
        pair = (
            upper_mask[:, None, :, None] * matrix[None, :, None, :]
            + lower_mask[:, None, :, None] * matrix.transpose(0, 1)[None, :, None, :]
        )
        return pair.reshape(1, length, _VOCAB_SIZE, length, _VOCAB_SIZE)
