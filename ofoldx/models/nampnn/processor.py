r"""Canonical mixed-polymer Structure lowering for NA-MPNN."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from typing_extensions import Self

from ...data.conditions import AllowedResidueIdentities, TiedResidueIdentities
from ...data.molecular import AtomSite, ComponentReference, MolecularSystem, PolymerSequence
from ...data.structure import Structure
from ...io.structure import read_structure
from .config import NaMpnnProcessorConfig
from .model import (
    DNA_GENERATABLE_TOKEN_IDS,
    MIXED_POLYMER_ATOM_NAMES,
    MIXED_POLYMER_TOKEN_BY_COMPONENT,
    PROTEIN_GENERATABLE_TOKEN_IDS,
    RNA_SEPARATE_GENERATABLE_TOKEN_IDS,
    RNA_SHARED_GENERATABLE_TOKEN_IDS,
    NaMpnnInput,
)

_VOCAB_SIZE = 33
_REQUIRED_ATOMS = {
    "protein": ("N", "CA", "C", "O"),
    "dna": ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'"),
    "rna": ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"),
}
_GENERATED_COMPONENTS = {
    "protein": tuple(MIXED_POLYMER_TOKEN_BY_COMPONENT)[:20],
    "dna": ("DA", "DC", "DG", "DT"),
    "rna": ("A", "C", "G", "U"),
}
_DNA_COMPONENTS = frozenset({"DA", "DC", "DG", "DT", "DX"})
_RNA_COMPONENTS = frozenset({"A", "C", "G", "U", "RX"})
_SHARED_NUCLEIC_TOKENS = {
    "A": MIXED_POLYMER_TOKEN_BY_COMPONENT["DA"],
    "C": MIXED_POLYMER_TOKEN_BY_COMPONENT["DC"],
    "G": MIXED_POLYMER_TOKEN_BY_COMPONENT["DG"],
    "U": MIXED_POLYMER_TOKEN_BY_COMPONENT["DT"],
}
_POLYMER_TYPE_ID = {"protein": 0, "dna": 1, "rna": 2}

# One aligned lowered residue row.
_Row = tuple[ComponentReference, str, str, int, int, np.ndarray, tuple[bool, ...], bool]


class NaMpnnProcessor:
    r"""Lower one protein/DNA/RNA Structure into the released 16-atom layout.

    The Processor has two arms selected by the expression: a ``sequence`` override
    lowers the score arm over an explicit observed sequence, while the default
    design/probability arm lowers conditions, token symmetry, and the aligned
    bias tensors. Both arms derive residue indices from ``Structure.residue_numbers``
    with the released running-count renumber, type polymers from the canonical
    component, and demote incomplete or unoccupied residues to an invalid unknown
    token.
    """

    def __init__(
        self,
        config: NaMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = NaMpnnProcessorConfig() if config is None else config
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
        sequence: str | Mapping[str, str | Sequence[str]] | None = None,
        design_positions: Sequence[ComponentReference] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentReference]] = (),
        tied_weights: Mapping[ComponentReference, float] | None = None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None = None,
        pair_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
    ) -> NaMpnnInput:
        r"""Select the score arm on a sequence override, otherwise the design arm."""

        structure = self._read(input)
        rows = self._lower_rows(structure)
        if sequence is not None:
            return self._process_score(structure, rows, sequence=sequence, design_positions=design_positions)
        return self._process_design(
            structure,
            rows,
            design_positions=design_positions,
            conditions=tuple(conditions),
            symmetry_groups=tuple(tuple(group) for group in symmetry_groups),
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_bias=pair_bias,
            global_omit_tokens=tuple(global_omit_tokens),
        )

    # ------------------------------------------------------------------
    # Shared lowering.
    # ------------------------------------------------------------------
    def _read(self, input: Structure | str | PathLike[str]) -> Structure:
        if isinstance(input, (str, PathLike)):
            input = read_structure(Path(input).expanduser())
        if not isinstance(input, Structure):
            raise TypeError("NaMpnnProcessor input must be a Structure, string, or PathLike")
        return input

    def _polymer_type(self, entity: PolymerSequence, component_id: str) -> str:
        if entity.polymer_type in ("protein", "dna", "rna"):
            return entity.polymer_type
        if component_id in _DNA_COMPONENTS:
            return "dna"
        if component_id in _RNA_COMPONENTS:
            return "rna"
        raise ValueError(f"NA-MPNN cannot consume polymer type {entity.polymer_type!r}")

    def _lower_rows(self, structure: Structure) -> list[_Row]:
        r"""Lower every polymer residue to aligned coordinates, masks, and identity."""

        system = structure.system
        site_index = {site: index for index, site in enumerate(structure.sites)}
        occupancy = structure.occupancy
        residue_numbers = structure.residue_numbers

        raw: list[tuple[ComponentReference, str, str, int]] = []
        coordinates: list[np.ndarray] = []
        atom_masks: list[tuple[bool, ...]] = []
        valids: list[bool] = []
        source_numbers: list[int] = []
        chain_index = 0
        for instance_id, entity_id in system.instances.items():
            entity = system.entities[entity_id]
            if not isinstance(entity, PolymerSequence):
                continue
            for component_index, component in enumerate(entity.components):
                reference = (instance_id, component_index)
                polymer_type = self._polymer_type(entity, component.component_id)
                row_coordinates: list[np.ndarray] = []
                row_mask: list[bool] = []
                for atom_name in MIXED_POLYMER_ATOM_NAMES:
                    index = site_index.get(AtomSite(instance_id, component_index, atom_name))
                    present = (
                        index is not None
                        and bool(structure.coordinate_mask[index])
                        and (occupancy is None or bool(np.isnan(occupancy[index])) or float(occupancy[index]) > 0.0)
                    )
                    row_mask.append(present)
                    row_coordinates.append(structure.coordinates[index] if present else np.zeros(3, dtype=np.float32))
                valid = all(row_mask[MIXED_POLYMER_ATOM_NAMES.index(name)] for name in _REQUIRED_ATOMS[polymer_type])
                raw.append((reference, component.component_id, polymer_type, chain_index))
                coordinates.append(np.stack(row_coordinates))
                atom_masks.append(tuple(row_mask))
                valids.append(valid)
                if residue_numbers is not None and reference in residue_numbers:
                    source_numbers.append(residue_numbers[reference].number)
                else:
                    source_numbers.append(component_index)
            chain_index += 1

        residue_index = self._renumber(source_numbers)
        rows: list[_Row] = []
        for position, (reference, component_id, polymer_type, chain) in enumerate(raw):
            rows.append(
                (
                    reference,
                    component_id,
                    polymer_type,
                    chain,
                    residue_index[position],
                    coordinates[position],
                    atom_masks[position],
                    valids[position],
                )
            )
        if not rows:
            raise ValueError("NA-MPNN requires at least one protein, DNA, or RNA residue")
        if not any(row[7] for row in rows):
            raise ValueError("NA-MPNN requires at least one residue with a complete polymer backbone")
        return rows

    @staticmethod
    def _renumber(source_numbers: Sequence[int]) -> list[int]:
        r"""Apply the released running-count renumber for repeats and insertion codes."""

        renumbered: list[int] = []
        count = 0
        previous: int | None = None
        for source in source_numbers:
            if source == previous:
                count += 1
            renumbered.append(source + count)
            previous = source
        return renumbered

    def _component_token(self, component_id: str, polymer_type: str) -> int:
        if polymer_type == "rna" and self.config.shared_nucleic_tokens:
            return _SHARED_NUCLEIC_TOKENS.get(component_id, MIXED_POLYMER_TOKEN_BY_COMPONENT["DX"])
        unknown = {"protein": "UNK", "dna": "DX", "rna": "RX"}[polymer_type]
        return MIXED_POLYMER_TOKEN_BY_COMPONENT.get(component_id, MIXED_POLYMER_TOKEN_BY_COMPONENT[unknown])

    def _generatable_tokens(self, polymer_type: str) -> tuple[int, ...]:
        if polymer_type == "protein":
            return PROTEIN_GENERATABLE_TOKEN_IDS
        if polymer_type == "dna":
            return DNA_GENERATABLE_TOKEN_IDS
        return (
            RNA_SHARED_GENERATABLE_TOKEN_IDS
            if self.config.shared_nucleic_tokens
            else RNA_SEPARATE_GENERATABLE_TOKEN_IDS
        )

    def _tensor(self, values: list, dtype: torch.dtype) -> Tensor:
        return torch.tensor(values, device=self.device, dtype=dtype).unsqueeze(0)

    def _base_input(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        design_positions: list[bool],
        tied_group_ids: list[int],
        sequence_tokens: list[int],
        tied_weights: Tensor | None,
        residue_logit_bias: Tensor | None,
        allowed_token_mask: Tensor | None,
        pair_logit_bias: Tensor | None,
        conditions: tuple[AllowedResidueIdentities | TiedResidueIdentities, ...],
    ) -> NaMpnnInput:
        coordinates = np.stack([row[5] for row in rows])
        return NaMpnnInput(
            atom_coordinates=torch.as_tensor(coordinates, device=self.device, dtype=self.dtype).unsqueeze(0),
            atom_mask=self._tensor([list(row[6]) for row in rows], torch.bool),
            sequence_tokens=self._tensor(sequence_tokens, torch.int64),
            residue_mask=self._tensor([row[7] for row in rows], torch.bool),
            residue_index=self._tensor([row[4] for row in rows], torch.int64),
            chain_index=self._tensor([row[3] for row in rows], torch.int64),
            polymer_type_ids=self._tensor([_POLYMER_TYPE_ID[row[2]] if row[7] else 3 for row in rows], torch.int64),
            protein_mask=self._tensor([row[2] == "protein" and row[7] for row in rows], torch.bool),
            dna_mask=self._tensor([row[2] == "dna" and row[7] for row in rows], torch.bool),
            rna_mask=self._tensor([row[2] == "rna" and row[7] for row in rows], torch.bool),
            design_positions=self._tensor(design_positions, torch.bool),
            tied_group_ids=self._tensor(tied_group_ids, torch.int64),
            structures=(structure,),
            component_references=(tuple(row[0] for row in rows),),
            polymer_types=(tuple(row[2] for row in rows),),
            conditions=(conditions,),
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            allowed_token_mask=allowed_token_mask,
            pair_logit_bias=pair_logit_bias,
        )

    # ------------------------------------------------------------------
    # Score arm.
    # ------------------------------------------------------------------
    def _process_score(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        sequence: str | Mapping[str, str | Sequence[str]],
        design_positions: Sequence[ComponentReference] | None,
    ) -> NaMpnnInput:
        r"""Lower an explicit observed sequence for autoregressive scoring."""

        overrides = self._resolve_sequence_override(structure.system, rows, sequence)
        sequence_tokens = [
            (
                self._component_token(overrides.get(row[0], row[1]), row[2])
                if row[7]
                else MIXED_POLYMER_TOKEN_BY_COMPONENT[{"protein": "UNK", "dna": "DX", "rna": "RX"}[row[2]]]
            )
            for row in rows
        ]
        selected = None if design_positions is None else set(design_positions)
        if selected is not None:
            references = {row[0] for row in rows}
            missing = selected - references
            if missing:
                raise ValueError(f"NA-MPNN positions are not polymer residues: {sorted(missing)!r}")
        lowered_design_positions = [bool(row[7] and (selected is None or row[0] in selected)) for row in rows]
        tied_group_ids = list(range(len(rows)))
        return self._base_input(
            structure,
            rows,
            design_positions=lowered_design_positions,
            tied_group_ids=tied_group_ids,
            sequence_tokens=sequence_tokens,
            tied_weights=None,
            residue_logit_bias=None,
            allowed_token_mask=None,
            pair_logit_bias=None,
            conditions=(),
        )

    def _resolve_sequence_override(
        self,
        system: MolecularSystem,
        rows: list[_Row],
        sequence: str | Mapping[str, str | Sequence[str]],
    ) -> dict[ComponentReference, str]:
        polymer_instances = [
            instance_id
            for instance_id, entity_id in system.instances.items()
            if isinstance(system.entities[entity_id], PolymerSequence)
        ]
        if isinstance(sequence, str):
            if len(polymer_instances) != 1:
                raise ValueError("a bare NA-MPNN sequence requires exactly one polymer instance")
            sequence = {polymer_instances[0]: sequence}
        length_by_instance: dict[str, int] = {}
        for reference, *_ in rows:
            length_by_instance[reference[0]] = length_by_instance.get(reference[0], 0) + 1
        overrides: dict[ComponentReference, str] = {}
        for instance_id, value in sequence.items():
            entity = system.entities[system.instances[instance_id]]
            if not isinstance(entity, PolymerSequence):
                raise ValueError(f"NA-MPNN sequence instance {instance_id!r} is not a polymer")
            if isinstance(value, str):
                component_ids = tuple(
                    component.component_id
                    for component in PolymerSequence.from_one_letter(value, polymer_type=entity.polymer_type).components
                )
            else:
                component_ids = tuple(value)
            if len(component_ids) != len(entity):
                raise ValueError(
                    f"NA-MPNN sequence for instance {instance_id!r} has length "
                    f"{len(component_ids)}, expected {len(entity)}"
                )
            for component_index, component_id in enumerate(component_ids):
                overrides[(instance_id, component_index)] = component_id
        return overrides

    # ------------------------------------------------------------------
    # Design / probability arm.
    # ------------------------------------------------------------------
    def _process_design(
        self,
        structure: Structure,
        rows: list[_Row],
        *,
        design_positions: Sequence[ComponentReference] | None,
        conditions: tuple[AllowedResidueIdentities | TiedResidueIdentities, ...],
        symmetry_groups: tuple[tuple[ComponentReference, ...], ...],
        tied_weights: Mapping[ComponentReference, float] | None,
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
        pair_bias: Tensor | None,
        global_omit_tokens: tuple[str, ...],
    ) -> NaMpnnInput:
        r"""Lower conditions, token symmetry, and bias tensors for design and probabilities."""

        system = structure.system
        length = len(rows)
        index_by_reference = {row[0]: index for index, row in enumerate(rows)}
        sequence_tokens = [
            (
                self._component_token(row[1], row[2])
                if row[7]
                else MIXED_POLYMER_TOKEN_BY_COMPONENT[{"protein": "UNK", "dna": "DX", "rna": "RX"}[row[2]]]
            )
            for row in rows
        ]

        bonded = self._bonded_rows(system)
        selected = None if design_positions is None else set(design_positions)
        if selected is not None:
            for reference in selected:
                if reference not in index_by_reference:
                    raise ValueError(f"NA-MPNN position {reference!r} is not a polymer residue")
        lowered_design_positions = [
            bool(row[7] and row[0] not in bonded and (selected is None or row[0] in selected)) for row in rows
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

        # Same-entity shared instances tie identical monomers across copies.
        entity_monomer_rows: dict[str, list[list[int]]] = {}
        for index, row in enumerate(rows):
            entity_id = system.instances[row[0][0]]
            entity = system.entities[entity_id]
            entity_monomer_rows.setdefault(entity_id, [[] for _ in entity.components])[row[0][1]].append(index)
        for monomer_rows in entity_monomer_rows.values():
            for members in monomer_rows:
                for member in members[1:]:
                    union(members[0], member)

        # Exact identity ties may not cross polymer types; token symmetry may.
        for group in symmetry_groups:
            members = [index_by_reference[reference] for reference in group]
            for member in members[1:]:
                union(members[0], member)
        restrictions: list[set[str] | None] = [None] * length
        for condition in conditions:
            if isinstance(condition, TiedResidueIdentities):
                for group in condition.groups:
                    members = [index_by_reference[reference] for reference in group]
                    for member in members[1:]:
                        union(members[0], member)
            elif isinstance(condition, AllowedResidueIdentities):
                for target in condition.positions:
                    index = index_by_reference[target]
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
            polymer_types = {rows[index][2] for index in members}
            exact_tie = any(
                isinstance(condition, TiedResidueIdentities)
                and any(index_by_reference[reference] in members for group in condition.groups for reference in group)
                for condition in conditions
            )
            if exact_tie and len(polymer_types) != 1:
                raise ValueError("NA-MPNN cannot tie exact residue identities across different polymer types")
            declared: list[set[str]] = []
            for index in members:
                restriction = restrictions[index]
                if restriction is not None:
                    declared.append(restriction)
            if declared:
                allowed = set.intersection(*declared)
                for index in members:
                    row = rows[index]
                    supported = allowed & set(_GENERATED_COMPONENTS[row[2]])
                    if not supported:
                        if row[1] in allowed:
                            lowered_design_positions[index] = False
                            continue
                        raise ValueError("NA-MPNN allowed identities contain no residue it can generate")
                    mask = [False] * _VOCAB_SIZE
                    for component_id in supported:
                        mask[self._component_token(component_id, row[2])] = True
                    allowed_token_mask[0, index] = torch.tensor(mask, device=self.device, dtype=torch.bool)

        tied_group_ids = self._dense_ids([find(index) for index in range(length)])

        tied_weight_tensor = self._tied_weights_tensor(rows, index_by_reference, groups, tied_weights)
        residue_bias_tensor = self._residue_bias_tensor(index_by_reference, residue_logit_bias)
        pair_logit_bias = self._pair_logit_bias(rows, pair_bias)

        return self._base_input(
            structure,
            rows,
            design_positions=lowered_design_positions,
            tied_group_ids=tied_group_ids,
            sequence_tokens=sequence_tokens,
            tied_weights=tied_weight_tensor,
            residue_logit_bias=residue_bias_tensor,
            allowed_token_mask=allowed_token_mask,
            pair_logit_bias=pair_logit_bias,
            conditions=conditions,
        )

    def _bonded_rows(self, system: MolecularSystem) -> set[ComponentReference]:
        bonded: set[ComponentReference] = set()
        for bond in system.bonds:
            for site in (bond.site_a, bond.site_b):
                entity = system.entities[system.instances[site.instance_id]]
                if isinstance(entity, PolymerSequence):
                    bonded.add((site.instance_id, site.component_index))
        return bonded

    def _allowed_token_mask(self, rows: list[_Row], global_omit_tokens: tuple[str, ...]) -> Tensor:
        omit_tokens: set[int] = set()
        for token in global_omit_tokens:
            if token in MIXED_POLYMER_TOKEN_BY_COMPONENT:
                omit_tokens.add(MIXED_POLYMER_TOKEN_BY_COMPONENT[token])
            else:
                raise ValueError(f"unknown NA-MPNN omit token {token!r}")
        mask = torch.zeros(1, len(rows), _VOCAB_SIZE, dtype=torch.bool, device=self.device)
        for index, row in enumerate(rows):
            for generated_token in self._generatable_tokens(row[2]):
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
        index_by_reference: dict[ComponentReference, int],
        groups: dict[int, list[int]],
        tied_weights: Mapping[ComponentReference, float] | None,
    ) -> Tensor | None:
        if tied_weights is None:
            return None
        grouped = {index for members in groups.values() if len(members) > 1 for index in members}
        values = torch.ones(1, len(rows), device=self.device, dtype=self.dtype)
        for reference, weight in tied_weights.items():
            index = index_by_reference[reference]
            if index not in grouped:
                raise ValueError("NA-MPNN tied weights require a tied or symmetric residue")
            values[0, index] = float(weight)
        return values

    def _residue_bias_tensor(
        self,
        index_by_reference: dict[ComponentReference, int],
        residue_logit_bias: Mapping[ComponentReference, Tensor] | None,
    ) -> Tensor | None:
        if not residue_logit_bias:
            return None
        values = torch.zeros(1, len(index_by_reference), _VOCAB_SIZE, device=self.device, dtype=self.dtype)
        for reference, value in residue_logit_bias.items():
            converted = value.to(device=self.device, dtype=self.dtype)
            if converted.shape != (_VOCAB_SIZE,):
                raise ValueError("NA-MPNN residue logit bias must have shape (33,)")
            values[0, index_by_reference[reference]] = converted
        return values

    def _pair_logit_bias(self, rows: list[_Row], pair_bias: Tensor | None) -> Tensor | None:
        r"""Build the directed adjacent-token pair bias from a [33, 33] matrix."""

        if pair_bias is None:
            return None
        matrix = pair_bias.to(device=self.device, dtype=self.dtype)
        if matrix.shape != (_VOCAB_SIZE, _VOCAB_SIZE):
            raise ValueError("NA-MPNN pair bias must have shape (33, 33)")
        length = len(rows)
        residue_index = torch.tensor([row[4] for row in rows], device=self.device)
        chain_index = torch.tensor([row[3] for row in rows], device=self.device)
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
