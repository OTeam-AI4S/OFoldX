# OFoldX
# Copyright (C) 2026-Present  Valhalla Technology

# This file is part of OFoldX.

# OFoldX is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

# OFoldX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# For additional terms and clarifications, please refer to our License FAQ at:
# <https://ofoldx.valtech.ai/about/license-faq/>.

r"""
ProteinMPNN input processing, written against the shared preparation contract.

The layout addresses N, CA, C and O (or CA alone) of every protein component holding a CA (a cap such as ACE is no
row); the model's rows are those residue tokens, designed instances first (upstream chain order), and its tokens are
the canonical classes (MSE reads ``M`` with its ``encoding_loss`` record). Sequence design regenerates no coordinate:
the scope is empty and the design producer builds the final target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from chanfig import Config
from numpy.typing import NDArray
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ComponentRef, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, encode_sequence
from ...errors import RequestRef, fail
from ...model import Model
from ...processor import Processor
from ...provenance import Records, known
from ..conditions import AllowedResidueIdentities, TiedResidueIdentities
from ..design import PositionSelection, resolve_positions
from ..layout import LayoutTable, model_tokens, slot_map, token_layout
from ..preparation import (
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
)
from ..processing import project_model_coordinates
from ..targets import SequenceTarget, prepare_sequence_target
from .config import ProteinMpnnConfig, ProteinMpnnProcessorConfig
from .model import PROTEINMPNN_VOCABULARY, ProteinMpnnInput

_PROTEIN = MonomerType.PROTEIN


def _table(names: tuple[str, ...]) -> LayoutTable:
    return LayoutTable(
        family="proteinmpnn",
        monomer_types={_PROTEIN: 0},
        residue_slots=None,
        monomer_slots={_PROTEIN: names},
        hydrogens="exclude",
        per_atom_components="none",
        exclude_designated_leaving=False,
        centre_atom={_PROTEIN: "CA"},
        frame_atoms={},
        representative={},
        representative_default={},
    )


PROTEINMPNN_LAYOUT = _table(("N", "CA", "C", "O"))
PROTEINMPNN_CA_LAYOUT = _table(("CA",))
#: ProteinMPNN reads coordinates and classes of N, CA, C, O only; an excess valence elsewhere is not its input.
PROTEINMPNN_GATING = GatingPolicy(refuse_valence_excess=False)

_GENERATABLE = frozenset(TOKEN_INDEX[token] for token in PROTEINMPNN_VOCABULARY)
_AMINO = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
_CLASS_OF_ID = {
    **{name: TOKEN_INDEX[letter] for letter, name in zip("ARNDCQEGHILKMFPSTWYV", _AMINO)},
    "UNK": TOKEN_INDEX["X"],
}


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _reorder(layout: TokenLayout, order: NDArray[np.int64]) -> TokenLayout:
    r"""The same tokens in the given order (atoms follow their tokens)."""
    counts = np.diff(layout.token_atom_ptr)[order]
    pointer = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    atoms = np.concatenate(
        [layout.token_atoms[layout.token_atom_ptr[t] : layout.token_atom_ptr[t + 1]] for t in order.tolist()]
        or [np.zeros(0, dtype=np.int64)]
    )
    new = np.empty(len(order), dtype=np.int64)
    new[order] = np.arange(len(order))
    atom_token = np.append(new, -1)[layout.atom_token]  # an atom of no token (-1) reads the appended -1
    return TokenLayout(
        token_component=layout.token_component[order],
        token_centre_atom=layout.token_centre_atom[order],
        token_class=layout.token_class[order],
        atom_token=atom_token,
        token_atom_ptr=pointer,
        token_atoms=atoms.astype(np.int64),
    )


class ProteinMpnnProcessor(Processor):
    r"""
    Prepare protein backbones and sequence constraints for ProteinMPNN.

    Args:
        config: ProteinMPNN input options. Defaults to ``ProteinMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class: ClassVar[type[Config] | None] = ProteinMpnnProcessorConfig
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

    @property
    def layout(self) -> LayoutTable:
        r"""The layout of this processor's backbone representation."""
        return PROTEINMPNN_CA_LAYOUT if self.config.ca_only else PROTEINMPNN_LAYOUT

    def validate_model_input_layout(self, model: Model) -> None:
        ca_only = cast(ProteinMpnnConfig, model.config).ca_only
        if self.config.ca_only != ca_only:
            fail("value_invalid", RequestRef("ca_only", str(ca_only)), processor=self.config.ca_only)

    def prepare_input(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        sequence: str | Mapping[str, str] | None = None,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pssm_coefficients: Mapping[ComponentRef, float] | None = None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None = None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None = None,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[ProteinMpnnInput]:
        r"""
        Prepare a structure for ProteinMPNN.

        Args:
            structure: The target; every protein component holding a CA is a row, valid when its backbone atoms are
                observed.
            design: The positions to design among the valid protein components (default every one).
            sequence: Optional one-letter sequence scored instead of the target's classes (a bare string needs
                exactly one protein instance); letters outside ProteinMPNN's vocabulary are refused.
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            pssm_coefficients: Per-residue PSSM mixing coefficients; each key requires ``pssm_probabilities``.
            pssm_probabilities: Per-residue probability vectors over the 72 canonical classes.
            pssm_log_odds: Per-residue log-odds vectors over the 72 canonical classes.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            cropper: Optional training crop over the residue tokens; needs ``generator``.
            generator: The numpy generator of the crop.

        Notes:
            Rows order designed protein instances before fixed ones (each set by instance symbol) and keep component
            order within an instance; ``residue_index`` is ``100 * chain_index + row``.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        table = self.layout
        backbone_slots = cast(Mapping[MonomerType, tuple[str | None, ...]], table.monomer_slots)
        width = len(backbone_slots[_PROTEIN])
        observed = structure.coordinate_mask.copy()
        if filter_by_occupancy:
            observed &= ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
        layout = token_layout(system, table, atom_mask=_residue_atoms(system))
        valid = _valid_rows(slot_map(system, layout, table), observed, width)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component[valid]] = True
        positions = resolve_positions(structure, design, allowed=allowed)
        if cropper is not None:
            named = _named_components(system, conditions, tied_weights, residue_logit_bias, pssm_probabilities)
            required = np.isin(layout.token_component, named)
            layout = crop_token_layout(
                structure, layout, cropper, required=required, generator=cast(np.random.Generator, generator)
            )
        layout = _reorder(layout, _row_order(system, layout, positions))
        slots = slot_map(system, layout, table)
        accepted = gate(structure, slots, PROTEINMPNN_GATING)
        valid = _valid_rows(slots, observed, width)
        tokens, losses = model_tokens(system, layout)
        if sequence is not None:
            tokens = _sequence_tokens(system, layout, tokens, sequence)
        features = self._features(
            structure,
            slots,
            layout,
            tokens,
            valid,
            np.isin(layout.token_component, positions) & valid,
            conditions=conditions,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pssm_coefficients=pssm_coefficients,
            pssm_probabilities=pssm_probabilities,
            pssm_log_odds=pssm_log_odds,
        )
        records = Records.concat((component_records("encoding_loss", losses.component), accepted))
        record = PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))
        return PreparedInput(
            features,
            PreparationContext(structure, slots, RegenerationScope.none(system), record, spec=self.preparation_spec),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[ProteinMpnnInput]) -> SequenceTarget:
        r"""Align protein sequence truth to the prepared residue rows.

        Missing and unknown identities remain unavailable. Known protein classes
        retain their canonical IDs even when a particular checkpoint cannot score
        them; the objective checks its supported output domain.
        """
        self._validate_prepared(prepared)
        result = prepare_sequence_target(prepared.context.structure, target, prepared.features.component_index)
        kind = torch.as_tensor(CLASS_MONOMER, device=result.tokens.device)[result.tokens]
        if bool((result.mask & (kind != MonomerType.PROTEIN)).any()):
            fail("model_capacity", RequestRef("target", "protein sequence supervision is required"))
        return result

    def _features(
        self,
        structure: Structure,
        slots: SlotMap,
        layout: TokenLayout,
        tokens: NDArray[np.int64],
        valid: NDArray[np.bool_],
        designed: NDArray[np.bool_],
        *,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        tied_weights: Mapping[ComponentRef, float] | None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None,
        pssm_coefficients: Mapping[ComponentRef, float] | None,
        pssm_probabilities: Mapping[ComponentRef, Tensor] | None,
        pssm_log_odds: Mapping[ComponentRef, Tensor] | None,
    ) -> ProteinMpnnInput:
        r"""The ProteinMPNN tensors: backbone rows, tie closure, identity support, biases and PSSM terms."""
        system = structure.system
        rows = len(layout.token_component)
        vocab = len(TOKENS)
        device, dtype = self.device, self.dtype
        row_of = np.full(len(system.component_symbol), -1, dtype=np.int64)
        row_of[layout.token_component] = np.arange(rows)

        def row(reference: ComponentRef) -> int:
            component = int(system.component_index([reference])[0])
            found = int(row_of[component]) if component >= 0 else -1
            if found < 0:
                fail("request_invalid_positions", RequestRef("reference", str(reference)))
            return found

        parent = list(range(rows))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        restrictions: dict[int, set[int]] = {}
        tied_rows: set[int] = set()
        for condition in conditions:
            if isinstance(condition, TiedResidueIdentities):
                for group in condition.groups:
                    members = tuple(dict.fromkeys(row(reference) for reference in group))
                    if len(members) < 2:
                        continue
                    tied_rows.update(members)
                    root = find(members[0])
                    for member in members[1:]:
                        other = find(member)
                        if other != root:
                            parent[other] = root
            else:
                classes = {_CLASS_OF_ID[name] for name in condition.component_ids if name in _CLASS_OF_ID}
                for reference in condition.positions:
                    at = row(reference)
                    restrictions[at] = restrictions[at] & classes if at in restrictions else set(classes)
        groups_by_root: dict[int, list[int]] = {}
        for at in sorted(tied_rows):
            groups_by_root.setdefault(find(at), []).append(at)
        tied_groups = tuple(tuple(members) for members in groups_by_root.values() if len(members) > 1)
        group_id: dict[int, int] = {}
        tied_group_ids = [group_id.setdefault(find(at), len(group_id)) for at in range(rows)]

        support_by_row: dict[int, set[int]] = {}
        grouped: set[int] = set()
        for members in tied_groups:
            grouped.update(members)
            limits = [restrictions[at] for at in members if at in restrictions]
            anchors = {int(tokens[at]) for at in members if not designed[at]}
            if len(anchors) > 1:
                fail("value_invalid", RequestRef("conditions", "a tied group with conflicting fixed identities"))
            support = set(_GENERATABLE)
            for limit in limits:
                support &= limit
            if anchors:
                support &= anchors
            if not support:
                fail("value_invalid", RequestRef("conditions", "a tied group with no generatable identity"))
            if limits or anchors:
                support_by_row.update({at: support for at in members})
        for at, limit in restrictions.items():
            if at in grouped:
                continue
            restricted_support = _GENERATABLE & limit
            if not designed[at]:
                restricted_support = restricted_support & {int(tokens[at])}
            if not restricted_support:
                fail("value_invalid", RequestRef("conditions", "allowed identities with no generatable identity"))
            support_by_row[at] = set(restricted_support)
        allowed_token_mask = None
        if support_by_row:
            allowed_token_mask = torch.ones(1, rows, vocab, dtype=torch.bool, device=device)
            for at, support in support_by_row.items():
                allowed_token_mask[0, at] = False
                allowed_token_mask[0, at, sorted(support)] = True

        tied_weight_values = None
        if tied_weights:
            tied_weight_values = torch.ones(1, rows, device=device, dtype=dtype)
            for members in tied_groups:
                tied_weight_values[0, list(members)] = 1.0 / len(members)
            for reference, weight in tied_weights.items():
                at = row(reference)
                if at not in grouped:
                    fail("value_invalid", RequestRef("tied_weights", str(reference)))
                tied_weight_values[0, at] = weight

        def vectors(values: Mapping[ComponentRef, Tensor] | None, name: str, fill: float) -> Tensor | None:
            if not values:
                return None
            result = torch.full((1, rows, vocab), fill, device=device, dtype=dtype)
            for reference, value in values.items():
                converted = value.to(device=device, dtype=dtype)
                if converted.shape != (vocab,):
                    fail("value_invalid", RequestRef(name, str(tuple(converted.shape))), expected=vocab)
                result[0, row(reference)] = converted
            return result

        residue_bias_values = vectors(residue_logit_bias, "residue_logit_bias", 0.0)
        pssm_probability_values = vectors(pssm_probabilities, "pssm_probabilities", 0.0)
        pssm_log_odds_values = vectors(pssm_log_odds, "pssm_log_odds", float("inf"))
        pssm_coefficient_values = None
        if pssm_probabilities:
            pssm_coefficient_values = torch.zeros(1, rows, device=device, dtype=dtype)
            pssm_coefficient_values[0, [row(reference) for reference in pssm_probabilities]] = 1
            for reference, coefficient in (pssm_coefficients or {}).items():
                if reference not in pssm_probabilities:
                    fail("value_invalid", RequestRef("pssm_coefficients", str(reference)))
                pssm_coefficient_values[0, row(reference)] = coefficient
        elif pssm_coefficients:
            fail("value_invalid", RequestRef("pssm_coefficients", "without pssm_probabilities"))
        for members in tied_groups:
            index = torch.tensor(members, device=device)
            for name, values in (
                ("residue_logit_bias", residue_bias_values),
                ("pssm_coefficients", pssm_coefficient_values),
                ("pssm_probabilities", pssm_probability_values),
            ):
                if values is not None and not torch.equal(
                    values[0, index], values[0, index[:1]].expand_as(values[0, index])
                ):
                    fail("value_invalid", RequestRef(name, "differs within a tied group"))

        backbone_slots = cast(Mapping[MonomerType, tuple[str | None, ...]], self.layout.monomer_slots)
        width = len(backbone_slots[_PROTEIN])
        atoms = slots.atom.reshape(rows, width)
        coordinates = np.where((atoms >= 0)[..., None], structure.coordinates[np.maximum(atoms, 0)], 0.0)
        points = torch.from_numpy(coordinates[:, 0] if width == 1 else coordinates)
        available = torch.from_numpy(valid.copy())
        if points.dim() == available.dim() + 2:
            available = available.unsqueeze(-1) & torch.ones_like(points[..., 0], dtype=torch.bool)
        points = project_model_coordinates(points, available, dtype=dtype, feature="protein backbone")
        instance = _owner(system.instance_component_ptr)[layout.token_component]
        chain_of = {value: index for index, value in enumerate(dict.fromkeys(instance.tolist()))}
        chain_index = np.array([chain_of[value] for value in instance.tolist()], dtype=np.int64)

        def tensor(values, kind=torch.int64) -> Tensor:
            return torch.as_tensor(np.asarray(values), dtype=kind, device=device).unsqueeze(0)

        return ProteinMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=points.to(device=device).unsqueeze(0),
            sequence_tokens=tensor(tokens),
            residue_mask=tensor(valid, torch.bool),
            residue_index=tensor(100 * chain_index + np.arange(rows)),
            chain_index=tensor(chain_index),
            design_mask=tensor(designed, torch.bool),
            component_index=tensor(layout.token_component),
            tied_group_ids=tensor(tied_group_ids),
            tied_weights=tied_weight_values,
            allowed_token_mask=allowed_token_mask,
            residue_logit_bias=residue_bias_values,
            pssm_coefficients=pssm_coefficient_values,
            pssm_probabilities=pssm_probability_values,
            pssm_log_odds=pssm_log_odds_values,
            batch_size=[],
        )


def _residue_atoms(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""The atoms of components holding a CA: a cap (ACE, NH2, FOR) is no row, as upstream reads residues only."""
    component = _owner(system.component_atom_ptr)
    holds = np.zeros(len(system.component_symbol), dtype=bool)
    holds[component[system.text("atom", system.atom_symbol) == "CA"]] = True
    return holds[component]


def _valid_rows(slots: SlotMap, observed: NDArray[np.bool_], width: int) -> NDArray[np.bool_]:
    r"""Per row, whether every backbone slot is bound to an observed atom."""
    atoms = slots.atom.reshape(-1, width)
    return ((atoms >= 0) & observed[np.maximum(atoms, 0)]).all(axis=1)


def _row_order(system: MolecularSystem, layout: TokenLayout, positions: NDArray[np.int64]) -> NDArray[np.int64]:
    r"""Rows of designed instances first, then the others, each by instance symbol; component order within."""
    instance = _owner(system.instance_component_ptr)[layout.token_component]
    symbol = system.text("instance", system.instance_symbol[instance])
    designed = np.isin(instance, _owner(system.instance_component_ptr)[positions])
    return np.lexsort((layout.token_component, symbol, ~designed)).astype(np.int64)


def _named_components(system: MolecularSystem, conditions, *mappings) -> NDArray[np.int64]:
    r"""The components every condition and per-residue input names (a crop must keep them)."""
    references: list[ComponentRef] = []
    for condition in conditions:
        if isinstance(condition, TiedResidueIdentities):
            references += [reference for group in condition.groups for reference in group]
        else:
            references += list(condition.positions)
    for values in mappings:
        references += list(values or ())
    found = system.component_index(references)
    if (found < 0).any():
        fail("request_invalid_positions", RequestRef("reference", str(references[int(np.flatnonzero(found < 0)[0])])))
    return found


def _sequence_tokens(
    system: MolecularSystem, layout: TokenLayout, tokens: NDArray[np.int64], sequence: str | Mapping[str, str]
) -> NDArray[np.int64]:
    r"""The rows' classes with each named instance's rows read from its one-letter sequence."""
    instance = _owner(system.instance_component_ptr)[layout.token_component]
    names = system.text("instance", system.instance_symbol[instance])
    present = list(dict.fromkeys(names.tolist()))
    if isinstance(sequence, str):
        if len(present) != 1:
            fail("request_invalid_sequence", RequestRef("sequence", "a bare string with several protein instances"))
        sequence = {present[0]: sequence}
    tokens = tokens.copy()
    for name, text in sequence.items():
        rows = np.flatnonzero(names == name)
        if len(rows) != len(text) or not set(text) <= set(PROTEINMPNN_VOCABULARY):
            fail("request_invalid_sequence", RequestRef(f"sequence[{name}]", text), rows=len(rows))
        tokens[rows] = encode_sequence(text, _PROTEIN)
    return tokens
