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
LigandMPNN input processing, written against the shared preparation contract.

The rows are the protein components holding an observed CA (canonical order), each addressed by its atom37 names:
N, CA, C, O are the backbone, the names after O the side-chain context. Every placed heavy atom of a component
outside the protein classes and the water entities is a context candidate (OFoldX's rule: a protein-class component
without a CA, such as a cap, is neither a row nor context); the nearest ones to each row's virtual Cβ are consumed
and appended to the slot map as slots of no token. The gate reads every row's backbone, the side chains of the valid
rows kept fixed and the consumed context. Tokens are the canonical classes (MSE reads ``M`` with its
``encoding_loss`` record). Sequence design regenerates no coordinate: the scope is empty and the design producer
builds the final target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar, cast

import numpy as np
import torch
from chanfig import Config
from numpy.typing import NDArray
from torch import Tensor

from ...data import residue_constants as rc
from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ComponentRef, EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, token_classes
from ...errors import RequestRef, fail
from ...modules.geometry import virtual_beta_carbon
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
from .config import LigandMpnnProcessorConfig
from .model import LIGANDMPNN_GENERATABLE, LigandMpnnInput

_PROTEIN = MonomerType.PROTEIN
_ATOM37 = tuple(rc.atom_types)
_BACKBONE = [rc.atom_order[name] for name in ("N", "CA", "C", "O")]
_SIDE_CHAIN = 5  # atom37 columns from CG on are the residue context (upstream ``xyz_37[:, :, 5:]``)

LIGANDMPNN_LAYOUT = LayoutTable(
    family="ligandmpnn",
    monomer_types={_PROTEIN: 0},
    residue_slots=None,
    monomer_slots={_PROTEIN: _ATOM37},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: LigandMPNN reads coordinates and elements of its consumed atoms and the classes of its rows, nothing else.
LIGANDMPNN_GATING = GatingPolicy(
    refuse_undetermined_in_slots=False,
    refuse_unknown_kind_in_slots=False,
    refuse_incomplete_topology_in_slots=False,
    refuse_valence_excess=False,
    required_states=frozenset({"atomic_number"}),
)

_GENERATABLE = frozenset(TOKEN_INDEX[token] for token in LIGANDMPNN_GENERATABLE)
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
_CLASS_OF_ID = {name: TOKEN_INDEX[letter] for letter, name in zip("ARNDCQEGHILKMFPSTWYV", _AMINO)}


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class LigandMpnnProcessor(Processor):
    r"""
    Prepare protein-ligand structures and sequence constraints for LigandMPNN.

    Args:
        config: LigandMPNN context options. Defaults to ``LigandMpnnProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class: ClassVar[type[Config] | None] = LigandMpnnProcessorConfig
    config: LigandMpnnProcessorConfig
    layout: ClassVar[LayoutTable] = LIGANDMPNN_LAYOUT

    def __init__(
        self,
        config: LigandMpnnProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = LigandMpnnProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def prepare_input(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        global_logit_bias: Tensor | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        use_atom_context: bool = True,
        ligand_interface_cutoff: float = 8.0,
        filter_by_occupancy: bool = True,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[LigandMpnnInput]:
        r"""
        Prepare a structure for LigandMPNN.

        Args:
            structure: The target; every protein component holding an observed CA is a row, valid when its N, CA, C
                and O are observed.
            design: The positions to design among the valid rows (default every one).
            conditions: Allowed-identity and tied-identity conditions.
            tied_weights: Optional sampling coefficient for each member of a tied group (default one).
            global_logit_bias: Bias vector over the 72 canonical classes added at every row.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            use_atom_context: Whether the nearest non-protein, non-water heavy atoms are visible to the model.
            ligand_interface_cutoff: Distance in ångström from a row's virtual Cβ to its nearest context atom below
                which the row is at the ligand interface.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.
            cropper: Optional training crop over the rows; needs ``generator``.
            generator: The numpy generator of the crop.

        Notes:
            Rows keep canonical component order; ``residue_index`` is the component's position in its instance and
            ``chain_index`` the instance's order of first appearance.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        observed = structure.coordinate_mask.copy()
        if filter_by_occupancy:
            observed &= ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
        atom_component = _owner(system.component_atom_ptr)
        protein = CLASS_MONOMER[token_classes(system).token] == _PROTEIN
        rows = np.zeros(len(system.component_symbol), dtype=bool)
        rows[atom_component[observed & (system.text("atom", system.atom_symbol) == "CA")]] = True
        layout = token_layout(system, LIGANDMPNN_LAYOUT, atom_mask=(rows & protein)[atom_component])
        valid = _valid_rows(slot_map(system, layout, LIGANDMPNN_LAYOUT), observed)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component[valid]] = True
        positions = resolve_positions(structure, design, allowed=allowed)
        if cropper is not None:
            named = _named_components(system, conditions, tied_weights, residue_logit_bias)
            required = np.isin(layout.token_component, named)
            layout = crop_token_layout(
                structure, layout, cropper, required=required, generator=cast(np.random.Generator, generator)
            )
        slots = slot_map(system, layout, LIGANDMPNN_LAYOUT)
        valid = _valid_rows(slots, observed)
        tokens, losses = model_tokens(system, layout)
        candidates = _context_atoms(system, observed, protein)
        designed = np.isin(layout.token_component, positions) & valid
        input, context = self._features(
            structure,
            slots,
            layout,
            tokens,
            observed,
            valid,
            designed,
            candidates,
            conditions=conditions,
            tied_weights=tied_weights,
            global_logit_bias=global_logit_bias,
            residue_logit_bias=residue_logit_bias,
            use_atom_context=use_atom_context,
            ligand_interface_cutoff=ligand_interface_cutoff,
        )
        # The backbone of every row is read, the side chain only of a valid row kept fixed (side-chain context).
        column = np.arange(len(slots.atom)) % len(_ATOM37)
        context_row = (valid & ~designed)[slots.token]
        read = np.isin(column, _BACKBONE) | ((column >= _SIDE_CHAIN) & context_row)
        placed = (slots.atom >= 0) & observed[np.maximum(slots.atom, 0)]
        accepted = gate(structure, _with_context(system, slots, context, placed & read), LIGANDMPNN_GATING)
        records = Records.concat((component_records("encoding_loss", losses.component), accepted))
        record = PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))
        slots = _with_context(system, slots, context, np.ones(len(slots.atom), dtype=bool))
        return PreparedInput(
            input,
            PreparationContext(structure, slots, RegenerationScope.none(system), record, spec=self.preparation_spec),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[LigandMpnnInput]) -> SequenceTarget:
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
        observed: NDArray[np.bool_],
        valid: NDArray[np.bool_],
        designed: NDArray[np.bool_],
        candidates: NDArray[np.int64],
        *,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        tied_weights: Mapping[ComponentRef, float] | None,
        global_logit_bias: Tensor | None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None,
        use_atom_context: bool,
        ligand_interface_cutoff: float,
    ) -> tuple[LigandMpnnInput, NDArray[np.int64]]:
        r"""The LigandMPNN tensors and the context atoms they consume (upstream featurisation)."""
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

        # explicit restrictions and fixed tied members (anchors) bound the support; X is never generated
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
            for reference, weight in tied_weights.items():
                at = row(reference)
                if at not in grouped:
                    fail("value_invalid", RequestRef("tied_weights", str(reference)))
                tied_weight_values[0, at] = weight

        def vector(value: Tensor, name: str) -> Tensor:
            converted = value.to(device=device, dtype=dtype)
            if converted.shape != (vocab,):
                fail("value_invalid", RequestRef(name, str(tuple(converted.shape))), expected=vocab)
            return converted

        bias_values = None
        if global_logit_bias is not None or residue_logit_bias:
            bias_values = torch.zeros(1, rows, vocab, device=device, dtype=dtype)
            if global_logit_bias is not None:
                bias_values[0] += vector(global_logit_bias, "global_logit_bias")
            for reference, value in (residue_logit_bias or {}).items():
                bias_values[0, row(reference)] += vector(value, "residue_logit_bias")
            for members in tied_groups:
                values = bias_values[0, list(members)]
                if not torch.equal(values, values[:1].expand_as(values)):
                    fail("value_invalid", RequestRef("residue_logit_bias", "differs within a tied group"))

        atoms = slots.atom.reshape(rows, len(_ATOM37))
        bound = atoms >= 0
        coordinates = np.where(bound[..., None], structure.coordinates[np.maximum(atoms, 0)], 0.0)
        numbers = np.where(bound, system.atomic_number[np.maximum(atoms, 0)], 0).astype(np.int64)
        present = bound & observed[np.maximum(atoms, 0)]
        backbone = project_model_coordinates(
            torch.from_numpy(coordinates[:, _BACKBONE]),
            torch.from_numpy(valid[:, None].repeat(4, axis=1)),
            dtype=dtype,
            feature="protein backbone",
        ).to(device)
        residue_mask = torch.from_numpy(valid.copy()).to(device)
        side_chain_mask = torch.from_numpy(present[:, _SIDE_CHAIN:].copy())
        side_chain = project_model_coordinates(
            torch.from_numpy(coordinates[:, _SIDE_CHAIN:]), side_chain_mask, dtype=dtype, feature="side-chain context"
        ).to(device)
        selected, selected_numbers, selected_mask, interface, consumed = self._select_context(
            structure, backbone, residue_mask, candidates, ligand_interface_cutoff=ligand_interface_cutoff
        )
        if not use_atom_context:
            selected_mask = torch.zeros_like(selected_mask)
            consumed = np.zeros(0, dtype=np.int64)
        instance = _owner(system.instance_component_ptr)[layout.token_component]
        chain_of = {value: index for index, value in enumerate(dict.fromkeys(instance.tolist()))}
        chain_index = np.array([chain_of[value] for value in instance.tolist()], dtype=np.int64)

        def tensor(values, kind=torch.int64) -> Tensor:
            return torch.as_tensor(np.asarray(values), dtype=kind, device=device).unsqueeze(0)

        input = LigandMpnnInput(  # type: ignore[call-arg]
            backbone_coordinates=backbone.unsqueeze(0),
            sequence_tokens=tensor(tokens),
            residue_mask=residue_mask.unsqueeze(0),
            residue_index=tensor(layout.token_component - system.instance_component_ptr[instance]),
            chain_index=tensor(chain_index),
            context_coordinates=selected.unsqueeze(0),
            context_atomic_numbers=selected_numbers.unsqueeze(0),
            context_mask=selected_mask.unsqueeze(0),
            residue_context_coordinates=side_chain.unsqueeze(0),
            residue_context_atomic_numbers=tensor(numbers[:, _SIDE_CHAIN:]),
            residue_context_mask=side_chain_mask.to(device).unsqueeze(0),
            design_mask=tensor(designed, torch.bool),
            ligand_interface_mask=interface.unsqueeze(0),
            component_index=tensor(layout.token_component),
            tied_group_ids=tensor(tied_group_ids),
            tied_weights=tied_weight_values,
            allowed_token_mask=allowed_token_mask,
            residue_logit_bias=bias_values,
            batch_size=[],
        )
        return input, consumed

    def _select_context(
        self,
        structure: Structure,
        backbone: Tensor,
        residue_mask: Tensor,
        candidates: NDArray[np.int64],
        *,
        ligand_interface_cutoff: float,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, NDArray[np.int64]]:
        r"""
        Per row, the nearest ``num_context_atoms`` candidates to its virtual Cβ (padded), whether the row is at the
        ligand interface, and the candidates the valid rows select.
        """

        size = self.config.num_context_atoms
        rows = backbone.shape[0]
        device, dtype = self.device, self.dtype
        coordinates = torch.zeros(rows, size, 3, device=device, dtype=dtype)
        numbers = torch.zeros(rows, size, device=device, dtype=torch.int64)
        mask = torch.zeros(rows, size, device=device, dtype=torch.bool)
        interface = torch.zeros(rows, device=device, dtype=torch.bool)
        if not len(candidates):
            return coordinates, numbers, mask, interface, np.zeros(0, dtype=np.int64)
        points = torch.from_numpy(structure.coordinates[candidates])
        context = project_model_coordinates(
            points, torch.ones_like(points[..., 0], dtype=torch.bool), dtype=dtype, feature="ligand context"
        ).to(device)
        elements = torch.from_numpy(structure.system.atomic_number[candidates].astype(np.int64)).to(device)
        beta = virtual_beta_carbon(backbone[:, 0], backbone[:, 1], backbone[:, 2])
        squared = ((beta[:, None] - context[None]) ** 2).sum(dim=-1)
        pair = residue_mask[:, None].expand_as(squared)
        maximum = torch.where(pair, squared, 0).amax(dim=-1, keepdim=True)
        adjusted = torch.where(pair, squared, maximum + 1)
        k = min(size, len(candidates))
        nearest = torch.topk(adjusted, k, dim=-1, largest=False).indices
        coordinates[:, :k] = context[nearest]
        numbers[:, :k] = elements[nearest]
        mask[:, :k] = pair.gather(1, nearest)
        interface = residue_mask & (adjusted.gather(1, nearest[:, :1])[:, 0] < ligand_interface_cutoff**2)
        consumed = np.unique(candidates[nearest.cpu().numpy()[residue_mask.cpu().numpy()]])
        return coordinates, numbers, mask, interface, consumed


def _valid_rows(slots: SlotMap, observed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    r"""Per row, whether its N, CA, C and O slots are bound to observed atoms."""
    atoms = slots.atom.reshape(-1, len(_ATOM37))[:, _BACKBONE]
    return ((atoms >= 0) & observed[np.maximum(atoms, 0)]).all(axis=1)


def _context_atoms(
    system: MolecularSystem, observed: NDArray[np.bool_], protein: NDArray[np.bool_]
) -> NDArray[np.int64]:
    r"""The observed heavy atoms of every component outside the protein classes and water entities, in atom order."""
    s = system.arrays()
    atom_component = _owner(s["component_atom_ptr"])
    entity = s["instance_entity"][_owner(s["instance_component_ptr"])]
    water = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.WATER)
    keep = observed & ~(protein | water)[atom_component] & (s["atomic_number"] != 1)
    return np.flatnonzero(keep).astype(np.int64)


def _with_context(
    system: MolecularSystem, slots: SlotMap, context: NDArray[np.int64], keep: NDArray[np.bool_]
) -> SlotMap:
    r"""The slots of ``keep`` with the consumed context atoms appended as slots of no token."""
    atom = np.concatenate((slots.atom[keep], context))
    addressed = np.zeros(len(system.atom_symbol), dtype=bool)
    addressed[atom[atom >= 0]] = True
    return SlotMap(
        atom=atom,
        token=np.concatenate((slots.token[keep], np.full(len(context), -1, dtype=np.int64))),
        layout=slots.layout,
        unaddressed=np.flatnonzero(~addressed).astype(np.int64),
    )


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
