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
NA-MPNN input processing, written against the shared preparation contract.

The layout addresses every protein, DNA and RNA component by its NA-MPNN backbone atoms (N, CA, C, O; the eleven DNA
and twelve RNA sugar-phosphate atoms); a component that does not hold its centre atom (CA, C1') is no row. A row is
valid when all its backbone atoms are observed. A row reads its canonical class, or its type's unknown (``X``,
``DNA_X``, ``RNA_X``) when the class is outside the released vocabulary, as upstream reads ``UNK``, ``DX``, ``RX``; with
``shared_nucleic_tokens`` an RNA row reads (and is sampled as) the DNA class of its base, as upstream's shared token
columns. Sequence design regenerates no coordinate: the scope is empty and the design producer builds the final target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ComponentRef, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, encode_sequence
from ...errors import RequestRef, fail
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
from .config import NaMpnnProcessorConfig
from .model import NA_MPNN_ATOM_INDEX, NA_MPNN_ATOM_NAMES, NAMPNN_VOCABULARY, NaMpnnInput

_PROTEIN, _DNA, _RNA = MonomerType.PROTEIN, MonomerType.DNA, MonomerType.RNA
_SUGAR_PHOSPHATE = ("OP1", "OP2", "P", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'")
_MONOMER_SLOTS = {
    _PROTEIN: ("N", "CA", "C", "O"),
    _DNA: (*_SUGAR_PHOSPHATE, "C1'"),
    _RNA: (*_SUGAR_PHOSPHATE, "O2'", "C1'"),
}

NAMPNN_LAYOUT = LayoutTable(
    family="nampnn",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=_MONOMER_SLOTS,
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: NA-MPNN reads coordinates and classes of its backbone atoms only; an excess valence elsewhere is not its input.
NAMPNN_GATING = GatingPolicy(refuse_valence_excess=False)

#: Per NA-MPNN polymer code (protein 0, DNA 1, RNA 2), the classes it generates.
GENERATABLE = (
    tuple(TOKEN_INDEX[letter] for letter in "ACDEFGHIKLMNPQRSTVWY"),
    tuple(TOKEN_INDEX[f"DNA_{base}"] for base in "ACGT"),
    tuple(TOKEN_INDEX[f"RNA_{base}"] for base in "ACGU"),
)
_TRAINED = np.array([TOKEN_INDEX[token] for token in NAMPNN_VOCABULARY], dtype=np.int64)
_UNKNOWN = {_PROTEIN: TOKEN_INDEX["X"], _DNA: TOKEN_INDEX["DNA_X"], _RNA: TOKEN_INDEX["RNA_X"]}
#: Per canonical class of a row, the class NA-MPNN reads: a class outside the released vocabulary reads as its
#: monomer type's unknown (upstream's ``UNK``, ``DX``, ``RX``).
AS_MODEL = np.array(
    [k if k in _TRAINED or not m else _UNKNOWN[MonomerType(m)] for k, m in enumerate(CLASS_MONOMER.tolist())],
    dtype=np.int64,
)
#: Per class NA-MPNN reads, the class an RNA row reads under shared nucleic tokens (RNA base → DNA base), and back.
RNA_AS_DNA = np.arange(len(TOKENS), dtype=np.int64)
DNA_AS_RNA = np.arange(len(TOKENS), dtype=np.int64)
for _r, _d in zip("ACGUX", "ACGTX"):
    _rna, _dna = TOKEN_INDEX[f"RNA_{_r}"], TOKEN_INDEX[f"DNA_{_d}"]
    RNA_AS_DNA[_rna], DNA_AS_RNA[_dna] = _dna, _rna
_AMINO = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]
#: Per NA-MPNN polymer code, the class of each standard component id it generates.
_STANDARD_IDS = (
    dict(zip(_AMINO, GENERATABLE[0])),
    dict(zip(("DA", "DC", "DG", "DT"), GENERATABLE[1])),
    dict(zip(("A", "C", "G", "U"), GENERATABLE[2])),
)
_CHANNELS = {
    monomer: np.array([NA_MPNN_ATOM_INDEX[name] for name in names], dtype=np.int64)
    for monomer, names in _MONOMER_SLOTS.items()
}


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


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

    def prepare_input(
        self,
        structure: Structure,
        *,
        design: PositionSelection = PositionSelection(),
        sequence: str | Mapping[str, str] | None = None,
        nucleic_acid_only: bool = False,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        symmetry_groups: Sequence[Sequence[ComponentRef]] = (),
        tied_weights: Mapping[ComponentRef, float] | None = None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None = None,
        pair_logit_bias: Tensor | None = None,
        global_omit_tokens: Sequence[str] = (),
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[NaMpnnInput]:
        r"""
        Prepare one mixed-polymer structure for scoring or sequence design.

        Args:
            structure: The target; every protein, DNA and RNA component holding its centre atom is a row.
            design: The positions to design or score among the valid rows (default every one).
            sequence: Optional one-letter sequence scored instead of the target's classes, read per row under its
                monomer type (a bare string needs exactly one polymer instance); a class outside NA-MPNN's vocabulary
                is refused.
            nucleic_acid_only: Whether only DNA and RNA rows may be designed (protein stays context).
            conditions: Allowed-identity and tied-identity conditions; a tie stays within one polymer type.
            symmetry_groups: Residue groups sharing one decoding position (across polymer types too).
            tied_weights: Optional sampling coefficient for each member of a tied or symmetric group.
            residue_logit_bias: Per-residue bias vectors over the 72 canonical classes.
            pair_logit_bias: Optional adjacent-class bias matrix over the canonical classes, shape ``(72, 72)``.
            global_omit_tokens: Canonical class names of NA-MPNN's vocabulary excluded from generation; under shared
                nucleic tokens an RNA base is omitted by its DNA class (for DNA and RNA rows alike) and an ``RNA_``
                name is refused.
            cropper: Optional training crop over the rows; needs ``generator``.
            generator: The numpy generator of the crop.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        observed = structure.coordinate_mask & ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
        layout = token_layout(system, NAMPNN_LAYOUT, atom_mask=_residue_atoms(system))
        valid = _valid_rows(slot_map(system, layout, NAMPNN_LAYOUT), observed)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        designable = valid & ((CLASS_MONOMER[layout.token_class] != _PROTEIN) if nucleic_acid_only else True)
        allowed[layout.token_component[designable]] = True
        positions = resolve_positions(structure, design, allowed=allowed)
        if cropper is not None:
            named = _named_components(system, conditions, symmetry_groups, tied_weights, residue_logit_bias)
            required = np.isin(layout.token_component, named)
            layout = crop_token_layout(
                structure, layout, cropper, required=required, generator=cast(np.random.Generator, generator)
            )
        slots = slot_map(system, layout, NAMPNN_LAYOUT)
        accepted = gate(structure, slots, NAMPNN_GATING)
        valid = _valid_rows(slots, observed)
        classes, losses = model_tokens(system, layout)
        tokens = AS_MODEL[classes]
        lossy = np.union1d(losses.component, layout.token_component[tokens != classes])
        if sequence is not None:
            tokens = _sequence_tokens(system, layout, tokens, sequence)
        kind = np.array([NAMPNN_LAYOUT.monomer_types[MonomerType(m)] for m in CLASS_MONOMER[layout.token_class]])
        if self.config.shared_nucleic_tokens:
            tokens = np.where(kind == 2, RNA_AS_DNA[tokens], tokens)
        features = self._features(
            structure,
            slots,
            observed,
            kind,
            tokens,
            valid,
            np.isin(layout.token_component, positions) & valid,
            conditions=conditions,
            symmetry_groups=symmetry_groups,
            tied_weights=tied_weights,
            residue_logit_bias=residue_logit_bias,
            pair_logit_bias=pair_logit_bias,
            global_omit_tokens=global_omit_tokens,
        )
        records = Records.concat((component_records("encoding_loss", lossy), accepted))
        record = PreparationRecord(records, np.zeros(len(system.component_symbol), dtype=np.uint8))
        return PreparedInput(
            features,
            PreparationContext(structure, slots, RegenerationScope.none(system), record, spec=self.preparation_spec),
        )

    def prepare_target(self, target: Structure, *, prepared: PreparedInput[NaMpnnInput]) -> SequenceTarget:
        r"""Align polymer sequence truth using this processor's nucleic-class convention.

        Shared nucleic tokens map RNA bases to DNA output columns. Other known
        canonical classes are retained; unavailable identities are masked.
        """
        self._validate_prepared(prepared)
        result = prepare_sequence_target(
            prepared.context.structure,
            target,
            prepared.features.component_index,
            class_map=RNA_AS_DNA if self.config.shared_nucleic_tokens else None,
        )
        kind = torch.as_tensor(CLASS_MONOMER, device=result.tokens.device)[result.tokens]
        if bool((result.mask & (kind == MonomerType.NONE)).any()):
            fail("model_capacity", RequestRef("target", "polymer sequence supervision is required"))
        return result

    def generatable(self, kind: int) -> tuple[int, ...]:
        r"""The classes a row of NA-MPNN polymer code ``kind`` is generated as (RNA as DNA under shared tokens)."""
        return GENERATABLE[1] if kind == 2 and self.config.shared_nucleic_tokens else GENERATABLE[kind]

    def _features(
        self,
        structure: Structure,
        slots: SlotMap,
        observed: NDArray[np.bool_],
        kind: NDArray[np.int64],
        tokens: NDArray[np.int64],
        valid: NDArray[np.bool_],
        designed: NDArray[np.bool_],
        *,
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        symmetry_groups: Sequence[Sequence[ComponentRef]],
        tied_weights: Mapping[ComponentRef, float] | None,
        residue_logit_bias: Mapping[ComponentRef, Tensor] | None,
        pair_logit_bias: Tensor | None,
        global_omit_tokens: Sequence[str],
    ) -> NaMpnnInput:
        r"""The NA-MPNN tensors: backbone channels, tie closure, class support, biases."""
        system = structure.system
        layout = slots.layout
        rows = len(layout.token_component)
        vocab = len(TOKENS)
        device, dtype = self.device, self.dtype
        designed = designed.copy()
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

        def union(members: list[int]) -> None:
            for member in members[1:]:
                first, second = find(members[0]), find(member)
                if first != second:
                    parent[second] = first

        restrictions: dict[int, set[int]] = {}
        for group in symmetry_groups:
            union([row(reference) for reference in group])
        for condition in conditions:
            if isinstance(condition, TiedResidueIdentities):
                for group in condition.groups:
                    members = [row(reference) for reference in group]
                    if len({int(kind[member]) for member in members}) != 1:
                        fail("value_invalid", RequestRef("conditions", "a tie across polymer types"))
                    union(members)
            else:
                for reference in condition.positions:
                    at = row(reference)
                    ids = _STANDARD_IDS[kind[at]]
                    classes = {int(ids[name]) for name in condition.component_ids if name in ids}
                    if kind[at] == 2 and self.config.shared_nucleic_tokens:
                        classes = {int(RNA_AS_DNA[k]) for k in classes}
                    restrictions[at] = restrictions[at] & classes if at in restrictions else classes

        omit = set()
        for name in global_omit_tokens:
            if name not in NAMPNN_VOCABULARY:
                fail("value_invalid", RequestRef("global_omit_tokens", name))
            if self.config.shared_nucleic_tokens and name.startswith("RNA_"):
                expected = "a DNA class: RNA is generated in the DNA columns"
                fail("value_invalid", RequestRef("global_omit_tokens", name), expected=expected)
            omit.add(TOKEN_INDEX[name])
        support = [set(self.generatable(int(k))) - omit for k in kind]
        groups: dict[int, list[int]] = {}
        for at in range(rows):
            groups.setdefault(find(at), []).append(at)
        for members in groups.values():
            fixed = {int(tokens[at]) for at in members if not designed[at]}
            if len(fixed) > 1:
                fail("value_invalid", RequestRef("conditions", "a tied group with conflicting fixed identities"))
            declared = [restrictions[at] for at in members if at in restrictions]
            if declared:
                common = set.intersection(*declared)
                for at in members:
                    narrowed = support[at] & common
                    if not narrowed and int(tokens[at]) in common:
                        designed[at] = False
                    elif not narrowed:
                        fail("value_invalid", RequestRef("conditions", "allowed identities with no generatable one"))
                    else:
                        support[at] = narrowed
            if fixed and any(designed[at] for at in members):
                anchor = next(iter(fixed))
                if not all(anchor in support[at] for at in members):
                    fail("value_invalid", RequestRef("conditions", "a tied fixed identity outside the common support"))
                for at in members:
                    support[at] = {anchor}
            if any(designed[at] for at in members) and not set.intersection(*(support[at] for at in members)):
                fail("value_invalid", RequestRef("conditions", "a tied group with no common generatable identity"))
        allowed_token_mask = torch.zeros(1, rows, vocab, dtype=torch.bool, device=device)
        for at, classes in enumerate(support):
            allowed_token_mask[0, at, sorted(classes)] = True
        roots = [find(at) for at in range(rows)]
        group_id = {root: index for index, root in enumerate(dict.fromkeys(roots))}

        weights = torch.ones(1, rows, device=device, dtype=dtype)
        grouped = {at for members in groups.values() if len(members) > 1 for at in members}
        for reference, weight in (tied_weights or {}).items():
            at = row(reference)
            if at not in grouped:
                fail("value_invalid", RequestRef("tied_weights", str(reference)))
            weights[0, at] = weight
        bias = torch.zeros(1, rows, vocab, device=device, dtype=dtype)
        for reference, value in (residue_logit_bias or {}).items():
            converted = value.to(device=device, dtype=dtype)
            if converted.shape != (vocab,):
                fail("value_invalid", RequestRef("residue_logit_bias", str(tuple(converted.shape))), expected=vocab)
            bias[0, row(reference)] = converted

        instance = _owner(system.instance_component_ptr)[layout.token_component]
        chain_of = {value: index for index, value in enumerate(dict.fromkeys(instance.tolist()))}
        chain_index = np.array([chain_of[value] for value in instance.tolist()], dtype=np.int64)
        residue_index = layout.token_component - system.instance_component_ptr[instance]

        monomer = CLASS_MONOMER[layout.token_class]
        channel = np.concatenate([_CHANNELS[MonomerType(m)] for m in monomer.tolist()] or [np.zeros(0, np.int64)])
        bound = slots.atom >= 0
        present = bound & observed[np.maximum(slots.atom, 0)]
        atom_mask = np.zeros((rows, len(NA_MPNN_ATOM_NAMES)), dtype=bool)
        atom_mask[slots.token, channel] = present
        coordinates = np.zeros((rows, len(NA_MPNN_ATOM_NAMES), 3))
        coordinates[slots.token[present], channel[present]] = structure.coordinates[slots.atom[present]]
        points = project_model_coordinates(
            torch.from_numpy(coordinates),
            torch.from_numpy(atom_mask & valid[:, None]),
            dtype=dtype,
            feature="mixed-polymer backbone",
        )

        def tensor(values, element=torch.int64) -> Tensor:
            return torch.as_tensor(np.asarray(values), dtype=element, device=device).unsqueeze(0)

        return NaMpnnInput(  # type: ignore[call-arg]
            atom_coordinates=points.to(device=device).unsqueeze(0),
            atom_mask=tensor(atom_mask, torch.bool),
            sequence_tokens=tensor(tokens),
            residue_mask=tensor(valid, torch.bool),
            residue_index=tensor(residue_index),
            chain_index=tensor(chain_index),
            polymer_type_index=tensor(np.where(valid, kind, 3)),
            design_mask=tensor(designed, torch.bool),
            tied_group_ids=tensor([group_id[root] for root in roots]),
            component_index=tensor(layout.token_component),
            tied_weights=weights,
            residue_logit_bias=bias,
            allowed_token_mask=allowed_token_mask,
            pair_logit_bias=_pair_logit_bias(pair_logit_bias, residue_index, chain_index, device=device, dtype=dtype),
            batch_size=[],
        )


def _residue_atoms(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""The atoms of components holding their centre atom (CA, C1'): NA-MPNN reads residues only."""
    component = _owner(system.component_atom_ptr)
    names = system.text("atom", system.atom_symbol)
    holds = np.zeros(len(system.component_symbol), dtype=bool)
    holds[component[np.isin(names, list(NAMPNN_LAYOUT.centre_atom.values()))]] = True
    return holds[component]


def _valid_rows(slots: SlotMap, observed: NDArray[np.bool_]) -> NDArray[np.bool_]:
    r"""Per row, whether every backbone slot is bound to an observed atom."""
    present = (slots.atom >= 0) & observed[np.maximum(slots.atom, 0)]
    count = len(slots.layout.token_component)
    return np.bincount(slots.token[~present], minlength=count) == 0


def _named_components(system: MolecularSystem, conditions, symmetry_groups, *mappings) -> NDArray[np.int64]:
    r"""The components every condition, symmetry group and per-residue input names (a crop must keep them)."""
    references: list[ComponentRef] = [reference for group in symmetry_groups for reference in group]
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
    if isinstance(sequence, str):
        present = list(dict.fromkeys(names.tolist()))
        if len(present) != 1:
            fail("request_invalid_sequence", RequestRef("sequence", "a bare string with several polymer instances"))
        sequence = {present[0]: sequence}
    monomer = CLASS_MONOMER[layout.token_class]
    tokens = tokens.copy()
    for name, text in sequence.items():
        rows = np.flatnonzero(names == name)
        if len(rows) != len(text):
            fail("request_invalid_sequence", RequestRef(f"sequence[{name}]", text), rows=len(rows))
        read = np.array([encode_sequence(letter, MonomerType(m))[0] for letter, m in zip(text, monomer[rows])])
        if not np.isin(read, _TRAINED).all():
            fail("request_invalid_sequence", RequestRef(f"sequence[{name}]", text), rows=len(rows))
        tokens[rows] = read
    return tokens


def _pair_logit_bias(
    matrix: Tensor | None,
    residue_index: NDArray[np.int64],
    chain_index: NDArray[np.int64],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor | None:
    r"""The directed adjacent-class pair bias of a ``[72, 72]`` matrix between consecutive rows of one chain."""
    if matrix is None:
        return None
    vocab = len(TOKENS)
    matrix = matrix.to(device=device, dtype=dtype)
    if matrix.shape != (vocab, vocab):
        fail("value_invalid", RequestRef("pair_logit_bias", str(tuple(matrix.shape))), expected=vocab)
    rows = len(residue_index)
    residue = torch.as_tensor(residue_index, device=device)
    chain = torch.as_tensor(chain_index, device=device)
    same_chain = (chain[:, None] == chain[None, :]).to(dtype)
    step = (residue[1:] - residue[:-1] == 1).to(dtype)
    upper = torch.diag(step, 1) * same_chain
    lower = torch.diag(step, -1) * same_chain
    pair = upper[:, None, :, None] * matrix[None, :, None, :] + lower[:, None, :, None] * matrix.T[None, :, None, :]
    return pair.reshape(1, rows, vocab, rows, vocab)
