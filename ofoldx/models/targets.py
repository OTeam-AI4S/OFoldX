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
Training truth on the full canonical axis (replaces ``models/equivalence.py``).

Every atom of the prepared target carries one ``TruthState``; the family's symmetry candidates come once from
``chemistry.rdkit.symmetry_candidates`` under the family's ``SymmetryPolicy`` — no chemistry is re-derived here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal, Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..chemistry.rdkit import SymmetryCandidates, symmetry_candidates
from ..data.molecular import AtomFate, AtomSite, MolecularSystem
from ..data.structure import Structure
from ..data.tokenization import TOKEN_INDEX, Encoding, token_classes
from ..provenance import known
from .preparation import I, PreparedInput


class TruthState(IntEnum):
    OBSERVED = 0
    OBSERVED_UNOCCUPIED = 1
    UNOBSERVED = 2
    EXCLUDED_BY_LAYOUT = 3
    UNDETERMINED_CHEMISTRY = 4


@dataclass(frozen=True)
class SymmetryPolicy:
    r"""Which components receive symmetry candidates, whether resonance alternatives count, and whether atoms without
    truth are fixed points."""

    components: Literal["ligands", "all"]
    resonance: bool
    pin_unobserved: bool


@dataclass(frozen=True)
class StructureTarget:
    r"""
    Truth on the full canonical atom axis: coordinates (zero where not ``OBSERVED``), the per-atom ``TruthState``, the
    symmetry candidates of the consumed components, independently known B factors, and the structure's resolution
    (``None`` when it states none).
    """

    coordinates: NDArray[np.float64]
    truth: NDArray[np.uint8]
    symmetry: SymmetryCandidates
    b_factor: NDArray[np.float64]
    b_factor_mask: NDArray[np.bool_]
    resolution: float | None


@dataclass(frozen=True)
class SequenceTarget:
    r"""Sequence classes and their knownness on an explicitly aligned component axis.

    `mask` identifies matched known truth, not the objective's scoring domain.
    `tokens` follow the processor's declared vocabulary convention. Known but
    unsupported output classes remain known and require an explicit objective policy.
    """

    tokens: Tensor
    mask: Tensor

    def to(self, device: torch.device | str) -> Self:
        r"""Move sequence truth to a device without changing token or mask dtype."""
        return type(self)(tokens=self.tokens.to(device), mask=self.mask.to(device))


def align_target_components(
    structure: Structure, target: Structure, components: NDArray[np.int64]
) -> NDArray[np.int64]:
    r"""Match components by canonical instance identity and component ordinal; missing rows are -1."""
    return target.system.component_index(structure.system.component_refs(components))


def prepare_sequence_target(
    structure: Structure,
    target: Structure,
    components: Tensor,
    *,
    class_map: NDArray[np.int64] | None = None,
) -> SequenceTarget:
    r"""Align compact sequence truth to component-index tensors without copying input features."""
    source_rows = components.detach().cpu().numpy().reshape(-1)
    rows = align_target_components(structure, target, source_rows)
    truth = token_classes(target.system)
    tokens = np.full(len(rows), TOKEN_INDEX["<unk>"], dtype=np.int64)
    known = np.zeros(len(rows), dtype=bool)
    present = rows >= 0
    tokens[present] = truth.token[rows[present]]
    known[present] = truth.encoding[rows[present]] != Encoding.UNMAPPED
    if class_map is not None:
        tokens = class_map[tokens]
    shape = tuple(components.shape)
    return SequenceTarget(
        torch.from_numpy(tokens.reshape(shape)).to(components.device),
        torch.from_numpy(known.reshape(shape)).to(components.device),
    )


def match_atoms(system: MolecularSystem, atoms: NDArray[np.int64], other: Structure) -> NDArray[np.int64]:
    r"""The atoms in ``other`` with the same instance, component ordinal and atom name; ``-1`` for missing sites."""
    component = np.repeat(np.arange(len(system.component_symbol)), np.diff(system.component_atom_ptr))
    refs = system.component_refs(component[atoms])
    names = system.text("atom", system.atom_symbol[atoms])
    return other.system.atom_index([AtomSite(ref.instance, ref.component, str(name)) for ref, name in zip(refs, names)])


def prepare_target(
    prepared: PreparedInput[I], truth: Structure, *, symmetry: SymmetryPolicy, exclude_unoccupied: bool = True
) -> StructureTarget:
    r"""
    The training target of one prepared task, matching ``truth`` by exact molecular atom identity.

    The state of an atom is, by precedence: ``UNDETERMINED_CHEMISTRY`` (fate UNDETERMINED), ``EXCLUDED_BY_LAYOUT``
    (no slot), ``UNOBSERVED`` (no coordinates), ``OBSERVED_UNOCCUPIED`` (a KNOWN occupancy at or below zero, with
    ``exclude_unoccupied``), else ``OBSERVED``. Candidates are computed for the consumed components the policy names
    (``"ligands"``: those of class ``OTHER``), with the atoms that are not ``OBSERVED`` pinned when the policy says so.
    """

    system = prepared.context.structure.system
    count = len(system.component_symbol)
    atom = match_atoms(system, np.arange(len(system.atom_symbol)), truth)
    matched = atom >= 0
    observed = matched.copy()
    observed[matched] = truth.coordinate_mask[atom[matched]]
    state = np.full(len(atom), TruthState.UNOBSERVED, dtype=np.uint8)
    state[observed] = TruthState.OBSERVED
    if exclude_unoccupied:
        unoccupied = known(truth.occupancy_state) & (truth.occupancy <= 0)
        state[observed] = np.where(unoccupied[atom[observed]], TruthState.OBSERVED_UNOCCUPIED, TruthState.OBSERVED)
    state[prepared.context.slots.unaddressed] = TruthState.EXCLUDED_BY_LAYOUT
    state[system.atom_fate == AtomFate.UNDETERMINED] = TruthState.UNDETERMINED_CHEMISTRY
    observed = state == TruthState.OBSERVED
    layout = prepared.context.slots.layout
    chosen = np.zeros(count, dtype=bool)
    chosen[layout.token_component] = True
    if symmetry.components == "ligands":
        ligand = np.zeros(count, dtype=bool)
        ligand[layout.token_component] = layout.token_class == TOKEN_INDEX["OTHER"]
        chosen &= ligand
    pinned = ~observed if symmetry.pin_unobserved else np.zeros(len(state), dtype=bool)
    candidates, _ = symmetry_candidates(system, np.flatnonzero(chosen), resonance=symmetry.resonance, pinned=pinned)
    b_factor_mask = matched.copy()
    b_factor_mask[matched] = known(truth.b_factor_state[atom[matched]])
    b_factor = np.zeros(len(atom), dtype=np.float64)
    b_factor[b_factor_mask] = truth.b_factor[atom[b_factor_mask]]
    resolution = truth.resolution[known(truth.resolution_state)]
    coordinates = np.zeros((len(atom), 3), dtype=np.float64)
    coordinates[observed] = truth.coordinates[atom[observed]]
    return StructureTarget(
        coordinates=coordinates,
        truth=state,
        symmetry=candidates,
        b_factor=b_factor,
        b_factor_mask=b_factor_mask,
        resolution=float(resolution[0]) if len(resolution) else None,
    )
