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
MPNN Packer input processing, written against the shared preparation contract.

Every protein residue is a row: a component the atom14 tables represent is addressed by its atom14 names, any other
(MSE) enters as fixed N, CA, C and O with an encoding-loss record and can be context only — packing it is
``UnsupportedError(model_capacity)``. With atom context on, the placed non-protein heavy atoms nearest each residue's
virtual Cβ are consumed too. The regeneration scope is the side chains of the packed residues.
"""

from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
import torch
from chanfig import Config
from numpy.typing import NDArray
from torch import Tensor

from ...data import residue_constants as rc
from ...data.molecular import EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import TOKEN_INDEX, TokenLayout
from ...errors import RequestRef, StructureRef, fail
from ...model import Model
from ...modules.geometry import virtual_beta_carbon
from ...processor import Processor
from ...provenance import Records, known
from ..design import PositionSelection, resolve_positions
from ..layout import LayoutTable, slot_map, token_layout
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
from .config import MpnnPackerConfig, MpnnPackerProcessorConfig
from .model import AMINO_ACID_COMPONENT_IDS, MpnnPackerInput

_PROTEIN = MonomerType.PROTEIN
_BACKBONE = ("N", "CA", "C", "O")
#: Atoms of a packed residue that keep their coordinates: the backbone, its terminal atoms and amide hydrogens.
_KEPT = frozenset((*_BACKBONE, "OXT", "HXT", "H", "H2", "H3"))
_ATOM14 = {name: tuple(rc.AMINO_ACID_ATOM14_NAMES[name]) for name in AMINO_ACID_COMPONENT_IDS}
_LETTER = dict(
    zip(
        [
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
        ],
        "ARNDCQEGHILKMFPSTWYV",
    )
)

PACKER_LAYOUT = LayoutTable(
    family="mpnn_packer",
    monomer_types={_PROTEIN: 0},
    residue_slots=_ATOM14,
    monomer_slots={_PROTEIN: _BACKBONE},
    hydrogens="exclude",
    per_atom_components="none",
    exclude_designated_leaving=False,
    centre_atom={_PROTEIN: "CA"},
    frame_atoms={},
    representative={},
    representative_default={},
)
#: The packer reads coordinates, elements and identity tokens of its consumed atoms, nothing else.
PACKER_GATING = GatingPolicy(
    refuse_undetermined_in_slots=False,
    refuse_unknown_kind_in_slots=False,
    refuse_incomplete_topology_in_slots=False,
    refuse_valence_excess=False,
    required_states=frozenset({"atomic_number"}),
)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class MpnnPackerProcessor(Processor):
    r"""
    Prepare structures for side-chain packing.

    Args:
        config: Side-chain context options. Defaults to ``MpnnPackerProcessorConfig()``.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class: ClassVar[type[Config] | None] = MpnnPackerProcessorConfig
    config: MpnnPackerProcessorConfig

    def __init__(
        self,
        config: MpnnPackerProcessorConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = MpnnPackerProcessorConfig() if config is None else config
        super().__init__(device=device, dtype=dtype)

    def validate_model_input_layout(self, model: Model) -> None:
        expected = cast(MpnnPackerConfig, model.config).num_context_atoms
        if self.config.num_context_atoms != expected:
            fail("value_invalid", RequestRef("num_context_atoms", str(self.config.num_context_atoms)), model=expected)

    def prepare_input(
        self,
        structure: Structure,
        *,
        positions: PositionSelection = PositionSelection(),
        use_atom_context: bool = True,
        filter_by_occupancy: bool = True,
    ) -> PreparedInput[MpnnPackerInput]:
        r"""
        Prepare one structure for packing ``positions``.

        Args:
            structure: The target (a read structure or a design's final target).
            positions: The residues to pack among the protein residues with an observed backbone; naming none packs
                every such residue the atom14 tables represent.
            use_atom_context: Whether non-protein heavy atoms are visible to the packer.
            filter_by_occupancy: Whether atoms with a KNOWN occupancy at or below zero are treated as unobserved.

        Raises:
            UnsupportedError: ``model_capacity`` for a packed residue the atom14 tables cannot represent.
        """

        system = structure.system
        observed = structure.coordinate_mask.copy()
        if filter_by_occupancy:
            observed &= ~(known(structure.occupancy_state) & (structure.occupancy <= 0))
        layout = token_layout(system, PACKER_LAYOUT)
        slots = slot_map(system, layout, PACKER_LAYOUT)
        rows = np.arange(len(layout.token_component))
        starts = np.searchsorted(slots.token, rows)
        residue = np.bincount(slots.token, minlength=len(rows)) >= len(_BACKBONE)  # a residue token, not an atom
        backbone = slots.atom[np.minimum(starts[:, None] + np.arange(4), len(slots.atom) - 1)]
        valid = residue & ((backbone >= 0) & observed[np.maximum(backbone, 0)]).all(axis=1)
        packable = np.isin(system.text("component", system.component_symbol), AMINO_ACID_COMPONENT_IDS)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[layout.token_component[valid]] = True
        named = (positions.components, positions.instances, positions.indices, positions.author)
        if all(form is None for form in named):
            allowed &= packable
        components = resolve_positions(structure, positions, allowed=allowed)
        if not packable[components].all():
            refused = components[~packable[components]]
            fail(
                "model_capacity",
                StructureRef(component=int(refused[0])),
                position=str(system.component_refs(refused[:1])[0]),
            )
        packed = np.isin(layout.token_component, components)
        input, context = self._features(structure, slots, layout, starts, observed, valid, packed, use_atom_context)
        # the placed backbone of every row and side chain of every context row are read; a packed side chain is
        # the regeneration scope, not an input.
        rank = np.arange(len(slots.atom)) - starts[slots.token]
        placed = (slots.atom >= 0) & observed[np.maximum(slots.atom, 0)]
        read = placed & ((rank < len(_BACKBONE)) | ~packed[slots.token])
        accepted = gate(structure, _with_context(system, slots, context, read), PACKER_GATING)
        unpackable = np.unique(layout.token_component[~packable[layout.token_component]])
        losses = component_records("encoding_loss", unpackable)
        atom_component = _owner(system.component_atom_ptr)
        names = system.text("atom", system.atom_symbol)
        side_chain = np.isin(atom_component, components) & ~np.isin(names, list(_KEPT))
        record = PreparationRecord(
            Records.concat((losses, accepted)), np.zeros(len(system.component_symbol), dtype=np.uint8)
        )
        slots = _with_context(system, slots, context, np.ones(len(slots.atom), dtype=bool))
        return PreparedInput(
            input,
            PreparationContext(
                structure, slots, RegenerationScope.atoms_of(system, side_chain), record, spec=self.preparation_spec
            ),
        )

    def _features(
        self,
        structure: Structure,
        slots: SlotMap,
        layout: TokenLayout,
        starts: NDArray[np.int64],
        observed: NDArray[np.bool_],
        valid: NDArray[np.bool_],
        packed: NDArray[np.bool_],
        use_atom_context: bool,
    ) -> tuple[MpnnPackerInput, NDArray[np.int64]]:
        r"""The packer tensors and the context atoms they consume (upstream featurisation)."""
        system = structure.system
        rows = len(layout.token_component)
        counts = np.diff(np.append(starts, len(slots.atom)))
        names = system.text("atom", system.atom_symbol)
        atom37 = np.zeros((rows, len(rc.atom_types), 3))
        atom37_mask = np.zeros((rows, len(rc.atom_types)), dtype=bool)
        atom14 = np.full((rows, 14), -1, dtype=np.int64)
        chi_index = np.full((rows, 14), -1, dtype=np.int64)
        chi_mask = np.zeros((rows, 4), dtype=bool)
        ids = system.text("component", system.component_symbol[layout.token_component]).tolist()
        for row in range(rows):
            atoms = slots.atom[starts[row] : starts[row] + counts[row]]
            for atom in atoms[atoms >= 0].tolist():
                name = str(names[atom])
                if name != "OXT" and name in rc.atom_order:
                    atom37[row, rc.atom_order[name]] = structure.coordinates[atom]
                    atom37_mask[row, rc.atom_order[name]] = observed[atom]
            if packed[row]:
                atom14[row] = atoms
                groups = rc.AMINO_ACID_ATOM14_TO_RIGID_GROUP[rc.AMINO_ACID_ONE_LETTER_CODES.index(_LETTER[ids[row]])]
                chi_index[row] = np.where(groups >= 4, groups - 4, -1)
                chi_mask[row, : len(rc.chi_angles_atoms[ids[row]])] = True
        packable = np.isin(np.array(ids), AMINO_ACID_COMPONENT_IDS)
        tokens = np.where(packable, layout.token_class, TOKEN_INDEX["X"]).astype(np.int64)
        instance = _owner(system.instance_component_ptr)[layout.token_component]
        chain_of = {value: index for index, value in enumerate(dict.fromkeys(instance.tolist()))}
        chain_index = np.array([chain_of[value] for value in instance.tolist()], dtype=np.int64)
        residue_index = layout.token_component - system.instance_component_ptr[instance]

        device, dtype = self.device, self.dtype
        backbone = torch.from_numpy(atom37[:, [0, 1, 2, 4]].copy())
        available = torch.from_numpy(valid.copy())
        backbone = project_model_coordinates(
            backbone,
            available[:, None] & torch.ones_like(backbone[..., 0], dtype=torch.bool),
            dtype=dtype,
            feature="packing backbone",
        ).to(device)
        size = self.config.num_context_atoms
        context_coordinates = torch.zeros(rows, size, 3, device=device, dtype=dtype)
        context_numbers = torch.zeros(rows, size, device=device, dtype=torch.int64)
        context_mask = torch.zeros(rows, size, device=device, dtype=torch.bool)
        consumed = np.zeros(0, dtype=np.int64)
        candidates = _context_atoms(system, layout, observed) if use_atom_context else np.zeros(0, dtype=np.int64)
        if len(candidates):
            points = torch.from_numpy(structure.coordinates[candidates])
            context = project_model_coordinates(
                points, torch.ones_like(points[..., 0], dtype=torch.bool), dtype=dtype, feature="packing context"
            ).to(device)
            numbers = torch.from_numpy(system.atomic_number[candidates].astype(np.int64)).to(device)
            n, ca, c = backbone[:, 0], backbone[:, 1], backbone[:, 2]
            squared = ((virtual_beta_carbon(n, ca, c)[:, None] - context[None]) ** 2).sum(dim=-1)
            squared = torch.where(available.to(device)[:, None], squared, torch.full_like(squared, 1000.0))
            k = min(size, len(candidates))
            nearest = torch.topk(squared, k, dim=-1, largest=False, sorted=True).indices
            context_coordinates[:, :k] = context[nearest]
            context_numbers[:, :k] = numbers[nearest]
            context_mask[:, :k] = True
            consumed = np.unique(candidates[nearest.cpu().numpy()[valid]])

        def tensor(values: NDArray, kind: torch.dtype = torch.int64) -> Tensor:
            return torch.as_tensor(values, dtype=kind, device=device).unsqueeze(0)

        mask37 = torch.from_numpy(atom37_mask)
        input = MpnnPackerInput(  # type: ignore[call-arg]
            backbone_coordinates=backbone.unsqueeze(0),
            sequence_tokens=tensor(tokens),
            residue_mask=tensor(valid, torch.bool),
            residue_index=tensor(residue_index),
            chain_index=tensor(chain_index),
            context_coordinates=context_coordinates.unsqueeze(0),
            context_atomic_numbers=context_numbers.unsqueeze(0),
            context_mask=context_mask.unsqueeze(0),
            atom37_coordinates=project_model_coordinates(
                torch.from_numpy(atom37), mask37, dtype=dtype, feature="packing atom37 observations"
            )
            .to(device)
            .unsqueeze(0),
            atom37_mask=mask37.to(device).unsqueeze(0),
            pack_mask=tensor(packed, torch.bool),
            atom14_atom_index=tensor(atom14),
            atom14_chi_index=tensor(chi_index),
            chi_mask=tensor(chi_mask, torch.bool),
            component_index=tensor(layout.token_component),
            batch_size=[],
        )
        return input, consumed


def _context_atoms(system: MolecularSystem, layout: TokenLayout, observed: NDArray[np.bool_]) -> NDArray[np.int64]:
    r"""The placed heavy atoms of every component that is not a packer row, outside water entities, in atom order."""
    s = system.arrays()
    atom_component = _owner(s["component_atom_ptr"])
    entity = s["instance_entity"][_owner(s["instance_component_ptr"])]
    water = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.WATER)
    row = np.zeros(len(entity), dtype=bool)
    row[layout.token_component] = True
    keep = observed & ~(row | water)[atom_component] & (s["atomic_number"] != 1)
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
