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
Boltz-1 input processing, written against the shared preparation contract.

Everything that produces the model's inputs is OFoldX's own (layout, gating, crop, MSA pairing, reference conformers,
constraint features); this module keeps only the Boltz-1 featurisation of those inputs, which matches upstream for
identical inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.rdkit import ConstraintFeatures, constraint_features
from ...data.crop import Cropper, crop_token_layout
from ...data.molecular import ELEMENTS, BondKind, EntityType, MolecularSystem, MonomerType
from ...data.structure import Structure
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, Encoding, TokenLayout, token_classes
from ...errors import RequestRef, fail
from ...msa import MSA, MsaCollection, MsaRows, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Records, known
from ..layout import LayoutTable, model_tokens, slot_map, token_frames, token_layout, token_representatives
from ..preparation import (
    GatingPolicy,
    PreparationContext,
    PreparationRecord,
    PreparedInput,
    RegenerationScope,
    SlotMap,
    component_records,
    gate,
    missing_reference,
    reference_conformers,
)
from ..processing import augment_reference_positions, project_model_coordinates, project_model_features
from ..targets import StructureTarget, SymmetryPolicy, TruthState, prepare_target
from ..tensors import to_tensors
from .config import Boltz1ProcessorConfig
from .model import Boltz1Input, Boltz1Pocket, Boltz1StructureTarget

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA
_NUCLEIC_FRAME = ("C1'", "C3'", "C4'")

BOLTZ1_LAYOUT = LayoutTable(
    family="boltz1",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="nonstandard",
    exclude_designated_leaving=True,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={_PROTEIN: ("N", "CA", "C"), _DNA: _NUCLEIC_FRAME, _RNA: _NUCLEIC_FRAME},
    representative={
        TOKEN_INDEX[token]: name
        for token, name in (("G", "CA"), ("RNA_A", "C4"), ("RNA_G", "C4"), ("DNA_A", "C4"), ("DNA_G", "C4"))
    },
    representative_default={_PROTEIN: "CB", _DNA: "C2", _RNA: "C2"},
)
BOLTZ1_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
BOLTZ1_SYMMETRY = SymmetryPolicy(components="all", resonance=True, pin_unobserved=False)
#: The family's MSA depth constant (upstream ``max_msa_seqs``): the resource cap passed to ``pair_msas``.
BOLTZ1_MSA_DEPTH = 8192

# Boltz-1 pocket one-hot classes: UNSPECIFIED, UNSELECTED, POCKET, and BINDER.
_POCKET_UNSPECIFIED, _POCKET_UNSELECTED, _POCKET_POCKET, _POCKET_BINDER, _POCKET_SIZE = 0, 1, 2, 3, 4
_NONPOLYMER = 3


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


class Boltz1Processor(Processor):
    r"""
    Prepare one structure for Boltz-1.

    Args:
        config: Input-processing configuration. Defaults to ``Boltz1ProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.

    Notes:
        Component classes, atom identities, and declared representative atoms determine the model features.
        Reference-conformer availability is independent of experimental observations; generated reference geometry
        does not make an unobserved coordinate known.

        The token relation records positive chemical-edge presence and discards edge kind, role, order, and
        direction. It establishes neither covalence nor known absence. Reference-conformer operations that cannot
        represent directed coordination report an unsupported capability.
    """

    config_class = Boltz1ProcessorConfig
    config: Boltz1ProcessorConfig

    def __init__(
        self,
        config: Boltz1ProcessorConfig | None = None,
        *,
        seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.config = self.config_class() if config is None else self.config_class(config)
        if seed is not None:
            self.config.seed = seed
        super().__init__(device=device, dtype=dtype)

    @property
    def seed(self) -> int:
        r"""Seed used to embed missing reference conformers."""
        return self.config.seed

    @seed.setter
    def seed(self, seed: int) -> None:
        self.config.seed = seed

    def prepare_input(
        self,
        structure: Structure,
        *,
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        pocket: Boltz1Pocket | None = None,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[Boltz1Input]:
        r"""
        Prepare one structure: layout, optional training crop, slots, gating, reference conformers, constraint
        features, MSA rows and the Boltz-1 tensors. The scope is every atom in a new frame.

        Args:
            structure: The target; its chemistry is consumed as read (no library).
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            pocket: Optional binder and pocket-residue specification.
            cropper: Optional training crop policy; needs ``generator``.
            generator: The numpy generator of the crop and of the reference-position augmentation.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        layout = token_layout(system, BOLTZ1_LAYOUT)
        if cropper is not None:
            required = _pocket_required(system, layout, pocket)
            layout = crop_token_layout(structure, layout, cropper, required=required, generator=generator)
        slots = slot_map(system, layout, BOLTZ1_LAYOUT)
        accepted = gate(structure, slots, BOLTZ1_GATING)
        references, sources, embedded = reference_conformers(
            structure, slots, embed=missing_reference(system), seed=self.seed
        )
        constraints = [constraint_features(system, atoms) for atoms in _constraint_scopes(system, slots)]
        tokens, losses = model_tokens(system, layout)
        rows = _msa_rows(msa, system, layout, cropped=cropper is not None)
        torch_generator = None if generator is None else torch.Generator().manual_seed(int(generator.integers(2**62)))
        features = _features(
            structure,
            slots,
            tokens,
            references,
            constraints,
            rows,
            pocket,
            generator=torch_generator,
            device=self.device,
            dtype=self.dtype,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        bound = slots.atom >= 0
        conformers[_owner(system.component_atom_ptr)[slots.atom[bound]]] = sources[bound]
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.all(system),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    conformers,
                ),
                spec=self.preparation_spec,
            ),
        )

    def prepare_target(
        self,
        target: Structure,
        *,
        prepared: PreparedInput[Boltz1Input],
    ) -> Boltz1StructureTarget:
        r"""Align coordinate truth to an existing prepared input.

        Args:
            target: Independent truth in the task's Cartesian frame.
            prepared: Input features and correspondence returned by `prepare_input`.

        Returns:
            The family's coordinate target. Measured coordinates retain Float64
            precision; input preparation, crop, and reference sampling are not repeated.
        """
        self._validate_prepared(prepared)
        truth = prepare_target(
            prepared,
            target,
            symmetry=BOLTZ1_SYMMETRY,
        )
        return self._target_tensors(prepared, truth)

    def _target_tensors(self, prepared: PreparedInput[Boltz1Input], target: StructureTarget) -> Boltz1StructureTarget:
        r"""The model's tensor target: canonical truth, chain spans, chain equivalence and slot-axis symmetry groups."""
        system = prepared.context.structure.system
        atom_ptr = system.component_atom_ptr[system.instance_component_ptr]
        observed = torch.from_numpy(target.truth == TruthState.OBSERVED)
        coordinates = torch.from_numpy(target.coordinates.copy())
        group_atoms, permutations = _symmetry_groups(prepared.context.slots, target)
        return Boltz1StructureTarget(  # type: ignore[call-arg]
            coordinates=project_model_coordinates(coordinates, observed, feature="Boltz1 target coordinates")
            .to(self.device)
            .unsqueeze(0),
            coordinate_mask=observed.to(self.device).unsqueeze(0),
            chain_span=torch.from_numpy(np.stack((atom_ptr[:-1], atom_ptr[1:]), axis=1)).to(self.device).unsqueeze(0),
            chain_equivalence_index=torch.from_numpy(system.instance_entity.astype(np.int64))
            .to(self.device)
            .unsqueeze(0),
            symmetry_atom_index=group_atoms.to(self.device).unsqueeze(0),
            symmetry_permutation=permutations.to(self.device).unsqueeze(0),
            batch_size=[1],
        )


def _symmetry_groups(slots: SlotMap, target: StructureTarget) -> tuple[Tensor, Tensor]:
    r"""The symmetry candidates on the slot axis: ``[G, W]`` slot indices and ``[G, P, W]`` permutations, identity
    first, ``-1`` padded; a candidate that moves a slot atom onto an atom without a slot is dropped."""
    candidates = target.symmetry
    slot_of = np.full(len(target.truth), -1, dtype=np.int64)
    slot_of[slots.atom[slots.atom >= 0]] = np.flatnonzero(slots.atom >= 0)
    groups: list[tuple[list[int], list[list[int]]]] = []
    for g in range(len(candidates.component)):
        atoms = candidates.atoms[candidates.atoms_ptr[g] : candidates.atoms_ptr[g + 1]]
        rows = candidates.permutation[candidates.permutation_ptr[g] : candidates.permutation_ptr[g + 1]]
        rows = rows.reshape(-1, len(atoms))
        visible = slot_of[atoms] >= 0
        if visible.sum() < 2:
            continue
        identity = slot_of[atoms[visible]].tolist()
        found = [identity]
        for row in rows:
            mapped = row[visible]
            if set(mapped.tolist()) == set(atoms[visible].tolist()):
                candidate = slot_of[mapped].tolist()
                if candidate not in found:
                    found.append(candidate)
        if len(found) > 1:
            groups.append((identity, found))
    width = max((len(identity) for identity, _ in groups), default=0)
    count = max((len(found) for _, found in groups), default=0)
    group_atoms = torch.full((len(groups), width), -1, dtype=torch.long)
    permutations = torch.full((len(groups), count, width), -1, dtype=torch.long)
    for g, (identity, found) in enumerate(groups):
        group_atoms[g, : len(identity)] = torch.tensor(identity)
        permutations[g, : len(found), : len(identity)] = torch.tensor(found)
    return group_atoms, permutations


def _pocket_required(system: MolecularSystem, layout: TokenLayout, pocket: Boltz1Pocket | None) -> NDArray[np.bool_]:
    r"""The tokens a crop must keep: every token of the binder instance and of the pocket residues."""
    required = np.zeros(len(layout.token_component), dtype=bool)
    if pocket is None:
        return required
    binder, components = _pocket_indices(system, pocket)
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    return (token_instance == binder) | np.isin(layout.token_component, components)


def _pocket_indices(system: MolecularSystem, pocket: Boltz1Pocket) -> tuple[int, NDArray[np.int64]]:
    binder = int(system.instance_index([pocket.binder])[0])
    if binder < 0:
        fail("request_invalid_positions", RequestRef("pocket.binder", pocket.binder))
    components = system.component_index(list(pocket.pocket))
    if (components < 0).any():
        missing = pocket.pocket[int(np.flatnonzero(components < 0)[0])]
        fail("request_invalid_positions", RequestRef("pocket.pocket", str(missing)))
    return binder, components


def _pocket_feature(system: MolecularSystem, layout: TokenLayout, pocket: Boltz1Pocket | None) -> Tensor:
    r"""A ``(N, 4)`` one-hot: every token of the binder instance BINDER, the named pocket residues POCKET, the rest
    UNSELECTED; ``pocket=None`` gives every token UNSPECIFIED."""
    count = len(layout.token_component)
    labels = np.full(count, _POCKET_UNSPECIFIED, dtype=np.int64)
    if pocket is not None:
        binder, components = _pocket_indices(system, pocket)
        absent = components[~np.isin(components, layout.token_component)]
        if len(absent):
            fail("request_invalid_positions", RequestRef("pocket.pocket", str(system.component_refs(absent[:1])[0])))
        token_instance = _owner(system.instance_component_ptr)[layout.token_component]
        labels[:] = _POCKET_UNSELECTED
        labels[np.isin(layout.token_component, components)] = _POCKET_POCKET
        labels[token_instance == binder] = _POCKET_BINDER
    return torch.nn.functional.one_hot(torch.from_numpy(labels), _POCKET_SIZE).float()


def _msa_rows(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout, *, cropped: bool
) -> MsaRows | None:
    r"""Pools are paired by the one OFoldX rule after the crop; an exact plan is used as given."""
    if msa is None:
        return None
    plan: MsaCollection | None
    if isinstance(msa, MsaCollection):
        if cropped:
            fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
        if msa.row_pairing.shape[0] > BOLTZ1_MSA_DEPTH:
            fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=BOLTZ1_MSA_DEPTH)
        plan = msa
    else:
        plan = pair_msas(msa, system, layout, max_rows=BOLTZ1_MSA_DEPTH)
    return None if plan is None else msa_rows(plan, system, layout)


def _constraint_scopes(system: MolecularSystem, slots: SlotMap) -> list[NDArray[np.int64]]:
    r"""
    The atom sets Boltz-1 derives steering constraints for: each non-polymer instance, and each non-standard polymer
    component other than MSE (standard residues and MSE are constraint-free upstream), that holds a slot atom and
    projects to RDKit on its own — every claim True, connections complete, no edge or external connection leaving it.
    """

    s = system.arrays()
    count, instances = len(s["component_symbol"]), len(s["instance_symbol"])
    atom_component = _owner(s["component_atom_ptr"])
    component_instance = _owner(s["instance_component_ptr"])
    entity = s["instance_entity"][component_instance]
    polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    standard = token_classes(system).encoding == Encoding.EXACT
    standard |= system.text("component", s["component_symbol"]) == "MSE"
    unit = np.where(polymer, np.arange(count), count + component_instance)
    complete = s["component_topology_complete"] & s["component_heavy_atoms_complete"]
    complete &= system.component_hydrogens_complete & s["instance_connections_complete"][component_instance]
    atom_unit = unit[atom_component]
    first, second = s["edge_index"].astype(np.int64)
    crossing = atom_unit[first] != atom_unit[second]
    leaving = np.zeros(count + instances, dtype=bool)
    leaving[atom_unit[first[crossing]]] = leaving[atom_unit[second[crossing]]] = True
    leaving[atom_unit[s["external_atom"].astype(np.int64)]] = True
    blocked = np.zeros(count + instances, dtype=bool)
    blocked[unit[~complete | (polymer & standard)]] = True
    used = np.zeros(count + instances, dtype=bool)
    used[atom_unit[slots.atom[slots.atom >= 0]]] = True
    scopes = np.flatnonzero(used & ~leaving & ~blocked)
    return [np.flatnonzero(atom_unit == u) for u in scopes.tolist()]


def _atom_names(system: MolecularSystem, atoms: NDArray[np.int64]) -> list[str]:
    r"""Atom names in four characters; a longer or non-printable name becomes its element and a per-instance count."""
    names = system.text("atom", system.atom_symbol[atoms]).tolist()
    instance = _owner(system.instance_component_ptr)[_owner(system.component_atom_ptr)[atoms]].tolist()
    elements = system.atomic_number[atoms].tolist()
    counts: dict[tuple[int, str], int] = {}
    encoded = []
    for name, owner, number in zip(names, instance, elements):
        if len(name) > 4 or any(not 32 <= ord(character) < 96 for character in name):
            element = ELEMENTS[number].upper()
            counts[owner, element] = counts.get((owner, element), 0) + 1
            name = f"{element}{counts[owner, element]}"
        encoded.append(name)
    return encoded


def _cyclic_periods(system: MolecularSystem) -> NDArray[np.int64]:
    r"""Per instance, its component count when a KNOWN covalent edge joins its first component's previous port to its
    last component's next port (a head-to-tail cyclic polymer), else 0."""
    s = system.arrays()
    pointer = s["instance_component_ptr"]
    periods = np.zeros(len(pointer) - 1, dtype=np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    pairs = {tuple(edge) for edge in s["edge_index"][:, covalent].T.tolist()}
    for instance, (start, stop) in enumerate(zip(pointer[:-1].tolist(), pointer[1:].tolist())):
        if stop - start < 2:
            continue
        head, tail = int(s["component_port"][start, 0]), int(s["component_port"][stop - 1, 1])
        if head >= 0 and tail >= 0 and (min(head, tail), max(head, tail)) in pairs:
            periods[instance] = stop - start
    return periods


def _steering(
    system: MolecularSystem, slots: SlotMap, constraints: list[ConstraintFeatures], dtype: torch.dtype
) -> dict[str, Tensor]:
    r"""Constraint features on the slot axis, covalent connections between instances and same-entity chain pairs."""
    s = system.arrays()
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[slots.atom[slots.atom >= 0]] = np.flatnonzero(slots.atom >= 0)

    def rows(index: NDArray[np.int64], *flags: NDArray) -> tuple[Tensor, ...]:
        mapped = slot_of[index]
        keep = (mapped >= 0).all(axis=1)
        return (torch.from_numpy(mapped[keep].T.copy()), *(torch.from_numpy(flag[keep].copy()) for flag in flags))

    def joined(name: str, width: int) -> NDArray:
        parts = [getattr(c, name) for c in constraints]
        return np.concatenate(parts) if parts else np.zeros((0, width) if width else 0)

    pairs, bond, angle, lower, upper = rows(
        joined("pairs", 2).astype(np.int64),
        joined("bond_pair", 0).astype(bool),
        joined("angle_pair", 0).astype(bool),
        joined("lower", 0).astype(np.float64),
        joined("upper", 0).astype(np.float64),
    )
    chiral, positive = rows(joined("chiral", 4).astype(np.int64), joined("chiral_positive", 0).astype(bool))
    stereo, trans = rows(joined("stereo_bond", 4).astype(np.int64), joined("stereo_trans", 0).astype(bool))
    (planar,) = rows(joined("planar", 6).astype(np.int64))
    component_instance = _owner(s["instance_component_ptr"])
    atom_instance = component_instance[_owner(s["component_atom_ptr"])]
    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    between = covalent & (atom_instance[first] != atom_instance[second])
    between &= (slot_of[first] >= 0) & (slot_of[second] >= 0)
    connected_chains = np.stack((atom_instance[first[between]], atom_instance[second[between]]))
    connected_atoms = np.stack((slot_of[first[between]], slot_of[second[between]]))
    active = np.unique(atom_instance[slots.atom[slots.atom >= 0]])
    entity = s["instance_entity"]
    left, right = np.triu_indices(len(active), 1)
    same = entity[active[left]] == entity[active[right]]
    symmetric = np.stack((active[left][same], active[right][same]))
    return {
        "geometry_pair_index": pairs,
        "geometry_bond_mask": bond,
        "geometry_angle_mask": angle,
        "geometry_upper_bounds": upper.to(dtype),
        "geometry_lower_bounds": lower.to(dtype),
        "chiral_atom_index": chiral,
        "chiral_atom_positive": positive,
        "stereo_bond_index": stereo,
        "stereo_bond_trans": trans,
        "planar_bond_index": planar,
        "connected_chain_index": torch.from_numpy(connected_chains.astype(np.int64)),
        "connected_atom_index": torch.from_numpy(connected_atoms.astype(np.int64)),
        "symmetric_chain_index": torch.from_numpy(symmetric.astype(np.int64)),
    }


def _token_bonds(system: MolecularSystem, layout: TokenLayout) -> Tensor:
    r"""
    ``[N, N]``: 1 between two tokens a KNOWN covalent edge joins (upstream's per-atom component bonds and covalent
    connections), except polymer junctions — an edge from one component's next port to another's previous port in the
    same instance — which upstream never lists.
    """
    s = system.arrays()
    first, second = s["edge_index"].astype(np.int64)
    component = _owner(s["component_atom_ptr"])
    left, right = component[first], component[second]
    port = s["component_port"].astype(np.int64)
    junction = ((port[left, 1] == first) & (port[right, 0] == second)) | (
        (port[left, 0] == first) & (port[right, 1] == second)
    )
    instance = _owner(s["instance_component_ptr"])
    junction &= (left != right) & (instance[left] == instance[right])
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = covalent & ~junction & (a >= 0) & (b >= 0) & (a != b)
    bonds = torch.zeros(len(layout.token_component), len(layout.token_component))
    bonds[torch.from_numpy(a[linked]), torch.from_numpy(b[linked])] = 1
    bonds[torch.from_numpy(b[linked]), torch.from_numpy(a[linked])] = 1
    return bonds


def _msa_features(rows: MsaRows | None, tokens: NDArray[np.int64]) -> dict[str, Tensor]:
    if rows is None:
        token = torch.from_numpy(tokens[None].copy())
        deletions = torch.zeros(1, len(tokens))
        paired = torch.ones(1, len(tokens))
    else:
        token = torch.from_numpy(rows.token.astype(np.int64))
        deletions = torch.from_numpy(rows.deletions.astype(np.float32))
        paired = torch.from_numpy(rows.paired.astype(np.float32))
    deletion_value = (math.pi / 2) * torch.atan(deletions / 3)
    return {
        "msa_tokens": token,
        "has_deletion": deletions > 0,
        "deletion_value": deletion_value,
        "msa_paired": paired,
        "msa_mask": torch.ones_like(token, dtype=torch.bool),
        "msa_profile": torch.nn.functional.one_hot(token, len(TOKENS)).float().mean(dim=0),
        "deletion_mean": deletion_value.mean(dim=0),
    }


def _features(
    structure: Structure,
    slots: SlotMap,
    tokens: NDArray[np.int64],
    references: NDArray[np.float64],
    constraints: list[ConstraintFeatures],
    rows: MsaRows | None,
    pocket: Boltz1Pocket | None,
    *,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Boltz1Input:
    r"""The Boltz-1 tensors of one prepared task (upstream featurisation over the slot axis)."""
    system = structure.system
    s = system.arrays()
    layout = slots.layout
    atom = slots.atom
    num_atoms, num_tokens = len(atom), len(layout.token_component)
    atom_component = _owner(s["component_atom_ptr"])
    component_instance = _owner(s["instance_component_ptr"])
    token_instance = component_instance[layout.token_component]
    entity = s["instance_entity"]
    instance_polymer = known(s["entity_type_state"][entity]) & (s["entity_type"][entity] == EntityType.POLYMER)
    residue_index = np.where(
        instance_polymer[token_instance], layout.token_component - s["instance_component_ptr"][token_instance], 0
    )
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    monomer = CLASS_MONOMER[layout.token_class]
    molecule_type = np.array(
        [BOLTZ1_LAYOUT.monomer_types.get(MonomerType(m), _NONPOLYMER) for m in monomer.tolist()], dtype=np.int64
    )
    slot_of = np.full(len(s["atom_symbol"]), -1, dtype=np.int64)
    slot_of[atom] = np.arange(num_atoms)

    reference_positions = torch.from_numpy(references.astype(np.float32))
    reference_space_id = torch.from_numpy(atom_component[atom])
    augment_reference_positions(
        reference_positions, reference_space_id, torch.ones(num_atoms, dtype=torch.bool), generator=generator
    )
    names = _atom_names(system, atom)
    characters = torch.tensor(
        [[ord(c) - 32 if c != " " else 0 for c in name.ljust(4)] for name in names], dtype=torch.long
    ).reshape(num_atoms, 4)
    canonical = to_tensors(system, ("atomic_number", "formal_charge"), device=torch.device("cpu"))
    atomic_number = canonical["atomic_number"][torch.from_numpy(atom)].long()
    charges = canonical["formal_charge"][torch.from_numpy(atom)].float()
    atom_mask = torch.ones(num_atoms, dtype=torch.bool)
    # Boltz-1 reference-atom features: positions (3), charge (1), mask (1), element (128), name characters (4x64).
    atom_features = torch.cat(
        (
            reference_positions,
            charges.unsqueeze(-1),
            atom_mask.unsqueeze(-1).float(),
            torch.nn.functional.one_hot(atomic_number, 128).float(),
            torch.nn.functional.one_hot(characters, 64).reshape(num_atoms, 256).float(),
        ),
        dim=-1,
    )
    token_bonds = _token_bonds(system, layout)
    tensors: dict[str, Tensor] = {
        "token_mask": torch.ones(num_tokens, dtype=torch.bool),
        "residue_index": torch.from_numpy(residue_index.astype(np.int64)),
        "chain_index": torch.from_numpy(token_instance),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64)),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "token_index": torch.arange(num_tokens),
        "atom_features": atom_features,
        "atom_to_token": torch.from_numpy(slots.token),
        "atom_mask": atom_mask,
        "reference_positions": reference_positions,
        "reference_space_id": reference_space_id,
        "atom_token_index": torch.from_numpy(np.arange(num_atoms) - layout.token_atom_ptr[slots.token]),
        "token_center_atom_index": torch.from_numpy(slot_of[layout.token_centre_atom]),
        "token_representative_atom_index": torch.from_numpy(
            slot_of[token_representatives(system, layout, BOLTZ1_LAYOUT)]
        ),
        "residue_type_features": torch.nn.functional.one_hot(torch.from_numpy(tokens), len(TOKENS)).float(),
        "token_bonds": token_bonds,
        "molecule_type": torch.from_numpy(molecule_type),
        "frame_atom_index": torch.from_numpy(slot_of[token_frames(system, layout, BOLTZ1_LAYOUT)]),
        "cyclic_period": torch.from_numpy(_cyclic_periods(system)[token_instance].astype(np.float32)),
        "pocket_feature": _pocket_feature(system, layout, pocket),
        **_steering(system, slots, constraints, dtype),
        **_msa_features(rows, tokens),
    }
    placed = {
        name: (
            project_model_features(
                value, torch.ones_like(value, dtype=torch.bool), dtype=dtype, feature=f"Boltz1 prepared {name}"
            ).to(device=device)
            if value.is_floating_point()
            else value.to(device=device)
        )
        for name, value in tensors.items()
    }
    return Boltz1Input(  # type: ignore[call-arg]
        **{name: value.unsqueeze(0) for name, value in placed.items()},
        atom_index=torch.from_numpy(atom.copy()).to(device).unsqueeze(0),
        batch_size=[],
    )
