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
RDKit as a pure view of a determined graph.

Build-time half, run only when the artifact is built (tools): the stereo arbitration (a CIP label that contradicts its
own conformer is checked against the component's InChI descriptor). Runtime half, over a ``MolecularSystem`` and never
on a read: the projection ``to_rdkit`` (it refuses every undetermined fact it would need), the one ETKDG embedding, the
constraint features of model preparation and the training symmetry candidates (searched by ``multimolecule.graph``).
Chiral tags come from the parity table by permutation; no CIP search. RDKit is imported inside functions only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ..data.molecular import AtomFate, BondKind
from ..data.stereo import StereoGroupKind, permutation_parity
from ..errors import CODE_NAMES, InsufficientInformationError, StructureRef, UnsupportedError, fail
from ..provenance import Records, SourceRefs, SubjectKind, known
from .definitions import StereoBasis

if TYPE_CHECKING:
    from ..data.molecular import MolecularSystem
    from .definitions import ChemicalComponent


@dataclass(frozen=True)
class StereoArbitration:
    r"""
    The outcome of arbitrating one definition's stereocentre rows, aligned to ``component.stereo_centre``.

    ``inverted`` and ``refused`` name the atoms whose stored configuration is the other one, as the stated label and
    the InChI descriptor agree, or none; ``contradicted`` says the conformer's configuration disagrees with both at
    the inverted atoms. ``unverified`` names each labelled atom
    whose label could not be compared with its conformer, with the reason; ``bonds_disagreeing`` names the bonds whose
    stated E/Z label contradicts the conformer's (counted, never acted on).
    """

    parity: NDArray[np.int8]
    neighbors: NDArray[np.int16]
    basis: NDArray[np.uint8]
    inverted: tuple[str, ...]
    refused: tuple[str, ...]
    contradicted: bool
    unverified: tuple[tuple[str, str], ...] = ()
    bonds_disagreeing: tuple[tuple[str, str], ...] = ()


def _chiral_tag(parity: int, slots: Sequence[int], centre: int, molecule):
    r"""A stored parity as an RDKit chiral tag. RDKit reads its tag over the atom's own bond order, with the position
    that holds no graph atom last; the two orders differ by one permutation, and an odd one inverts the tag."""

    from rdkit import Chem

    order = [bond.GetOtherAtomIdx(centre) for bond in molecule.GetAtomWithIdx(centre).GetBonds()]
    target = [*order, *([-1] * (4 - len(order)))]
    position = {value: index for index, value in enumerate(target)}
    odd = int(permutation_parity(np.array([position[slot] for slot in slots], dtype=np.int64)))
    anticlockwise = (parity == 1) == (odd == 0)
    return Chem.ChiralType.CHI_TETRAHEDRAL_CCW if anticlockwise else Chem.ChiralType.CHI_TETRAHEDRAL_CW


def _component_mol(component: ChemicalComponent, *, conformer: bool):
    r"""
    The definition as a sanitised RDKit molecule with every atom explicit, or None when the projection is refused (an
    unknown element, a charge or bond order not KNOWN, no atoms, or a molecule RDKit cannot sanitise), with the chiral
    tags of the nonzero parity rows and the cis/trans of the nonzero stereobond rows; ``conformer`` attaches the
    reference conformer.
    """

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    c = component
    if not c.atom_names or (c.element == 0).any() or not known(c.charge_state).all():
        return None
    if not known(c.bond_order_state).all():
        return None
    editable = Chem.RWMol()
    for element, isotope, charge in zip(c.element.tolist(), c.isotope.tolist(), c.charge.tolist()):
        atom = Chem.Atom(element)
        atom.SetIsotope(isotope)
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.QUADRUPLE}
    for (a, b), order in zip(c.bond_atoms.tolist(), c.bond_order.tolist()):
        editable.AddBond(a, b, types[order])
    molecule = editable.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except (RuntimeError, ValueError):
        return None
    if conformer:
        frame = Chem.Conformer(len(c.atom_names))
        for index, (x, y, z) in enumerate(c.reference.astype(np.float64).tolist()):
            frame.SetAtomPosition(index, Point3D(x, y, z))
        molecule.AddConformer(frame, assignId=True)
    for centre, parity, slots in zip(c.stereo_centre.tolist(), c.stereo_parity.tolist(), c.stereo_neighbors.tolist()):
        if parity:
            molecule.GetAtomWithIdx(centre).SetChiralTag(_chiral_tag(parity, slots, centre, molecule))
    for bond, value, (left, right) in zip(
        c.bond_stereo_bond.tolist(), c.bond_stereo_value.tolist(), c.bond_stereo_reference.tolist()
    ):
        if value:
            rdkit_bond = molecule.GetBondWithIdx(bond)
            rdkit_bond.SetStereoAtoms(left, right)
            rdkit_bond.SetStereo(Chem.BondStereo.STEREOCIS if value == 1 else Chem.BondStereo.STEREOTRANS)
    return molecule


def _layers(inchi: str, keep: bool) -> tuple[str, ...]:
    r"""The ``/t``, ``/m``, ``/s`` layers (``keep``) or the skeleton layers (all but t, m, s, b) before ``/i``."""
    body = inchi.split("/i", 1)[0].split("/")
    return tuple(layer for layer in body if (layer[:1] in "tms" if keep else layer[:1] not in "tmsb") and layer)


def arbitrate_stereo(component: ChemicalComponent, descriptors: Sequence[tuple[str, str, str]]) -> StereoArbitration:
    r"""
    The stereo arbitration of one component: its stated CIP labels against its conformer and InChI descriptor.

    1. A labelled row with a nonzero parity is comparable; a labelled atom without one is unverified.
    2. The conformer is perceived in 3-D (``AssignStereochemistryFrom3D``, then exact CIP labels). A comparable atom
       is unverified when the perceived tag differs from the tag of its parity row, when CIP gives no label or a
       pseudo-asymmetric one; it is disputed when its CIP label differs from the stated one.
    3. Disputed centres are decided together by the component's first InChI descriptor, the one statement compared
       that is not read off the conformer: the InChI of the perceived molecule ("conformer reading") and of the same
       molecule with every disputed centre inverted ("label reading") are compared with it by stereo layers, after the
       skeletons agree. Only the label reading matches: the parities are inverted (DESCRIPTOR_VERIFIED) and the
       conformer is contradicted; only the conformer reading matches: the label is stale and the rows stand on the
       conformer (CONFORMER: the descriptor may itself be generated from these coordinates, so this agreement confirms
       nothing more); anything else (no InChI, no stereo layer, another skeleton, both or neither reading matching)
       is insufficient evidence and refuses the disputed centres (parity 0, REFUSED). The two readings are never
       compared with each other.

    The step checks the stereo configuration against an existing descriptor; it does not prove the 3-D conformer.
    """

    from rdkit import Chem, rdBase
    from rdkit.Chem import inchi as rdinchi
    from rdkit.Chem import rdCIPLabeler

    names, c = component.atom_names, component
    parity, basis = c.stereo_parity.copy(), c.stereo_basis.copy()
    rows = {centre: r for r, centre in enumerate(c.stereo_centre.tolist())}
    labelled = [(atom, label) for atom, label in enumerate(c.cip_declared) if label and atom in rows]
    unverified = [(names[a], "no_conformer_parity") for a, _ in labelled if not parity[rows[a]]]
    comparable = [(a, label) for a, label in labelled if parity[rows[a]]]

    def result(inverted=(), refused=(), contradicted=False, bonds=()) -> StereoArbitration:
        return StereoArbitration(
            parity, c.stereo_neighbors.copy(), basis, inverted, refused, contradicted, tuple(unverified), bonds
        )

    blocker = rdBase.BlockLogs()  # noqa: F841 — RDKit's perception warnings are not results; silenced while alive
    needed = comparable or (any(c.bond_stereo_declared) and c.reference_source)
    molecule = _component_mol(c, conformer=True) if needed else None
    if molecule is not None:
        projected = {a: molecule.GetAtomWithIdx(a).GetChiralTag() for a, _ in comparable}
        try:
            Chem.AssignStereochemistryFrom3D(molecule)
            rdCIPLabeler.AssignCIPLabels(molecule, maxRecursiveIterations=10_000_000)
        except (RuntimeError, ValueError):
            molecule = None
    if molecule is None:
        unverified += [(names[a], "projection_failed") for a, _ in comparable]
        return result()
    disputed = []
    for a, label in comparable:
        atom = molecule.GetAtomWithIdx(a)
        derived = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None
        if atom.GetChiralTag() != projected[a]:
            unverified.append((names[a], "derivation_differs"))
        elif derived is None:
            unverified.append((names[a], "no_cip_label"))
        elif not derived.isupper():
            unverified.append((names[a], "pseudoasymmetric_label"))
        elif derived != label:
            disputed.append(a)
    bonds = []
    for bond, bond_label in zip(c.bond_stereo_bond.tolist(), c.bond_stereo_declared):
        rdkit_bond = molecule.GetBondWithIdx(bond)
        derived = rdkit_bond.GetProp("_CIPCode") if rdkit_bond.HasProp("_CIPCode") else None
        if bond_label and derived in ("E", "Z") and derived != bond_label:
            bonds.append(tuple(names[x] for x in c.bond_atoms[bond].tolist()))
    if not disputed:
        return result(bonds=tuple(bonds))
    stated = next((text.strip() for kind, _, text in descriptors if kind.strip() == "InChI"), None)
    readings = None
    if stated is not None and stated.startswith("InChI=") and _layers(stated, True):
        try:
            molecule.RemoveAllConformers()  # InChI reads a conformer before the tags
            from_conformer = rdinchi.MolToInchi(molecule)
            for a in disputed:
                molecule.GetAtomWithIdx(a).InvertChirality()
            from_label = rdinchi.MolToInchi(molecule)
        except (RuntimeError, ValueError):
            from_conformer = from_label = ""
        if from_conformer and from_label and _layers(from_conformer, False) == _layers(stated, False):
            readings = (
                _layers(from_conformer, True) == _layers(stated, True),
                _layers(from_label, True) == _layers(stated, True),
            )
    disputed_rows = [rows[a] for a in disputed]
    if readings == (True, False):
        return result(bonds=tuple(bonds))
    if readings == (False, True):
        parity[disputed_rows], basis[disputed_rows] = 3 - parity[disputed_rows], StereoBasis.DESCRIPTOR_VERIFIED
        return result(tuple(names[a] for a in disputed), (), True, tuple(bonds))
    parity[disputed_rows], basis[disputed_rows] = 0, StereoBasis.REFUSED
    return result((), tuple(names[a] for a in disputed), False, tuple(bonds))


# Runtime views over a MolecularSystem. Never on a read.


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


def _incomplete(fact: str, **where: int) -> None:
    fail("rdkit_incomplete", StructureRef(item=fact, **where))


def _check_projection(system: MolecularSystem, atoms: NDArray[np.int64], local: NDArray[np.int64]) -> None:
    r"""Refuse a projection that would need an undetermined fact or cut a connection (``to_rdkit``)."""

    component = _owner(system.component_atom_ptr)[atoms]
    claims = system.component_topology_complete & system.component_heavy_atoms_complete
    claims = claims & system.component_hydrogens_complete
    for c in component[~claims[component]][:1].tolist():
        _incomplete("component_claims", component=c)
    instance = _owner(system.instance_component_ptr)[component]
    for i in instance[~system.instance_connections_complete[instance]][:1].tolist():
        _incomplete("instance_connections", instance=i)
    for name in ("atomic_number", "formal_charge", "implicit_hydrogens"):
        for a in atoms[~known(getattr(system, f"{name}_state")[atoms])][:1].tolist():
            _incomplete(name, atom=a)
    for a in atoms[system.atom_fate[atoms] == AtomFate.UNDETERMINED][:1].tolist():
        _incomplete("atom_fate", atom=a)
    first, second = system.edge_index.astype(np.int64)
    cut = (local[first] >= 0) != (local[second] >= 0)
    for e in np.flatnonzero(cut)[:1].tolist():
        fail("rdkit_unsupported", StructureRef(atom=int(first[e] if local[first[e]] >= 0 else second[e]), item="cut"))
    for x in np.flatnonzero(local[system.external_atom] >= 0)[:1].tolist():
        fail("rdkit_unsupported", StructureRef(external=x, item="external_connection"))
    inside = np.flatnonzero((local[first] >= 0) & (local[second] >= 0))
    covalent = system.bond_kind[inside] == BondKind.COVALENT
    undetermined = {
        "bond_kind": ~known(system.bond_kind_state[inside]),
        "bond_order": covalent & ~known(system.bond_order_state[inside]),
        "bond_direction": ~covalent
        & ~(known(system.bond_direction_state[inside]) & (system.bond_direction[inside] != 0)),
    }
    for fact, mask in undetermined.items():
        for e in inside[mask][:1].tolist():
            _incomplete(fact, atom=int(first[e]))


def to_rdkit(system: MolecularSystem, atoms: NDArray[np.int64] | None = None):
    r"""
    The selected atoms (all by default) as a sanitised RDKit molecule whose atom ``k`` is ``atoms[k]``: hydrogens
    that are not graph atoms as explicit counts, the KNOWN Kekulé orders (aromaticity perceived by RDKit), a
    coordination edge as a dative bond from its donor, chiral tags from the KNOWN parity rows by permutation, cis/trans
    from the KNOWN stereobond rows and the AND/OR groups over those parity rows. A configuration that is not KNOWN is
    an unset tag, as an unspecified one in RDKit; an isotope that is not KNOWN is RDKit's unspecified isotope.

    Raises:
        InsufficientInformationError: ``rdkit_incomplete`` for a needed fact that is not KNOWN (the item names it): a
            component claim, an instance's connections, an atom's element, charge, hydrogen count or existence, an
            edge's kind, order or donor.
        UnsupportedError: ``rdkit_unsupported`` for a selection that cuts an edge or an external connection, or a
            graph RDKit cannot sanitise.
    """

    from rdkit import Chem

    n = len(system.atom_symbol)
    atoms = np.arange(n, dtype=np.int64) if atoms is None else np.asarray(atoms, dtype=np.int64)
    local = np.full(n, -1, dtype=np.int64)
    local[atoms] = np.arange(len(atoms))
    _check_projection(system, atoms, local)

    editable = Chem.RWMol()
    for a in atoms.tolist():
        atom = Chem.Atom(int(system.atomic_number[a]))
        if system.isotope[a] > 0:
            atom.SetIsotope(int(system.isotope[a]))
        atom.SetFormalCharge(int(system.formal_charge[a]))
        atom.SetNumExplicitHs(int(system.implicit_hydrogens[a]))
        atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.QUADRUPLE}
    first, second = system.edge_index.astype(np.int64)
    inside = np.flatnonzero((local[first] >= 0) & (local[second] >= 0))
    for e in inside.tolist():
        a, b = int(local[first[e]]), int(local[second[e]])
        if system.bond_kind[e] == BondKind.COVALENT:
            editable.AddBond(a, b, types[int(system.bond_order[e])])
        else:  # +1: first donates to second
            editable.AddBond(*((a, b) if system.bond_direction[e] > 0 else (b, a)), Chem.BondType.DATIVE)
    try:
        Chem.SanitizeMol(editable)
    except (RuntimeError, ValueError):
        fail("rdkit_unsupported", StructureRef(item="sanitization"))

    centres = system.stereocentre_atom.astype(np.int64)
    tagged = (local[centres] >= 0) & known(system.stereocentre_state)
    for r in np.flatnonzero(tagged).tolist():
        centre = int(local[centres[r]])
        slots = [int(local[s]) if s >= 0 else -1 for s in system.stereocentre_neighbors[r].tolist()]
        editable.GetAtomWithIdx(centre).SetChiralTag(
            _chiral_tag(int(system.stereocentre_parity[r]), slots, centre, editable)
        )
    edges = system.stereobond_edge.astype(np.int64)
    for r in np.flatnonzero((local[first[edges]] >= 0) & known(system.stereobond_state)).tolist():
        e = edges[r]
        bond = editable.GetBondBetweenAtoms(int(local[first[e]]), int(local[second[e]]))
        bond.SetStereoAtoms(*(int(local[x]) for x in system.stereobond_reference[r].tolist()))
        bond.SetStereo(Chem.BondStereo.STEREOCIS if system.stereobond_value[r] == 1 else Chem.BondStereo.STEREOTRANS)
    Chem.SetDoubleBondNeighborDirections(editable)  # RDKit's legacy stereo perception keeps cis/trans by directions
    kinds = {StereoGroupKind.AND: Chem.StereoGroupType.STEREO_AND, StereoGroupKind.OR: Chem.StereoGroupType.STEREO_OR}
    groups, pointer = [], system.stereogroup_ptr
    for g, kind in enumerate(system.stereogroup_kind.tolist()):
        rows = system.stereogroup_member[pointer[g] : pointer[g + 1]]
        members = local[centres[rows[tagged[rows]]]]
        if kind in kinds and len(members):
            groups.append(Chem.CreateStereoGroup(kinds[kind], editable, members.tolist()))
    if groups:
        editable.SetStereoGroups(groups)
    return editable.GetMol()


def molecular_weight(system: MolecularSystem, atoms: NDArray[np.int64]) -> float:
    r"""
    The average molecular weight of the selected atoms with their hydrogens (RDKit ``Descriptors.MolWt`` over
    ``to_rdkit``).

    Raises:
        InsufficientInformationError, UnsupportedError: the refusals of ``to_rdkit``.
    """

    from rdkit.Chem import Descriptors

    return float(Descriptors.MolWt(to_rdkit(system, atoms)))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class Conformer:
    r"""Coordinates of the embedded atoms, in the order they were given, and how they were made."""

    coordinates: NDArray[np.float64]
    seed: int
    method: Literal["etkdg_v3"]


@dataclass(frozen=True)
class EmbeddingFailed:
    r"""No conformer: the search was exhausted, or the projection was refused (the caller raises
    ``ResourceLimitError(embedding_failed)``)."""

    reason: Literal["exhausted", "projection_refused"]
    attempts: int


def embed_conformer(
    system: MolecularSystem, atoms: NDArray[np.int64], *, seed: int, attempts: int = 2
) -> Conformer | EmbeddingFailed:
    r"""
    The one ETKDG routine: the determined graph of ``atoms`` (``to_rdkit``) with its implicit hydrogens added,
    embedded by ETKDGv3 with ``seed`` (every attempt after the first from random coordinates) and UFF-optimised (1,000
    iterations), as the four model processors did. Needs no definition and no reference conformer.
    """

    from rdkit import Chem
    from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

    try:
        molecule = Chem.AddHs(to_rdkit(system, atoms))
    except (InsufficientInformationError, UnsupportedError):
        return EmbeddingFailed("projection_refused", 0)
    options = rdDistGeom.ETKDGv3()
    options.randomSeed = seed
    for attempt in range(attempts):
        options.useRandomCoords = attempt > 0
        conformer = rdDistGeom.EmbedMolecule(molecule, options)
        if conformer >= 0:
            break
    else:
        return EmbeddingFailed("exhausted", attempts)
    rdForceFieldHelpers.UFFOptimizeMolecule(molecule, confId=conformer, maxIters=1000)
    coordinates = molecule.GetConformer(conformer).GetPositions()[: len(atoms)]
    return Conformer(np.asarray(coordinates, dtype=np.float64), seed, "etkdg_v3")


@dataclass(frozen=True)
class ConstraintFeatures:
    r"""
    The constraint inputs of Boltz-1/2 and Protenix (columns; atom indices are system atom indices): every pair
    of selected atoms with its bond and angle (1-3) flags and RDKit distance bounds; chiral quadruples (three
    neighbours then the centre) and their sign; stereobond quadruples (reference, end, end, reference) and trans; planar
    sp² C=C sextuples; torsion quadruples with their six force constants and signs; linear triples at sp–sp triple
    bonds; planar improper quadruples at sp² C, N and O centres with three neighbours.
    """

    pairs: NDArray[np.int64]
    bond_pair: NDArray[np.bool_]
    angle_pair: NDArray[np.bool_]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]
    chiral: NDArray[np.int64]
    chiral_positive: NDArray[np.bool_]
    stereo_bond: NDArray[np.int64]
    stereo_trans: NDArray[np.bool_]
    planar: NDArray[np.int64]
    torsions: NDArray[np.int64]
    torsion_force_constant: NDArray[np.float64]
    torsion_sign: NDArray[np.int64]
    linear: NDArray[np.int64]
    planar_improper: NDArray[np.int64]


def _grouped(system: MolecularSystem) -> NDArray[np.bool_]:
    r"""Per stereocentre row: a member of an AND or OR group (its absolute configuration is not stated)."""

    group = np.repeat(system.stereogroup_kind, np.diff(system.stereogroup_ptr))
    grouped = np.zeros(len(system.stereocentre_atom), dtype=bool)
    grouped[system.stereogroup_member[group != StereoGroupKind.ABS]] = True
    return grouped


def constraint_features(
    system: MolecularSystem, atoms: NDArray[np.int64], *, default_bounds: bool = False
) -> ConstraintFeatures:
    r"""
    The constraint features of ``atoms`` through ``to_rdkit`` (``models/processing.py:705-789`` as it was): bounds from
    ``GetMoleculeBoundsMatrix`` (``set15bounds``, ``scaleVDW``, triangle smoothing; with ``default_bounds`` RDKit's
    defaults, no vdW scaling, as Protenix's geometry featuriser) of the molecule with its implicit hydrogens added;
    chiral rows from the parity table — per three slot atoms, positive iff ``@@`` agrees with an odd position of the
    omitted slot — skipping AND/OR members and rows that are not KNOWN; the KNOWN stereobond rows; planar rows from
    ``[C;X3;^2](*)(*)=[C;X3;^2](*)(*)``; torsions from RDKit's experimental torsion library (small rings included),
    then each four consecutive sp² atoms of a 4–6 ring whose central bond no torsion covers (force constants
    ``(0, 100, 0, 0, 0, 0)``, signs ``(1, -1, 1, 1, 1, 1)``); linear triples (neighbour, end, end) at non-aromatic
    triple bonds between sp atoms; three improper quadruples per sp² C, N or O with three neighbours (the torsion,
    linear and improper rules of Protenix's geometry featuriser). Rows naming an added hydrogen are dropped.

    Raises:
        InsufficientInformationError, UnsupportedError: the refusals of ``to_rdkit``.
    """

    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import GetExperimentalTorsions, GetMoleculeBoundsMatrix

    atoms = np.asarray(atoms, dtype=np.int64)
    molecule = Chem.AddHs(to_rdkit(system, atoms))
    k = len(atoms)
    if default_bounds:
        bounds = GetMoleculeBoundsMatrix(molecule)
    else:
        bounds = GetMoleculeBoundsMatrix(
            molecule, set15bounds=True, scaleVDW=True, doTriangleSmoothing=True, useMacrocycle14config=False
        )
    adjacency = np.zeros((k, k), dtype=bool)
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a < k and b < k:
            adjacency[a, b] = adjacency[b, a] = True
    angle = (adjacency.astype(np.int64) @ adjacency.astype(np.int64) > 0) & ~np.eye(k, dtype=bool)
    i, j = np.triu_indices(k, 1)

    selected = np.zeros(len(system.atom_symbol), dtype=bool)
    selected[atoms] = True
    chiral, positive = [], []
    stated = known(system.stereocentre_state) & ~_grouped(system)
    for r in np.flatnonzero(selected[system.stereocentre_atom] & stated).tolist():
        slots, centre = system.stereocentre_neighbors[r].tolist(), int(system.stereocentre_atom[r])
        for chosen in combinations(range(4), 3):
            if all(slots[p] >= 0 for p in chosen):
                omitted = next(p for p in range(4) if p not in chosen)
                chiral.append([*(slots[p] for p in chosen), centre])
                positive.append((system.stereocentre_parity[r] == 2) == (omitted % 2 == 1))
    first, second = system.edge_index.astype(np.int64)
    edges = system.stereobond_edge.astype(np.int64)
    rows = np.flatnonzero(selected[first[edges]] & known(system.stereobond_state))
    references = system.stereobond_reference[rows].astype(np.int64)
    stereo_bond = np.stack((references[:, 0], first[edges[rows]], second[edges[rows]], references[:, 1]), axis=1)

    pattern = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    planar = [match for match in molecule.GetSubstructMatches(pattern) if max(match) < k]
    torsions = [
        (list(t["atomIndices"]), list(t["V"]), list(t["signs"]))
        for t in GetExperimentalTorsions(molecule, useSmallRingTorsions=True)
    ]
    sp, sp2 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2
    covered = {frozenset(t[1:3]) for t, _, _ in torsions}
    Chem.GetSymmSSSR(molecule)
    for ring in molecule.GetRingInfo().AtomRings():
        if 3 < len(ring) < 7:
            for start in range(len(ring)):
                quadruple = [ring[(start + offset) % len(ring)] for offset in range(4)]
                central = frozenset(quadruple[1:3])
                if central not in covered and all(
                    molecule.GetAtomWithIdx(a).GetHybridization() == sp2 for a in quadruple
                ):
                    torsions.append((quadruple, [0.0, 100.0, 0.0, 0.0, 0.0, 0.0], [1, -1, 1, 1, 1, 1]))
                    covered.add(central)
    torsions = [t for t in torsions if max(t[0]) < k]
    linear = []
    for bond in molecule.GetBonds():
        begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
        if (
            bond.GetBondType() == Chem.BondType.TRIPLE
            and not (bond.GetIsAromatic() or begin.GetIsAromatic() or end.GetIsAromatic())
            and begin.GetHybridization() == sp
            and end.GetHybridization() == sp
        ):
            b, e = begin.GetIdx(), end.GetIdx()
            linear += [[n, b, e] for n in sorted(x.GetIdx() for x in begin.GetNeighbors()) if n != e]
            linear += [[b, e, n] for n in sorted(x.GetIdx() for x in end.GetNeighbors()) if n != b]
    improper = []
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() in ("C", "N", "O") and atom.GetHybridization() == sp2 and atom.GetDegree() == 3:
            first, second, third = sorted(x.GetIdx() for x in atom.GetNeighbors())
            centre = atom.GetIdx()
            improper += [[first, second, centre, third], [third, first, centre, second], [second, third, centre, first]]
    linear = [row for row in linear if max(row) < k]
    improper = [row for row in improper if max(row) < k]
    return ConstraintFeatures(
        pairs=np.stack((atoms[i], atoms[j]), axis=1),
        bond_pair=adjacency[i, j],
        angle_pair=angle[i, j],
        lower=np.asarray(bounds, dtype=np.float64)[j, i],
        upper=np.asarray(bounds, dtype=np.float64)[i, j],
        chiral=np.array(chiral, dtype=np.int64).reshape(-1, 4),
        chiral_positive=np.array(positive, dtype=bool),
        stereo_bond=stereo_bond,
        stereo_trans=system.stereobond_value[rows] == 2,
        planar=atoms[np.array(planar, dtype=np.int64).reshape(-1, 6)],
        torsions=atoms[np.array([t[0] for t in torsions], dtype=np.int64).reshape(-1, 4)],
        torsion_force_constant=np.array([t[1] for t in torsions], dtype=np.float64).reshape(-1, 6),
        torsion_sign=np.array([t[2] for t in torsions], dtype=np.int64).reshape(-1, 6),
        linear=atoms[np.array(linear, dtype=np.int64).reshape(-1, 3)],
        planar_improper=atoms[np.array(improper, dtype=np.int64).reshape(-1, 4)],
    )


class SymmetryKind(IntEnum):
    AUTOMORPHISM = 1
    RESONANCE = 2


#: The explicit bound of one automorphism search (one component, one labelling): more matches refuse.
SYMMETRY_LIMIT = 1000


@dataclass(frozen=True)
class SymmetryCandidates:
    r"""
    Per group (one component with more than the identity), its atoms and the permutations of them (columns):
    ``permutation_ptr`` is the CSR of each group's flattened permutations (``atoms`` width each, the identity first)
    and ``kind`` the ``SymmetryKind`` of each permutation in order.
    """

    component: NDArray[np.int64]
    atoms_ptr: NDArray[np.int64]
    atoms: NDArray[np.int64]
    permutation_ptr: NDArray[np.int64]
    permutation: NDArray[np.int64]
    kind: NDArray[np.uint8]


class _Graph:
    r"""The determined graph of a system as comparison labels: an atom or edge whose facts are not all KNOWN, an atom
    on a component boundary or an external connection, and the atoms of a stereo row that is not KNOWN or of an AND/OR
    group are pinned (labelled by their own index)."""

    def __init__(self, system: MolecularSystem, pinned: NDArray[np.bool_]) -> None:
        n = len(system.atom_symbol)
        self.system, self.component = system, _owner(system.component_atom_ptr)
        first, second = system.edge_index.astype(np.int64)
        self.neighbors: list[list[int]] = [[] for _ in range(n)]
        self.edge: dict[tuple[int, int], int] = {}
        for e, (a, b) in enumerate(zip(first.tolist(), second.tolist())):
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)
            self.edge[a, b] = self.edge[b, a] = e
        pinned = np.asarray(pinned, dtype=bool).copy()
        for name in ("atomic_number", "formal_charge", "implicit_hydrogens", "atom_aromatic"):
            pinned |= ~known(getattr(system, f"{name}_state"))
        pinned |= system.atom_fate == AtomFate.UNDETERMINED
        pinned[system.external_atom] = True
        boundary = self.component[first] != self.component[second]
        edge_known = known(system.bond_kind_state) & known(system.bond_aromatic_state)
        edge_known &= known(system.bond_order_state) | (system.bond_kind != BondKind.COVALENT)
        for mask in (boundary, ~edge_known):
            pinned[first[mask]] = pinned[second[mask]] = True
        centres, slots = system.stereocentre_atom.astype(np.int64), system.stereocentre_neighbors.astype(np.int64)
        for r in np.flatnonzero(~known(system.stereocentre_state) | _grouped(system)).tolist():
            pinned[[centres[r], *(s for s in slots[r].tolist() if s >= 0)]] = True
        edges, references = system.stereobond_edge.astype(np.int64), system.stereobond_reference.astype(np.int64)
        for r in np.flatnonzero(~known(system.stereobond_state)).tolist():
            pinned[[first[edges[r]], second[edges[r]], *(x for x in references[r].tolist() if x >= 0)]] = True
        self.pinned = pinned
        self.atom_label = [
            ("pinned", a) if pinned[a] else (int(z), int(i), int(q), int(h), bool(r))
            for a, (z, i, q, h, r) in enumerate(
                zip(
                    system.atomic_number.tolist(),
                    system.isotope.tolist(),
                    system.formal_charge.tolist(),
                    system.implicit_hydrogens.tolist(),
                    system.atom_aromatic.tolist(),
                )
            )
        ]
        # an aromatic bond's Kekulé order is one arbitrary assignment, not a fact a permutation must keep
        orders = np.where((system.bond_kind == BondKind.COVALENT) & ~system.bond_aromatic, system.bond_order, 0)
        self.edge_label = list(zip(system.bond_kind.tolist(), orders.tolist(), system.bond_aromatic.tolist()))
        self.centres = {int(centres[r]): r for r in np.flatnonzero(known(system.stereocentre_state)).tolist()}
        rows = np.flatnonzero(known(system.stereobond_state)).tolist()
        self.bonds = {frozenset((int(first[edges[r]]), int(second[edges[r]]))): r for r in rows}

    def stereo_kept(self, image: dict[int, int]) -> bool:
        r"""Whether a mapping (atoms absent from it fixed) carries every KNOWN parity and cis/trans row onto one with
        the same configuration."""

        s = self.system

        def to(atom: int) -> int:
            return image.get(atom, atom) if atom >= 0 else atom

        for centre, r in self.centres.items():
            if centre not in image:
                continue
            target = self.centres.get(image[centre])
            mapped = [to(x) for x in s.stereocentre_neighbors[r].tolist()]
            if target is None or sorted(mapped) != sorted(s.stereocentre_neighbors[target].tolist()):
                return False
            odd = permutation_parity(np.array([mapped.index(x) for x in s.stereocentre_neighbors[target].tolist()]))
            parity = int(s.stereocentre_parity[r])
            if (3 - parity if odd else parity) != s.stereocentre_parity[target]:
                return False
        first, second = s.edge_index.astype(np.int64)
        for ends, r in self.bonds.items():
            if not any(x in image for x in ends):
                continue
            e = s.stereobond_edge[r]
            a, b = to(int(first[e])), to(int(second[e]))
            target = self.bonds.get(frozenset((a, b)))
            if target is None:
                return False
            refs = [to(x) for x in s.stereobond_reference[r].tolist()]
            if a != first[s.stereobond_edge[target]]:
                refs = refs[::-1]
            flips = sum(x != y for x, y in zip(refs, s.stereobond_reference[target].tolist()))
            value = int(s.stereobond_value[r])
            if (3 - value if flips % 2 else value) != s.stereobond_value[target]:
                return False
        return True

    def resonance(self, atoms: range) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
        r"""
        The carboxylate, nitro and guanidinium motifs of a component whose atoms are all determined and unpinned
        (``models/equivalence.py:36-46`` guards): closed (terminal atoms of one heavy neighbour, hydrogen counts as the
        family requires), covalent KNOWN orders, no aromatic or stereo atom. Returns the family per terminal atom and
        per centre–terminal edge.
        """

        s, labels, edge_labels = self.system, {}, {}
        z, q, h = s.atomic_number, s.formal_charge, s.implicit_hydrogens
        heavy_degree = lambda x: sum(z[y] != 1 for y in self.neighbors[x])  # noqa: E731
        order = lambda x, y: int(s.bond_order[self.edge[x, y]])  # noqa: E731
        for centre in atoms:
            if z[centre] not in (6, 7) or self.pinned[centre] or s.atom_aromatic[centre] or centre in self.centres:
                continue
            heavy = [x for x in self.neighbors[centre] if z[x] != 1]
            terminal = [x for x in heavy if heavy_degree(x) == 1]
            oxygens, nitrogens = [x for x in terminal if z[x] == 8], [x for x in terminal if z[x] == 7]
            if z[centre] == 6 and q[centre] == 0 and len(oxygens) == 2 and len(heavy) in (2, 3):
                family, terminals, hydrogens, expected = "carboxylate", oxygens, 0, [(1, -1), (2, 0)]
            elif z[centre] == 7 and q[centre] == 1 and len(oxygens) == 2 and len(heavy) == 3:
                family, terminals, hydrogens, expected = "nitro", oxygens, 0, [(1, -1), (2, 0)]
            elif z[centre] == 6 and q[centre] == 0 and len(heavy) == 3 and all(z[x] == 7 for x in heavy):
                if len(nitrogens) < 2:
                    continue
                family, terminals, hydrogens = "guanidinium", nitrogens, 2
                expected = sorted([(2, 1)] + [(1, 0)] * (len(nitrogens) - 1))
            else:
                continue
            motif = [centre, *terminals]
            graph_hydrogens = [y for x in terminals for y in self.neighbors[x] if z[y] == 1]
            closed = all(
                not self.pinned[x]
                and not s.atom_aromatic[x]
                and x not in self.centres
                and self.component[x] == self.component[centre]
                for x in motif + graph_hydrogens
            )
            closed &= all(
                s.bond_kind[self.edge[x, y]] == BondKind.COVALENT
                and not s.bond_aromatic[self.edge[x, y]]
                and frozenset((x, y)) not in self.bonds
                for x in motif
                for y in self.neighbors[x]
            )
            closed &= all(q[x] == 0 and h[x] == 0 for x in graph_hydrogens)
            closed &= all(h[x] + sum(z[y] == 1 for y in self.neighbors[x]) == hydrogens for x in terminals)
            closed &= len({self.atom_label[x][1] for x in terminals}) == 1  # one isotope
            substituents = [x for x in heavy if x not in terminals]
            if family != "guanidinium":
                closed &= all(z[x] == 6 for x in substituents)
            else:
                closed &= all(q[x] == 0 for x in substituents)
            if not closed or sorted((order(centre, x), int(q[x])) for x in terminals) != expected:
                continue
            if h[centre] + sum(order(centre, x) for x in self.neighbors[centre]) != 4:
                continue
            if (family != "carboxylate" and h[centre]) or any(order(centre, x) != 1 for x in substituents):
                continue
            for x in terminals:
                labels[x] = family
                edge_labels[centre, x] = edge_labels[x, centre] = family
        return labels, edge_labels

    def automorphisms(self, atoms: range, labels: dict, edge_labels: dict, component: int) -> set[tuple[int, ...]]:
        r"""
        Every automorphism of the component's labelled graph that keeps the stereo rows, as images of ``atoms``, by
        ``multimolecule.graph.automorphisms`` with the pinned atoms fixed. Terminal hydrogens are folded into their
        parent's label, each by its own atom and bond label, and mapped in label order, so swaps among one parent's
        equal hydrogens are not enumerated. More than ``SYMMETRY_LIMIT`` matches → ``ResourceLimitError``.
        """

        from multimolecule.graph import ResourceLimitError, UndirectedGraph, automorphisms  # the one graph toolkit
        from multimolecule.graph.topology import torch as tensors  # its tensor type (no direct torch import)

        s, members = self.system, set(atoms)
        hydrogens: dict[int, list[int]] = {}
        for x in atoms:
            parent = self.neighbors[x][0] if len(self.neighbors[x]) == 1 else -1
            if s.atomic_number[x] == 1 and not self.pinned[x] and parent in members and s.atomic_number[parent] != 1:
                hydrogens.setdefault(parent, []).append(x)
        key = lambda h: (self.atom_label[h], self.edge_label[self.edge[h, self.neighbors[h][0]]])  # noqa: E731
        for group in hydrogens.values():
            group.sort(key=key)
        folded = {x for group in hydrogens.values() for x in group}
        retained = [x for x in atoms if x not in folded]
        node_labels = [
            (
                (labels.get(x), self.atom_label[x] if x not in labels else self.atom_label[x][:2]),
                tuple(key(h) for h in hydrogens.get(x, ())),
            )
            for x in retained
        ]
        bonds = [(x, y) for x in retained for y in self.neighbors[x] if x < y and y not in folded and y in members]
        graph = UndirectedGraph(
            tensors.tensor(bonds, dtype=tensors.long).reshape(-1, 2), nodes=tensors.tensor(retained, dtype=tensors.long)
        )
        try:
            matches = automorphisms(
                graph,
                node_labels=node_labels,
                edge_labels=[edge_labels.get(bond, self.edge_label[self.edge[bond]]) for bond in bonds],
                fixed_nodes=[x for x in retained if self.pinned[x]],
                max_mappings=SYMMETRY_LIMIT,
            )
        except ResourceLimitError as error:
            fail("symmetry_search_limit", StructureRef(component=component), resource=error.resource, limit=error.limit)
        found = set()
        for match in matches.tolist():
            image = dict(zip(retained, match))
            for parent, group in hydrogens.items():
                image.update(zip(group, hydrogens[image[parent]]))
            if self.stereo_kept(image):
                found.add(tuple(image[x] for x in atoms))
        return found


def symmetry_candidates(
    system: MolecularSystem, components: NDArray[np.int64], *, resonance: bool, pinned: NDArray[np.bool_]
) -> tuple[SymmetryCandidates, Records]:
    r"""
    The training symmetry of each component: automorphisms of the determined graph by
    ``multimolecule.graph.automorphisms``, preserving element, isotope, charge, hydrogen count, aromaticity, bond kind
    and order, and the parity and cis/trans rows; ``pinned`` atoms (external, unresolved, undetermined) and every
    atom ``_Graph`` pins stay fixed. With ``resonance``, the guarded carboxylate, nitro and
    guanidinium alternatives are added as RESONANCE permutations. A component whose topology, atoms or hydrogens are
    not complete gets none and a ``symmetry_unknown_incomplete`` record.

    Raises:
        ResourceLimitError: ``symmetry_search_limit`` when one search finds more than ``SYMMETRY_LIMIT`` matches.
    """

    components = np.asarray(components, dtype=np.int64)
    complete = system.component_topology_complete & system.component_heavy_atoms_complete
    complete = complete & system.component_hydrogens_complete
    graph = _Graph(system, pinned)
    pointer = system.component_atom_ptr
    groups = []
    for c in components[complete[components]].tolist():
        atoms = range(int(pointer[c]), int(pointer[c + 1]))
        identity = tuple(atoms)
        automorphisms = graph.automorphisms(atoms, {}, {}, c) - {identity}
        alternatives = set()
        if resonance:
            labels, edge_labels = graph.resonance(atoms)
            if labels:
                alternatives = graph.automorphisms(atoms, labels, edge_labels, c) - automorphisms - {identity}
        if automorphisms or alternatives:
            permutations = [identity, *sorted(automorphisms), *sorted(alternatives)]
            kinds = [SymmetryKind.AUTOMORPHISM] * (1 + len(automorphisms)) + [SymmetryKind.RESONANCE] * len(
                alternatives
            )
            groups.append((c, identity, permutations, kinds))
    sizes = [len(identity) for _, identity, _, _ in groups]
    candidates = SymmetryCandidates(
        component=np.array([c for c, _, _, _ in groups], dtype=np.int64),
        atoms_ptr=np.cumsum([0, *sizes], dtype=np.int64),
        atoms=np.array([x for _, identity, _, _ in groups for x in identity], dtype=np.int64),
        permutation_ptr=np.cumsum([0, *(len(identity) * len(p) for _, identity, p, _ in groups)], dtype=np.int64),
        permutation=np.array([x for _, _, p, _ in groups for row in p for x in row], dtype=np.int64),
        kind=np.array([k for _, _, _, kinds in groups for k in kinds], dtype=np.uint8),
    )
    return candidates, _component_records("symmetry_unknown_incomplete", components[~complete[components]])


def _component_records(code: str, components: NDArray[np.int64]) -> Records:
    r"""One record of a payload-free code per component (subject the component, no source reference)."""

    count = len(components)
    no_refs = SourceRefs(
        np.zeros(0, dtype=np.int16),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int16),
        (),
        (),
        None,
    )
    return Records(
        np.full(count, CODE_NAMES.index(code), dtype=np.uint16),
        np.full(count, SubjectKind.COMPONENT, dtype=np.uint8),
        np.asarray(components, dtype=np.int64),
        np.zeros(count + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        no_refs,
        np.zeros((count, 4), dtype=np.int64),
        np.zeros(count + 1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        (),
    )
