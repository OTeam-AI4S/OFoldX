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
BoltzGen input processing, written against the shared preparation contract.

``plan`` materialises the design topology once: generated placements become UNK protein chains through
``chemistry.construction``, requested links become one declared edit, and every symbolic condition is resolved onto
``ComponentRef``\ s. ``prepare`` produces the model's tensors from the shared layout, gating, reference conformers,
MSA pairing (``ofoldx.msa``) and template resolution (``ofoldx.templates``); everything else here is the BoltzGen
featurisation, which matches upstream for identical inputs: the atom14 design rows with their synthetic
side-chain slots, the conditioning channels, the template channels and the MSA transform.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from multimolecule.graph import UndirectedGraph
from numpy.typing import NDArray
from torch import Tensor

from ...chemistry.construction import PolymerSequence, Recipe, build_system
from ...chemistry.definitions import ComponentLibrary, configured_library
from ...data.assembly import union
from ...data.crop import Cropper, crop_token_layout
from ...data.editing import ChemicalEdit, EdgeRows, EditReport, apply_edit
from ...data.molecular import (
    ELEMENTS,
    AtomSite,
    BondKind,
    BondOrigin,
    ComponentRef,
    EntityType,
    MolecularSystem,
    MonomerType,
)
from ...data.stereo import ParityClears
from ...data.structure import CoordinateEvidence, Structure, replace_coordinates
from ...data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, TokenLayout, token_classes
from ...errors import RequestRef, StructureRef, fail
from ...modules.geometry import rotation_from_two_vectors
from ...msa import MSA, MsaCollection, MsaRows, msa_rows, pair_msas
from ...processor import Processor
from ...provenance import Basis, Records, Resolution, known, state
from ...templates import ChainCoordinates, Template, resolve_template, template_alignment_frames, template_atoms
from ..conditions import (
    AllowedResidueIdentities,
    Endpoint,
    ExistingSpan,
    GeneratedSegment,
    Hotspot,
    PlacementMap,
    PlacementSpec,
    SymbolicAtomSite,
    SymbolicBond,
    SymbolicResidueSite,
    TiedResidueIdentities,
    placement_instance_of,
    resolve_bond_site,
    resolve_placement,
)
from ..design import PositionSelection, resolve_positions
from ..layout import LayoutTable, model_tokens, token_frames, token_layout
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
from ..targets import (
    SequenceTarget,
    StructureTarget,
    SymmetryPolicy,
    TruthState,
    align_target_components,
    prepare_sequence_target,
    prepare_target,
)
from .config import BoltzGenProcessorConfig
from .model import (
    CANONICAL_TOKENS,
    BoltzGenBinding,
    BoltzGenContact,
    BoltzGenDesignMask,
    BoltzGenInput,
    BoltzGenInverseFoldingInput,
    BoltzGenSchema,
    BoltzGenSecondaryStructure,
    BoltzGenStructureGroup,
    BoltzGenStructureTarget,
    BoltzGenSymmetry,
    normalize_boltzgen_input,
)

_PROTEIN, _RNA, _DNA = MonomerType.PROTEIN, MonomerType.RNA, MonomerType.DNA
_NUCLEIC_FRAME = ("C1'", "C3'", "C4'")

BOLTZGEN_LAYOUT = LayoutTable(
    family="boltzgen",
    monomer_types={_PROTEIN: 0, _DNA: 1, _RNA: 2},
    residue_slots=None,
    monomer_slots=None,
    hydrogens="exclude",
    per_atom_components="none",  # a modified residue is one residue token, as upstream (Boltz-2's rule)
    exclude_designated_leaving=True,
    centre_atom={_PROTEIN: "CA", _DNA: "C1'", _RNA: "C1'"},
    frame_atoms={_PROTEIN: ("N", "CA", "C"), _DNA: _NUCLEIC_FRAME, _RNA: _NUCLEIC_FRAME},
    representative={
        TOKEN_INDEX[token]: name
        for token, name in (
            ("G", "CA"),
            ("RNA_A", "C4"),
            ("RNA_G", "C4"),
            ("DNA_A", "C4"),
            ("DNA_G", "C4"),
            ("RNA_C", "C2"),
            ("RNA_U", "C2"),
            ("DNA_C", "C2"),
            ("DNA_T", "C2"),
        )
    },
    representative_default={_PROTEIN: "CB", _DNA: "C1'", _RNA: "C1'"},
)
#: BoltzGen reads each slot atom's element and formal charge in its reference-atom features.
BOLTZGEN_GATING = GatingPolicy(required_states=frozenset({"atomic_number", "formal_charge"}))
BOLTZGEN_SYMMETRY = SymmetryPolicy(components="all", resonance=True, pin_unobserved=False)
#: The family's MSA depth constant (the Boltz data pipeline's ``max_msa_seqs``): the cap passed to ``pair_msas``.
BOLTZGEN_MSA_DEPTH = 8192
#: The monomer types BoltzGen gives a template channel.
BOLTZGEN_TEMPLATE_TYPES = (_PROTEIN, _RNA, _DNA)
#: The atoms a designed backbone is addressed by in inverse folding.
BOLTZGEN_BACKBONE = ("N", "CA", "C", "O")

# Contact-conditioning one-hot classes: 0 UNSPECIFIED, 1 UNSELECTED, 2 POCKET>BINDER, 3 BINDER>POCKET, 4 CONTACT.
_NUM_CONTACT_CLASSES = 5
_CONTACT_KIND_TO_CLASS = {"pocket>binder": 2, "binder>pocket": 3, "contact": 4}
_BINDING_UNSPECIFIED, _BINDING_BINDING, _BINDING_NOT_BINDING = 0, 1, 2
_SS_UNSPECIFIED = 3  # num_ss_types == 4; class 3 is the masked label
_SS_LABEL_TO_ID = {"H": 0, "E": 1, "L": 2}
#: ``molecule_type`` of a token of no polymer class.
_NONPOLYMER = 3
#: The element of a synthetic side-chain slot and the prefix of its atom name.
_SYNTHETIC_ELEMENT, _SYNTHETIC_NAME = 114, "FL"
#: The one identity of a generated protein position.
_GENERATED_COMPONENT = "UNK"

#: Per designed residue, its physical atom14 rows in upstream order.
_ATOM14_NAMES = {
    "UNK": (*BOLTZGEN_BACKBONE, "CB"),
    "GLY": BOLTZGEN_BACKBONE,
    "ALA": ("N", "CA", "C", "O", "CB"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}
#: Per designed residue, the real atom whose coordinates train each of its synthetic slots (upstream's choice).
_ATOM14_SYNTHETIC = {
    "UNK": ("N",) * 9,
    "GLY": ("O",) * 10,
    "ALA": ("O",) * 9,
    "CYS": ("O",) * 8,
    "SER": ("N",) * 8,
    "PRO": ("O",) * 7,
    "THR": ("N", "N", "N", "O", "O", "O", "O"),
    "VAL": ("N",) * 7,
    "ASN": ("N", "O", "O", "O", "O", "O"),
    "ASP": ("N", "N", "O", "O", "O", "O"),
    "ILE": ("O",) * 6,
    "LEU": ("N", "N", "N", "N", "O", "O"),
    "MET": ("N",) * 6,
    "GLN": ("O",) * 5,
    "GLU": ("N", "N", "O", "O", "O"),
    "LYS": ("N",) * 5,
    "HIS": ("O",) * 4,
    "PHE": ("O",) * 3,
    "ARG": ("N",) * 3,
    "TYR": ("O", "O"),
    "TRP": (),
}
#: Upstream identity codes: counts of the ten trailing slots close to N, CA, C and O respectively.
_ATOM14_PLACEMENT_COUNTS = {
    (9, 0, 0, 0): "UNK",
    (0, 0, 0, 10): "GLY",
    (0, 0, 0, 9): "ALA",
    (0, 0, 0, 8): "CYS",
    (8, 0, 0, 0): "SER",
    (0, 0, 0, 7): "PRO",
    (3, 0, 0, 4): "THR",
    (7, 0, 0, 0): "VAL",
    (0, 0, 0, 6): "ILE",
    (1, 0, 0, 5): "ASN",
    (2, 0, 0, 4): "ASP",
    (4, 0, 0, 2): "LEU",
    (6, 0, 0, 0): "MET",
    (0, 0, 0, 5): "GLN",
    (2, 0, 0, 3): "GLU",
    (5, 0, 0, 0): "LYS",
    (0, 0, 0, 4): "HIS",
    (0, 0, 0, 3): "PHE",
    (3, 0, 0, 0): "ARG",
    (0, 0, 0, 2): "TYR",
    (0, 0, 0, 0): "TRP",
}


def decode_boltzgen_atom14(coordinates: Tensor) -> tuple[tuple[str, ...], tuple[bool, ...]]:
    r"""Decode upstream atom14 residue identities and whether each marker signature was recognized.

    Each row contains N, CA, C, O and ten remaining predicted slots. A slot
    at most 0.5 angstrom from a backbone atom votes for its nearest atom; ties
    use N, CA, C, O order. An unrecognized count decodes to ``UNK`` with a
    false recognition flag, distinct from the recognized UNK signature.
    """
    if not coordinates.is_floating_point() or coordinates.ndim != 3 or coordinates.shape[-2:] != (14, 3):
        fail("value_invalid", RequestRef("atom14", str(tuple(coordinates.shape))))
    if not bool(torch.isfinite(coordinates).all()):
        fail("value_invalid", RequestRef("atom14", "nonfinite coordinates"))
    if coordinates.dtype in (torch.float16, torch.bfloat16):
        coordinates = coordinates.float()
    distance = torch.cdist(coordinates[:, :4], coordinates[:, 4:])
    value, nearest = distance.min(dim=1)
    nearest = nearest.masked_fill(value > 0.5, -1)
    counts = (nearest[..., None] == torch.arange(4, device=coordinates.device)).sum(dim=1)
    signatures = [tuple(row) for row in counts.detach().cpu().tolist()]
    return (
        tuple(_ATOM14_PLACEMENT_COUNTS.get(signature, "UNK") for signature in signatures),
        tuple(signature in _ATOM14_PLACEMENT_COUNTS for signature in signatures),
    )


def boltzgen_atom14_names(residue: str) -> tuple[str, ...]:
    r"""Return the upstream real-atom slot names for a decoded residue."""
    try:
        return _ATOM14_NAMES[residue]
    except KeyError:
        fail("value_invalid", RequestRef("residue", residue))


#: The canonical classes BoltzGen inverse folding can design (its 20 trained residues).
_GENERATABLE = tuple(sorted(TOKEN_INDEX[letter] for letter in "ACDEFGHIKLMNPQRSTVWY"))
#: The canonical letter of each three-letter residue the inverse folder designs.
CANONICAL_LETTER: Mapping[str, str] = {name: letter for letter, name in zip("ACDEFGHIKLMNPQRSTVWY", CANONICAL_TOKENS)}
_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_ABSENT = state(Resolution.UNKNOWN, Basis.NONE)


def _owner(pointer: NDArray[np.int64]) -> NDArray[np.int64]:
    return np.repeat(np.arange(len(pointer) - 1, dtype=np.int64), np.diff(pointer))


@dataclass(frozen=True)
class _AtomRows:
    r"""
    The BoltzGen model-row axis of one token layout.

    A context token contributes its addressed atoms; a designed token of the structure designer contributes its
    atom14 rows, whose synthetic side-chain slots have no canonical atom (``atom`` ``-1``) and never become chemical
    atoms. ``reference`` is the atom whose reference conformer seeds each row (a synthetic row takes its CA).
    """

    atom: NDArray[np.int64]
    token: NDArray[np.int64]
    offset: NDArray[np.int64]
    reference: NDArray[np.int64]
    designed: NDArray[np.bool_]


# ===========================================================================
# The shared layout: designed rows, slot map and model-row features.
# ===========================================================================


def _designed_components(system: MolecularSystem, conditions: Sequence[BoltzGenSchema]) -> NDArray[np.int64]:
    r"""The components a design mask names, sorted."""
    positions = [position for c in conditions if isinstance(c, BoltzGenDesignMask) for position in c.positions]
    if not positions:
        return np.zeros(0, dtype=np.int64)
    found = system.component_index(positions)
    if (found < 0).any():
        fail("request_invalid_positions", RequestRef("design_mask", str(positions[int(np.flatnonzero(found < 0)[0])])))
    return np.unique(found)


def _atom14_table(system: MolecularSystem, components: NDArray[np.int64]) -> dict[int, str]:
    r"""Per designed component, the residue whose atom14 rows address it; another identity is ``model_capacity``."""
    ids = system.text("component", system.component_symbol)
    monomer = CLASS_MONOMER[token_classes(system).token]
    table = {}
    for component in components.tolist():
        name = str(ids[component])
        if monomer[component] != _PROTEIN or name not in _ATOM14_NAMES:
            fail("model_capacity", StructureRef(component=component), identity=name)
        table[component] = name
    return table


def _addressed_atoms(system: MolecularSystem, names_of: Mapping[int, tuple[str, ...]]) -> NDArray[np.bool_]:
    r"""Every atom, except that a component of ``names_of`` is addressed only at the atoms it names."""
    names = system.text("atom", system.atom_symbol)
    atom_component = _owner(system.component_atom_ptr)
    visible = np.ones(len(names), dtype=bool)
    for component, allowed in names_of.items():
        atoms = np.flatnonzero(atom_component == component)
        visible[atoms] = np.isin(names[atoms], allowed)
    return visible


def _atom_rows(system: MolecularSystem, layout: TokenLayout, atom14: Mapping[int, str]) -> _AtomRows:
    r"""The model rows of ``layout``: a designed token's atom14 rows, every other token's addressed atoms."""
    names = system.text("atom", system.atom_symbol)
    atom, token, offset, reference, designed = [], [], [], [], []
    for index, component in enumerate(layout.token_component.tolist()):
        held = layout.token_atoms[layout.token_atom_ptr[index] : layout.token_atom_ptr[index + 1]].tolist()
        residue = atom14.get(component)
        if residue is None:
            atom += held
            reference += held
            designed += [False] * len(held)
            offset += list(range(len(held)))
            token += [index] * len(held)
            continue
        by_name = {str(names[a]): a for a in held}
        missing = [
            name for name in _ATOM14_NAMES[residue] if name not in by_name and not (residue == "UNK" and name == "CB")
        ]
        if missing:
            fail("model_capacity", StructureRef(component=component), missing=tuple(missing))
        real = [by_name.get(name, -1) for name in _ATOM14_NAMES[residue]]
        fillers = len(_ATOM14_SYNTHETIC[residue])
        atom += real + [-1] * fillers
        reference += [value if value >= 0 else by_name["CA"] for value in real] + [by_name["CA"]] * fillers
        designed += [True] * (len(real) + fillers)
        offset += list(range(len(real) + fillers))
        token += [index] * (len(real) + fillers)
    return _AtomRows(
        atom=np.array(atom, dtype=np.int64),
        token=np.array(token, dtype=np.int64),
        offset=np.array(offset, dtype=np.int64),
        reference=np.array(reference, dtype=np.int64),
        designed=np.array(designed, dtype=bool),
    )


def _slots(system: MolecularSystem, layout: TokenLayout, rows: _AtomRows) -> SlotMap:
    addressed = np.zeros(len(system.atom_symbol), dtype=bool)
    addressed[rows.atom[rows.atom >= 0]] = True
    return SlotMap(
        atom=rows.atom, token=rows.token, layout=layout, unaddressed=np.flatnonzero(~addressed).astype(np.int64)
    )


def _truth_source(system: MolecularSystem, slots: SlotMap) -> NDArray[np.int64]:
    r"""Per model row, the canonical atom whose observed coordinates it carries (a synthetic row its filler)."""
    layout, source = slots.layout, slots.atom.copy()
    if (source >= 0).all():
        return source
    names = system.text("atom", system.atom_symbol)
    ids = system.text("component", system.component_symbol)
    for token in np.unique(slots.token[source < 0]).tolist():
        rows = np.flatnonzero(slots.token == token)
        residue = str(ids[layout.token_component[token]])
        by_name = {str(names[a]): int(a) for a in source[rows] if a >= 0}
        real = len(_ATOM14_NAMES[residue])
        for offset, filler in enumerate(_ATOM14_SYNTHETIC[residue]):
            source[rows[real + offset]] = by_name[filler]
    return source


def _require_residue_tokens(system: MolecularSystem, layout: TokenLayout) -> None:
    r"""Every polymer token is centred on its named centre atom: BoltzGen features need it (upstream's rule)."""
    monomer = CLASS_MONOMER[layout.token_class]
    centres = system.text("atom", system.atom_symbol)[layout.token_centre_atom]
    wanted = np.array([BOLTZGEN_LAYOUT.centre_atom.get(MonomerType(int(kind)), "") for kind in monomer.tolist()])
    broken = np.flatnonzero((monomer > 0) & (centres != wanted))
    if len(broken):
        component = int(layout.token_component[broken[0]])
        fail("model_capacity", StructureRef(component=component), reason="no_centre_atom")


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


def _characters(names: Sequence[str]) -> Tensor:
    r"""The four-character code of each name (space is 0), as the reference-atom features carry it."""
    return torch.tensor(
        [[ord(c) - 32 if c != " " else 0 for c in name.ljust(4)] for name in names], dtype=torch.long
    ).reshape(len(names), 4)


def _atom_features(
    structure: Structure,
    rows: _AtomRows,
    truth: NDArray[np.int64],
    references: NDArray[np.float64],
    *,
    generator: torch.Generator | None,
) -> dict[str, Tensor]:
    r"""
    The 388-wide reference-atom features and the geometric model-row channels.

    The layout is reference positions (3), formal charge (1), element one-hot (128) and the four atom-name characters
    (4 x 64). A designed token's rows past its backbone are flagged — element ``_SYNTHETIC_ELEMENT`` named
    ``FL<slot>`` with no charge — whether the slot holds a real side-chain atom or a synthetic filler, so the
    designer never reads the identity it designs; every designed row has zero reference position, as upstream.
    Observed coordinates are centred on the mean of the observed rows, while a token's centre coordinate stays in the
    source frame.
    """

    system = structure.system
    count = len(rows.atom)
    flagged = rows.designed & (rows.offset >= len(BOLTZGEN_BACKBONE))
    stated = (rows.atom >= 0) & ~flagged
    positions = torch.from_numpy(references.astype(np.float32))
    space = torch.from_numpy(_owner(system.component_atom_ptr)[rows.reference])
    augment_reference_positions(positions, space, torch.ones(count, dtype=torch.bool), generator=generator)
    positions = positions.masked_fill(torch.from_numpy(rows.designed)[:, None], 0)
    names = ["" for _ in range(count)]
    for row, name in zip(np.flatnonzero(stated).tolist(), _atom_names(system, rows.atom[stated])):
        names[row] = name
    element = torch.from_numpy(np.where(stated, system.atomic_number[np.maximum(rows.atom, 0)], 0).astype(np.int64))
    charge = torch.from_numpy(np.where(stated, system.formal_charge[np.maximum(rows.atom, 0)], 0).astype(np.float32))
    for row in np.flatnonzero(flagged).tolist():
        names[row] = f"{_SYNTHETIC_NAME}{int(rows.offset[row])}"
        element[row] = _SYNTHETIC_ELEMENT
    bound = truth >= 0
    observed = torch.from_numpy(bound & structure.coordinate_mask[np.maximum(truth, 0)])
    coordinates = np.zeros((len(truth), 3), dtype=np.float64)
    coordinates[bound] = structure.coordinates[truth[bound]]
    source = torch.from_numpy(coordinates)
    coordinates = project_model_coordinates(source, observed, feature="BoltzGen observed coordinates")
    coordinates = coordinates - coordinates.sum(dim=0) / observed.sum().clamp(min=1)
    coordinates = coordinates.masked_fill(~observed[:, None], 0)
    return {
        "atom_features": torch.cat(
            (
                positions,
                charge.unsqueeze(-1),
                torch.nn.functional.one_hot(element, 128).float(),
                torch.nn.functional.one_hot(_characters(names), 64).reshape(count, 256).float(),
            ),
            dim=-1,
        ),
        "reference_positions": positions,
        "reference_space_id": space,
        "atom_mask": torch.ones(count, dtype=torch.bool),
        "atom_to_token": torch.from_numpy(rows.token.copy()),
        "atom_token_index": torch.from_numpy(rows.offset.copy()),
        "atom_coordinates": coordinates,
        "atom_coordinate_mask": observed,
        "raw_coordinates": source,
    }


def _row_of(system: MolecularSystem, rows: _AtomRows) -> NDArray[np.int64]:
    r"""Per canonical atom, its model row (``-1`` unaddressed)."""
    row_of = np.full(len(system.atom_symbol), -1, dtype=np.int64)
    real = rows.atom >= 0
    row_of[rows.atom[real]] = np.flatnonzero(real)
    return row_of


def _named_rows(
    system: MolecularSystem,
    layout: TokenLayout,
    row_of: NDArray[np.int64],
    names_of: Mapping[MonomerType, tuple[str, ...]],
) -> NDArray[np.int64]:
    r"""Per token, the model rows of the atoms its monomer type names (a token of no polymer class its own atoms)."""
    names = system.text("atom", system.atom_symbol)
    width = len(next(iter(names_of.values())))
    monomer = CLASS_MONOMER[layout.token_class]
    found = np.full((len(layout.token_component), width), -1, dtype=np.int64)
    for token, kind in enumerate(monomer.tolist()):
        held = layout.token_atoms[layout.token_atom_ptr[token] : layout.token_atom_ptr[token + 1]]
        wanted = names_of.get(MonomerType(kind))
        if wanted is None:
            found[token] = row_of[layout.token_centre_atom[token]]
            continue
        by_name = {str(names[a]): int(row_of[a]) for a in held.tolist()}
        found[token] = [by_name.get(name, -1) for name in wanted]
    return found


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


def _token_bonds(system: MolecularSystem, layout: TokenLayout) -> Tensor:
    r"""``[N, N, 1]``: 1 between two tokens an edge joins (the trailing axis feeds ``token_bond_projection``)."""
    first, second = system.edge_index.astype(np.int64)
    a, b = layout.atom_token[first], layout.atom_token[second]
    linked = (a >= 0) & (b >= 0) & (a != b)
    count = len(layout.token_component)
    bonds = torch.zeros(count, count)
    bonds[torch.from_numpy(a[linked]), torch.from_numpy(b[linked])] = 1
    bonds[torch.from_numpy(b[linked]), torch.from_numpy(a[linked])] = 1
    return bonds.unsqueeze(-1)


def _bond_rows(system: MolecularSystem, rows: _AtomRows) -> Tensor:
    r"""The addressed KNOWN covalent edges between two components, as model-row pairs ``[m, 2]``."""
    s = system.arrays()
    row_of = _row_of(system, rows)
    component = _owner(s["component_atom_ptr"])
    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    keep = covalent & (component[first] != component[second]) & (row_of[first] >= 0) & (row_of[second] >= 0)
    pairs = np.stack((row_of[first[keep]], row_of[second[keep]]), axis=1)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0) if len(pairs) else np.zeros((0, 2), dtype=np.int64)
    return torch.from_numpy(pairs.astype(np.int64))


def _token_features(
    system: MolecularSystem, layout: TokenLayout, rows: _AtomRows, tokens: NDArray[np.int64]
) -> dict[str, Tensor]:
    r"""The token-axis features shared by the structure designer and the inverse folder."""
    s = system.arrays()
    count = len(layout.token_component)
    component_instance = _owner(s["instance_component_ptr"])
    token_instance = component_instance[layout.token_component]
    entity = s["instance_entity"]
    residue_index = layout.token_component - s["instance_component_ptr"][token_instance]
    order = np.argsort(entity, kind="stable")
    copy = np.empty(len(entity), dtype=np.int64)
    copy[order] = np.arange(len(entity)) - np.searchsorted(entity[order], entity[order])
    molecule_type = np.array(
        [
            BOLTZGEN_LAYOUT.monomer_types.get(MonomerType(kind), _NONPOLYMER)
            for kind in CLASS_MONOMER[layout.token_class].tolist()
        ],
        dtype=np.int64,
    )
    row_of = _row_of(system, rows)
    return {
        "token_mask": torch.ones(count, dtype=torch.bool),
        "residue_index": torch.from_numpy(residue_index.astype(np.int64)),
        "chain_index": torch.from_numpy(token_instance),
        "entity_index": torch.from_numpy(entity[token_instance].astype(np.int64)),
        "symmetry_index": torch.from_numpy(copy[token_instance]),
        "token_index": torch.arange(count),
        "token_center_atom_index": torch.from_numpy(row_of[layout.token_centre_atom]),
        "token_to_bb4_atoms": torch.from_numpy(
            _named_rows(
                system,
                layout,
                row_of,
                {
                    _PROTEIN: BOLTZGEN_BACKBONE,
                    _DNA: ("C1'", "C3'", "C4'", "C1'"),
                    _RNA: ("C1'", "C3'", "C4'", "C1'"),
                },
            )
        ),
        "frame_atom_index": torch.from_numpy(row_of[token_frames(system, layout, BOLTZGEN_LAYOUT)]),
        "molecule_type": torch.from_numpy(molecule_type),
        "cyclic_period": torch.from_numpy(_cyclic_periods(system)[token_instance].astype(np.float32)),
        "method_feature": torch.zeros(count, dtype=torch.long),
        "modified": torch.zeros(count, dtype=torch.long),
        "token_bonds": _token_bonds(system, layout),
        "type_bonds": torch.zeros(count, count, dtype=torch.long),
        "residue_type_features": torch.nn.functional.one_hot(torch.from_numpy(tokens), len(TOKENS)).float(),
    }


def _centre_coordinates(features: Mapping[str, Tensor], tokens: Tensor) -> Tensor:
    r"""Per token, its centre row's observed coordinate in the source frame (zero when unobserved)."""
    rows = tokens.long()
    observed = features["atom_coordinate_mask"][rows]
    return features["raw_coordinates"][rows].masked_fill(~observed[:, None], 0)


# ===========================================================================
# MSA, templates and the conditioning channels.
# ===========================================================================


def _msa_plan(
    msa: Mapping[str, MSA] | MsaCollection | None, system: MolecularSystem, layout: TokenLayout, *, cropped: bool
) -> MsaRows | None:
    r"""Pools are paired by the one OFoldX rule after the crop; an exact plan is used as given."""
    if msa is None:
        return None
    plan: MsaCollection | None
    if isinstance(msa, MsaCollection):
        if cropped:
            fail("msa_plan_with_crop", RequestRef("msa", "MsaCollection"))
        if msa.row_pairing.shape[0] > BOLTZGEN_MSA_DEPTH:
            fail("msa_plan_too_deep", RequestRef("msa", str(msa.row_pairing.shape[0])), limit=BOLTZGEN_MSA_DEPTH)
        plan = msa
    else:
        plan = pair_msas(msa, system, layout, max_rows=BOLTZGEN_MSA_DEPTH)
    return None if plan is None else msa_rows(plan, system, layout)


def _msa_features(rows: MsaRows | None, tokens: NDArray[np.int64]) -> dict[str, Tensor]:
    r"""The MSA channels: classes, the upstream deletion transform, pairing, the profile and the deletion mean."""
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


def _mask_designed_msa(values: dict[str, Tensor], designed: Tensor) -> dict[str, Tensor]:
    r"""Remove designed-position identity from every model-visible MSA channel (the family's design masking)."""
    if not bool(designed.any()):
        return values
    unknown = TOKEN_INDEX["X"]
    values["msa_tokens"][:, designed] = unknown
    values["has_deletion"][:, designed] = False
    values["deletion_value"][:, designed] = 0
    values["msa_paired"][:, designed] = 0
    values["msa_profile"][designed] = 0
    values["msa_profile"][designed, unknown] = 1
    values["deletion_mean"][designed] = 0
    return values


def _template_names(token_class: int) -> tuple[str, str]:
    r"""The template centre and distogram atom names of one canonical class."""
    monomer = MonomerType(int(CLASS_MONOMER[token_class]))
    centre = BOLTZGEN_LAYOUT.centre_atom[monomer]
    disto = BOLTZGEN_LAYOUT.representative.get(token_class, BOLTZGEN_LAYOUT.representative_default[monomer])
    return centre, disto


def _template_frame(chain: ChainCoordinates, position: int) -> tuple[Tensor, Tensor] | None:
    r"""The protein backbone frame of one template position, ``None`` without three present non-degenerate atoms."""
    coordinates, present = template_atoms(chain, position, BOLTZGEN_LAYOUT.frame_atoms[_PROTEIN])
    if not present.all():
        return None
    values = project_model_coordinates(
        torch.from_numpy(coordinates), torch.ones(3, dtype=torch.bool), feature="BoltzGen template frame"
    )
    nitrogen, alpha, carbon = values
    first, second = carbon - alpha, nitrogen - alpha
    degenerate = (
        torch.linalg.vector_norm(first) == 0
        or torch.linalg.vector_norm(second) == 0
        or torch.linalg.vector_norm(torch.linalg.cross(first, second)) == 0
    )
    return None if degenerate else (rotation_from_two_vectors(first, second), alpha)


def _template_features(
    system: MolecularSystem, layout: TokenLayout, templates: Sequence[Template]
) -> dict[str, Tensor]:
    r"""
    The BoltzGen template channels of resolved template evidence, ``{}`` without templates (the trunk then skips its
    template embedder).

    ``resolve_template`` binds each alignment's targets and ``template_alignment_frames`` assigns its placement frame
    (``ofoldx.templates``); this function packs the per-position residue class, backbone frame, pseudo-beta and
    alpha-carbon coordinates of each aligned token.
    """

    if not templates:
        return {}
    count = len(layout.token_component)
    token_of = _token_of_component(layout, len(system.component_symbol))
    pointer = system.instance_component_ptr
    number = len(templates)
    residue_class = torch.zeros(number, count, dtype=torch.long)
    frame_rotations = torch.zeros(number, count, 3, 3)
    frame_translations = torch.zeros(number, count, 3)
    pseudo_beta = torch.zeros(number, count, 3)
    alpha_carbon = torch.zeros(number, count, 3)
    pseudo_beta_mask = torch.zeros(number, count)
    frame_mask = torch.zeros(number, count)
    template_mask = torch.zeros(number, count)
    centre_mask = torch.zeros(number, count, dtype=torch.bool)
    visibility = torch.full((number, count), -1, dtype=torch.long)
    for index, template in enumerate(templates):
        resolved = resolve_template(template, system, monomer_types=BOLTZGEN_TEMPLATE_TYPES)
        frames = template_alignment_frames(resolved)
        for alignment, frame in zip(resolved, frames):
            chain = template.coordinates.chains[alignment.template_instance_id]
            start = int(pointer[alignment.target_instance_index])
            for target, position in alignment.residue_pairs:
                token = int(token_of[start + target])
                if token < 0:
                    continue
                token_class = int(chain.token_class[position])
                centre_name, disto_name = _template_names(token_class)
                coordinates, present = template_atoms(chain, position, (centre_name, disto_name))
                residue_class[index, token] = token_class
                alpha_carbon[index, token] = torch.from_numpy(coordinates[0])
                pseudo_beta[index, token] = torch.from_numpy(coordinates[1])
                pseudo_beta_mask[index, token] = float(present[1])
                centre_mask[index, token] = bool(present[0])
                template_mask[index, token] = 1.0
                visibility[index, token] = frame
                built = _template_frame(chain, position) if CLASS_MONOMER[token_class] == _PROTEIN else None
                if built is not None:
                    frame_rotations[index, token], frame_translations[index, token] = built
                    frame_mask[index, token] = 1.0
    return {
        "template_residue_type_features": torch.nn.functional.one_hot(residue_class, len(TOKENS)),
        "template_frame_rotations": frame_rotations,
        "template_frame_translations": frame_translations,
        "template_pseudo_beta_positions": pseudo_beta,
        "template_alpha_carbon_positions": alpha_carbon,
        "template_pseudo_beta_mask": pseudo_beta_mask,
        "template_frame_mask": frame_mask,
        "template_mask": template_mask,
        "template_center_mask": centre_mask,
        "template_visibility_index": visibility,
    }


def _token_of_component(layout: TokenLayout, count: int) -> NDArray[np.int64]:
    r"""Per component, its residue token (``-1`` when it has none, or when it is tokenized per atom)."""
    counts = np.bincount(layout.token_component, minlength=count)
    token_of = np.full(count, -1, dtype=np.int64)
    single = counts[layout.token_component] == 1
    token_of[layout.token_component[single]] = np.flatnonzero(single)
    return token_of


def _component_token(system: MolecularSystem, layout: TokenLayout, position: ComponentRef, argument: str) -> int:
    r"""The residue token of one named position."""
    component = int(system.component_index([position])[0])
    token_of = _token_of_component(layout, len(system.component_symbol))
    if component < 0 or token_of[component] < 0:
        fail("request_invalid_positions", RequestRef(argument, str(position)))
    return int(token_of[component])


def _endpoint_tokens(
    system: MolecularSystem, layout: TokenLayout, endpoints: Sequence[Endpoint], argument: str
) -> tuple[int, ...]:
    r"""The token rows an endpoint set names, in declaration order: a residue its token, a per-atom tokenized
    component all of its tokens, an atom site its own token."""
    token_of = _token_of_component(layout, len(system.component_symbol))
    atom_component = _owner(system.component_atom_ptr)
    rows: list[int] = []
    for endpoint in endpoints:
        if isinstance(endpoint, AtomSite):
            atom = int(system.atom_index([endpoint])[0])
            token = int(layout.atom_token[atom]) if atom >= 0 else -1
            if token < 0:
                fail("request_invalid_positions", RequestRef(argument, str(endpoint)))
            rows.append(token)
            continue
        component = int(system.component_index([cast(ComponentRef, endpoint)])[0])
        if component < 0:
            fail("request_invalid_positions", RequestRef(argument, str(endpoint)))
        if token_of[component] >= 0:
            rows.append(int(token_of[component]))
            continue
        atoms = np.flatnonzero(atom_component == component)
        found = np.unique(layout.atom_token[atoms][layout.atom_token[atoms] >= 0])
        if not len(found):
            fail("request_invalid_positions", RequestRef(argument, str(endpoint)))
        rows += found.tolist()
    return tuple(dict.fromkeys(rows))


def _contact_channels(
    system: MolecularSystem, layout: TokenLayout, conditions: Sequence[BoltzGenSchema]
) -> dict[str, Tensor]:
    r"""The ``[N, N, 5]`` contact one-hot and the ``[N, N]`` cutoff; with no request every pair is UNSPECIFIED."""
    count = len(layout.token_component)
    requested = [c for c in conditions if isinstance(c, BoltzGenContact)]
    classes = torch.full((count, count), int(bool(requested)), dtype=torch.long)
    threshold = torch.zeros(count, count)
    for contact in requested:
        rows = _endpoint_tokens(system, layout, contact.side_a, "contact.side_a")
        columns = _endpoint_tokens(system, layout, contact.side_b, "contact.side_b")
        if not rows or not columns:
            fail("request_invalid_positions", RequestRef("contact", contact.kind), reason="empty_side")
        forward = _CONTACT_KIND_TO_CLASS[contact.kind]
        reverse = {2: 3, 3: 2}.get(forward, forward)
        for row in rows:
            for column in columns:
                classes[row, column], classes[column, row] = forward, reverse
                threshold[row, column] = threshold[column, row] = contact.cutoff_angstrom
    return {
        "contact_conditioning": torch.nn.functional.one_hot(classes, _NUM_CONTACT_CLASSES).float(),
        "contact_threshold": threshold,
    }


def _binding_ids(system: MolecularSystem, layout: TokenLayout, conditions: Sequence[BoltzGenSchema]) -> Tensor:
    r"""Per token, its binding-conditioning id."""
    ids = torch.full((len(layout.token_component),), _BINDING_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenBinding):
            continue
        for token in _endpoint_tokens(system, layout, condition.binding.residues, "binding"):
            ids[token] = _BINDING_BINDING
        for position in condition.not_binding:
            ids[_component_token(system, layout, position, "not_binding")] = _BINDING_NOT_BINDING
    return ids


def _ss_ids(system: MolecularSystem, layout: TokenLayout, conditions: Sequence[BoltzGenSchema]) -> Tensor:
    r"""Per token, its secondary-structure conditioning id."""
    ids = torch.full((len(layout.token_component),), _SS_UNSPECIFIED, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenSecondaryStructure):
            continue
        for position, label in condition.assignments:
            ids[_component_token(system, layout, position, "secondary_structure")] = _SS_LABEL_TO_ID[label]
    return ids


def _symmetry_ids(
    system: MolecularSystem, layout: TokenLayout, conditions: Sequence[BoltzGenSchema], copies: Tensor
) -> Tensor:
    r"""The per-entity copy index, overridden at the positions a symmetry condition names by its unit ordinal."""
    ids = copies.clone()
    for condition in conditions:
        if not isinstance(condition, BoltzGenSymmetry):
            continue
        for unit, positions in enumerate(condition.units):
            for position in positions:
                ids[_component_token(system, layout, position, "symmetry")] = unit
    return ids


def _token_distance_mask(
    system: MolecularSystem, layout: TokenLayout, conditions: Sequence[BoltzGenSchema], designed: Tensor
) -> Tensor:
    r"""
    The ``[N, N]`` token-distance visibility the requested structure groups imply.

    A designed position cannot anchor distance conditioning, so a nonzero group may not name it. Cell ``(i, j)`` is
    visible when ``i`` is grouped and ``j`` shares its group; a system entirely in group 1 is fully visible.
    """

    count = len(layout.token_component)
    ids = torch.zeros(count, dtype=torch.long)
    for condition in conditions:
        if not isinstance(condition, BoltzGenStructureGroup):
            continue
        for position in condition.members:
            token = _component_token(system, layout, position, "structure_group")
            if condition.group_id and bool(designed[token]):
                fail("request_invalid_positions", RequestRef("structure_group", str(position)), reason="designed")
            ids[token] = condition.group_id
    mask = torch.zeros(count, count)
    if count and bool((ids == 1).all()):
        return mask + 1
    same = ids.unsqueeze(1) == ids.unsqueeze(0)
    mask[same & (ids > 0).unsqueeze(1)] = 1
    return mask


def design_instances(system: MolecularSystem, components: NDArray[np.int64]) -> NDArray[np.bool_]:
    r"""
    Per instance, whether it holds one of ``components`` or a KNOWN covalent edge joins it to such an instance.

    This is the family's connected design scope: the model conditions on it (``chain_design_mask``) and the Pipeline
    reports it per candidate.
    """

    s = system.arrays()
    count = len(s["instance_symbol"])
    component_instance = _owner(s["instance_component_ptr"])
    atom_instance = component_instance[_owner(s["component_atom_ptr"])]
    first, second = s["edge_index"].astype(np.int64)
    covalent = known(s["bond_kind_state"]) & (s["bond_kind"] == BondKind.COVALENT)
    joined = covalent & (atom_instance[first] != atom_instance[second])
    edges = torch.from_numpy(np.stack((atom_instance[first[joined]], atom_instance[second[joined]]), axis=1))
    selected = np.zeros(count, dtype=bool)
    selected[component_instance[components]] = True
    for group in UndirectedGraph(edge_index=edges, num_nodes=count).connected_components():
        members = np.asarray(group.tolist(), dtype=np.int64)
        if selected[members].any():
            selected[members] = True
    return selected


def _chain_design_mask(system: MolecularSystem, layout: TokenLayout, designed: Tensor) -> Tensor:
    r"""Per token, whether its instance is in the connected design scope (``design_instances``)."""
    token_instance = _owner(system.instance_component_ptr)[layout.token_component]
    components = np.unique(layout.token_component[designed.cpu().numpy()])
    selected = design_instances(system, components)
    return torch.from_numpy(selected[token_instance])


def _symmetry_groups(slots: SlotMap, target: StructureTarget) -> tuple[Tensor, Tensor]:
    r"""The target's symmetry candidates on the model-row axis: ``[G, W]`` rows and ``[G, P, W]`` permutations,
    identity first, ``-1`` padded; a candidate moving a row onto an unaddressed atom is dropped."""
    candidates = target.symmetry
    row_of = np.full(len(target.truth), -1, dtype=np.int64)
    row_of[slots.atom[slots.atom >= 0]] = np.flatnonzero(slots.atom >= 0)
    groups: list[tuple[list[int], list[list[int]]]] = []
    for group in range(len(candidates.component)):
        atoms = candidates.atoms[candidates.atoms_ptr[group] : candidates.atoms_ptr[group + 1]]
        permutations = candidates.permutation[candidates.permutation_ptr[group] : candidates.permutation_ptr[group + 1]]
        permutations = permutations.reshape(-1, len(atoms))
        visible = row_of[atoms] >= 0
        if visible.sum() < 2:
            continue
        identity = row_of[atoms[visible]].tolist()
        found = [identity]
        for row in permutations:
            mapped = row[visible]
            if set(mapped.tolist()) == set(atoms[visible].tolist()):
                candidate = row_of[mapped].tolist()
                if candidate not in found:
                    found.append(candidate)
        if len(found) > 1:
            groups.append((identity, found))
    width = max((len(identity) for identity, _ in groups), default=0)
    depth = max((len(found) for _, found in groups), default=0)
    rows = torch.full((len(groups), width), -1, dtype=torch.long)
    permutations = torch.full((len(groups), depth, width), -1, dtype=torch.long)
    for group, (identity, found) in enumerate(groups):
        rows[group, : len(identity)] = torch.tensor(identity)
        permutations[group, : len(found), : len(identity)] = torch.tensor(found)
    return rows, permutations


# ===========================================================================
# Plan materialisation.
# ===========================================================================


def _generated_lengths(placement: PlacementSpec, total_len: int | None) -> dict[str, int]:
    r"""Each generated placement's length inside the ``total_len`` budget: every minimum first, then capacity in
    declaration order up to each maximum; ``None`` takes every maximum."""
    segments = [
        (placement_id, value)
        for group in placement.values()
        for placement_id, value in group.items()
        if isinstance(value, GeneratedSegment)
    ]
    if total_len is None:
        return {placement_id: value.max_length for placement_id, value in segments}
    minimum = sum(value.min_length for _, value in segments)
    maximum = sum(value.max_length for _, value in segments)
    if not minimum <= total_len <= maximum:
        fail("request_invalid_placement", RequestRef("total_len", str(total_len)), range=(minimum, maximum))
    lengths = {placement_id: value.min_length for placement_id, value in segments}
    remaining = total_len - minimum
    for placement_id, value in segments:
        added = min(remaining, value.max_length - value.min_length)
        lengths[placement_id] += added
        remaining -= added
    return lengths


def _existing_span(system: MolecularSystem, placement_id: str, span: ExistingSpan) -> tuple[ComponentRef, ...]:
    r"""The components one retained span addresses."""
    instance = int(system.instance_index([span.instance_id])[0])
    if instance < 0:
        fail("request_invalid_placement", RequestRef(placement_id, span.instance_id), reason="unknown_instance")
    entity = int(system.instance_entity[instance])
    polymer = bool(known(system.entity_type_state[entity]))
    polymer = polymer and system.entity_type[entity] == EntityType.POLYMER
    if not polymer:
        fail("request_invalid_placement", RequestRef(placement_id, span.instance_id), reason="not_a_polymer")
    pointer = system.instance_component_ptr
    count = int(pointer[instance + 1] - pointer[instance])
    if span.stop > count:
        fail("request_invalid_placement", RequestRef(placement_id, f"{span.start}..{span.stop}"), components=count)
    return tuple(ComponentRef(span.instance_id, index) for index in range(span.start, span.stop))


def _assemble(parts: Sequence[Structure]) -> Structure:
    r"""
    One structure holding every instance of ``parts``, each atom's coordinates restated with its own evidence.

    The parts are joined as systems (``data.union``) because a plan is not one deposited entry: joining the
    structures would be refused (``union`` requires the parts to state one source document, method, resolution and
    frame), and the source document, author identity, occupancy, B factor and resolution do not survive the join.
    """

    system, maps = union([part.system for part in parts])
    joined = Structure.unobserved(system)
    for part, projection in zip(parts, maps, strict=True):
        placed = np.flatnonzero(part.coordinate_mask)
        evidence = part.coordinate_evidence[placed]
        for code in np.unique(evidence).tolist():
            rows = placed[evidence == code]
            atoms = projection.atom[rows]
            scope = np.zeros(len(joined.coordinates), dtype=bool)
            scope[atoms] = True
            joined, _ = replace_coordinates(
                joined, atoms, part.coordinates[rows], scope=scope, evidence=CoordinateEvidence(int(code))
            )
    return joined


def _materialize(
    structure: Structure | None, placement: PlacementSpec, total_len: int | None, library: ComponentLibrary
) -> tuple[Structure, PlacementMap, tuple[ComponentRef, ...]]:
    r"""
    Realise one placement specification: retained spans address the source, generated segments become UNK protein
    chains appended in declaration order under minted instance ids.
    """

    lengths = _generated_lengths(placement, total_len)
    reserved = set() if structure is None else set(structure.system.symbols()["instance"])
    reserved_entities = set() if structure is None else set(structure.system.symbols()["entity"])
    entities: dict[str, PolymerSequence] = {}
    instances: dict[str, str] = {}
    realized: PlacementMap = {}
    generated: list[ComponentRef] = []
    for group, values in placement.items():
        realized[group] = {}
        for placement_id, value in values.items():
            if isinstance(value, ExistingSpan):
                if structure is None:
                    fail("request_invalid_placement", RequestRef(placement_id, "ExistingSpan"), reason="no_input")
                realized[group][placement_id] = _existing_span(structure.system, placement_id, value)
                continue
            if value.polymer_type != "protein":
                fail("model_capacity", RequestRef(placement_id, value.polymer_type), reason="generated_polymer_type")
            length = lengths[placement_id]
            if not length:
                realized[group][placement_id] = ()
                continue
            ordinal = len(instances)
            while f"chain_{ordinal}" in reserved:
                ordinal += 1
            instance_id = f"chain_{ordinal}"
            reserved.add(instance_id)
            number = len(entities)
            while f"boltzgen_entity_{number}" in reserved_entities:
                number += 1
            entity_id = f"boltzgen_entity_{number}"
            reserved_entities.add(entity_id)
            entities[entity_id] = PolymerSequence("polypeptide(L)", (_GENERATED_COMPONENT,) * length)
            instances[instance_id] = entity_id
            occurrences = tuple(ComponentRef(instance_id, index) for index in range(length))
            realized[group][placement_id] = occurrences
            generated += occurrences
    if not generated:
        fail("request_invalid_placement", RequestRef("placement", "no generated segment"))
    system, _, _ = build_system(Recipe(entities, instances), library)
    appended = Structure.unobserved(system)
    if structure is None:
        return appended, realized, tuple(generated)
    return _assemble((structure, appended)), realized, tuple(generated)


def _edge_columns(bonds: Sequence[SymbolicBond]) -> dict[str, NDArray]:
    r"""The declared edge columns of the requested links: the caller's kind, order, aromaticity and direction."""
    count = len(bonds)
    orders = np.array([0 if bond.order is None else int(bond.order) for bond in bonds], dtype=np.uint8)
    aromatic = np.array([bool(bond.aromatic) for bond in bonds], dtype=bool)
    direction = np.array(
        [{"undirected": 0, "a_to_b": 1, "b_to_a": -1}[bond.direction] for bond in bonds], dtype=np.int8
    )
    kinds = np.array(
        [BondKind.COVALENT if bond.kind == "covalent" else BondKind.COORDINATION for bond in bonds], dtype=np.uint8
    )
    return {
        "bond_order": orders,
        "bond_order_state": np.where(orders > 0, _DECLARED, _ABSENT).astype(np.uint8),
        "bond_aromatic": aromatic,
        "bond_aromatic_state": np.array(
            [_ABSENT if bond.aromatic is None else _DECLARED for bond in bonds], dtype=np.uint8
        ),
        "bond_kind": kinds,
        "bond_kind_state": np.full(count, _DECLARED, dtype=np.uint8),
        "bond_direction": direction,
        "bond_direction_state": np.where(direction != 0, _DECLARED, _ABSENT).astype(np.uint8),
        "bond_origin": np.full(count, BondOrigin.RECIPE, dtype=np.uint8),
    }


def _link(
    structure: Structure,
    bonds: Sequence[SymbolicBond],
    placement_map: PlacementMap,
    instance_of: Mapping[str, str],
) -> tuple[Structure, EditReport]:
    r"""
    Apply the requested links as one declared edit: the stated edges join their endpoints and the stated leaving
    atoms depart. Every component the edit touches keeps the claims it already carries and every surviving atom its
    hydrogen count, restated — the request states the whole link and its leaving atoms are graph atoms, so nothing
    becomes unknown — and every surviving atom keeps its coordinates. Returns the edited value and the edit's report.
    """

    system = structure.system
    sites = [
        (
            resolve_bond_site(bond.site_a, placement_map, dict(instance_of)),
            resolve_bond_site(bond.site_b, placement_map, dict(instance_of)),
        )
        for bond in bonds
    ]
    atoms = system.atom_index([site for pair in sites for site in pair])
    if (atoms < 0).any():
        missing = [site for pair in sites for site in pair][int(np.flatnonzero(atoms < 0)[0])]
        fail("request_invalid_positions", RequestRef("SymbolicBond", str(missing)))
    leaving = [
        AtomSite(site.instance, site.component, atom_id)
        for bond, pair in zip(bonds, sites)
        for site, ids in zip(pair, (bond.leaving_a, bond.leaving_b))
        for atom_id in sorted(ids)
    ]
    departing = system.atom_index(leaving) if leaving else np.zeros(0, dtype=np.int64)
    if len(departing) and (departing < 0).any():
        fail("request_invalid_positions", RequestRef("leaving", str(leaving[int(np.flatnonzero(departing < 0)[0])])))
    component = _owner(system.component_atom_ptr)
    touched = np.unique(component[atoms])
    surviving = np.setdiff1d(np.flatnonzero(np.isin(component, touched)), departing)
    edit = ChemicalEdit(
        remove_atoms=departing.astype(np.int64),
        replace_components={},
        retained_atoms={},
        remove_edges=np.zeros(0, dtype=np.int64),
        add_edges=EdgeRows(
            atoms=atoms.reshape(-1, 2).T.astype(np.int64),
            columns=_edge_columns(bonds),
            connection_type=(None,) * len(bonds),
            leaving=tuple(
                tuple((atom_id, side) for side, ids in enumerate((bond.leaving_a, bond.leaving_b)) for atom_id in ids)
                for bond in bonds
            ),
        ),
        set_atoms={
            "implicit_hydrogens": (
                surviving,
                system.implicit_hydrogens[surviving],
                system.implicit_hydrogens_state[surviving],
            )
        },
        claims={
            int(c): (bool(system.component_topology_complete[c]), bool(system.component_heavy_atoms_complete[c]))
            for c in touched.tolist()
        },
        keep_coordinates=np.ones(len(system.atom_symbol), dtype=bool),
        identity="inherit",
    )
    edited, _, report = apply_edit(structure, edit)
    return edited, report


def _resolve_conditions(
    conditions: Sequence[BoltzGenSchema], placement_map: PlacementMap, instance_of: Mapping[str, str]
) -> tuple[BoltzGenSchema, ...]:
    r"""Every symbolic residue, atom and placement id of the conditions replaced by the realised position."""
    index = dict(instance_of)

    def residue(value: ComponentRef | SymbolicResidueSite) -> ComponentRef:
        return resolve_placement(value, placement_map, index) if isinstance(value, SymbolicResidueSite) else value

    def endpoints(values: Sequence[Endpoint]) -> tuple[Endpoint, ...]:
        resolved: list[Endpoint] = []
        for value in values:
            if isinstance(value, str):
                resolved += list(placement_map[index[value]][value])
            elif isinstance(value, SymbolicResidueSite):
                resolved.append(residue(value))
            elif isinstance(value, SymbolicAtomSite):
                reference = residue(value.residue)
                resolved.append(AtomSite(reference.instance, reference.component, value.atom_id))
            else:
                resolved.append(value)
        return tuple(resolved)

    output: list[BoltzGenSchema] = []
    for condition in conditions:
        if isinstance(condition, BoltzGenDesignMask):
            output.append(BoltzGenDesignMask(tuple(residue(value) for value in condition.positions)))
        elif isinstance(condition, BoltzGenBinding):
            output.append(
                BoltzGenBinding(
                    binding=Hotspot(endpoints(condition.binding.residues)),
                    not_binding=tuple(residue(value) for value in condition.not_binding),
                )
            )
        elif isinstance(condition, BoltzGenSecondaryStructure):
            output.append(
                BoltzGenSecondaryStructure(tuple((residue(value), label) for value, label in condition.assignments))
            )
        elif isinstance(condition, BoltzGenStructureGroup):
            output.append(
                BoltzGenStructureGroup(tuple(residue(value) for value in condition.members), condition.group_id)
            )
        elif isinstance(condition, BoltzGenContact):
            output.append(
                BoltzGenContact(
                    endpoints(condition.side_a),
                    endpoints(condition.side_b),
                    condition.cutoff_angstrom,
                    condition.kind,
                )
            )
        elif isinstance(condition, BoltzGenSymmetry):
            output.append(BoltzGenSymmetry(tuple(tuple(residue(value) for value in unit) for unit in condition.units)))
        else:
            output.append(condition)
    return tuple(output)


def _condition_components(system: MolecularSystem, conditions: Sequence[BoltzGenSchema]) -> NDArray[np.int64]:
    r"""The components every condition names (a crop must keep their tokens)."""
    positions: list[Endpoint] = []
    for condition in conditions:
        if isinstance(condition, BoltzGenBinding):
            positions += [*condition.binding.residues, *condition.not_binding]
        elif isinstance(condition, BoltzGenSecondaryStructure):
            positions += [position for position, _ in condition.assignments]
        elif isinstance(condition, BoltzGenStructureGroup):
            positions += list(condition.members)
        elif isinstance(condition, BoltzGenContact):
            positions += [*condition.side_a, *condition.side_b]
        elif isinstance(condition, BoltzGenSymmetry):
            positions += [position for unit in condition.units for position in unit]
    refs = [
        ComponentRef(position.instance, position.component) if isinstance(position, AtomSite) else position
        for position in positions
        if isinstance(position, AtomSite | ComponentRef)
    ]
    if not refs:
        return np.zeros(0, dtype=np.int64)
    found = system.component_index(refs)
    return np.unique(found[found >= 0])


def _restrict_conditions(
    system: MolecularSystem, conditions: Sequence[BoltzGenSchema], kept: NDArray[np.int64]
) -> tuple[BoltzGenSchema, ...]:
    r"""The design masks restricted to the components a crop kept (every other condition is required and stays)."""
    inside = set(kept.tolist())

    def selected(positions: Sequence[ComponentRef]) -> tuple[ComponentRef, ...]:
        found = system.component_index(list(positions))
        return tuple(p for p, c in zip(positions, found.tolist()) if c in inside)

    return tuple(
        BoltzGenDesignMask(selected(condition.positions)) if isinstance(condition, BoltzGenDesignMask) else condition
        for condition in conditions
    )


class BoltzGenProcessor(Processor):
    r"""
    Prepare one structure-design request for BoltzGen.

    Args:
        config: Input-processing configuration. Defaults to ``BoltzGenProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = BoltzGenProcessorConfig
    config: BoltzGenProcessorConfig

    def __init__(
        self,
        config: BoltzGenProcessorConfig | None = None,
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

    def plan(
        self,
        structure: Structure | None = None,
        *,
        placement: PlacementSpec | None = None,
        total_len: int | None = None,
        conditions: Sequence[BoltzGenSchema] = (),
        library: ComponentLibrary | None = None,
    ) -> tuple[Structure, tuple[BoltzGenSchema, ...], tuple[ComponentRef, ...], EditReport]:
        r"""
        Materialise the design topology once and resolve every symbolic condition onto it.

        Generated placements become UNK protein chains (``chemistry.construction``, definitions from ``library``,
        default the configured one) appended to ``structure``; ``SymbolicBond`` conditions become one declared edit,
        whose ``EditReport`` is returned (empty without links: the configurations it cleared and why). The generated
        positions are returned and added to the conditions as a ``BoltzGenDesignMask``, so ``prepare`` reads the
        design scope from the conditions alone.

        Args:
            structure: The target or context; ``None`` designs a whole assembly from ``placement``/``total_len``.
            placement: The generated and retained segments of the request.
            total_len: The generated-length budget; without ``placement`` it appends one generated chain.
            conditions: BoltzGen conditioning records.
            library: Definitions of the generated components.
        """

        links = tuple(c for c in conditions if isinstance(c, SymbolicBond))
        spatial = tuple(c for c in conditions if not isinstance(c, SymbolicBond))
        if placement is None and total_len is not None:
            placement = {"assembly": {"chain_A": GeneratedSegment("protein", total_len, total_len)}}
        if placement is None and structure is None:
            fail("request_invalid_placement", RequestRef("placement", "None"), reason="nothing_generated")
        placement_map: PlacementMap = {}
        instance_of: dict[str, str] = {}
        generated: tuple[ComponentRef, ...] = ()
        if placement is not None:
            instance_of = placement_instance_of(placement)
            structure, placement_map, generated = _materialize(
                structure, placement, total_len, configured_library() if library is None else library
            )
        assert structure is not None
        if generated:
            spatial = (*spatial, BoltzGenDesignMask(generated))
        nothing = np.zeros(0, dtype=np.int64)
        report = EditReport(ParityClears(nothing, np.zeros(0, dtype=np.uint16)), {}, nothing, nothing)
        if links:
            structure, report = _link(structure, links, placement_map, instance_of)
        return structure, _resolve_conditions(spatial, placement_map, instance_of), generated, report

    def prepare_input(
        self,
        structure: Structure,
        *,
        conditions: Sequence[BoltzGenSchema] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        templates: Sequence[Template] = (),
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[BoltzGenInput]:
        r"""
        Prepare one materialised design plan: layout, optional training crop, atom14 design rows, gating, reference
        conformers, conditioning channels, MSA rows, template channels and the BoltzGen tensors.

        The scope is every atom of the designed components in the source frame.

        Args:
            structure: The materialised plan (``plan``); its chemistry is consumed as read (no library).
            conditions: Resolved conditioning records, the design mask included.
            msa: MSA pools by instance (paired by ``pair_msas`` after the crop), an exact plan, or ``None``.
            templates: Structural templates.
            cropper: Optional training crop policy; needs ``generator``.
            generator: The numpy generator of the crop and of the reference-position augmentation.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        designed = _designed_components(system, conditions)
        atom14 = _atom14_table(system, designed)
        addressed = _addressed_atoms(system, {c: _ATOM14_NAMES[r] for c, r in atom14.items()})
        layout = token_layout(system, BOLTZGEN_LAYOUT, atom_mask=addressed)
        if cropper is not None:
            required = np.isin(layout.token_component, _condition_components(system, conditions))
            layout = crop_token_layout(structure, layout, cropper, required=required, generator=generator)
            kept = np.unique(layout.token_component)
            conditions = _restrict_conditions(system, conditions, kept)
            designed = designed[np.isin(designed, kept)]
            atom14 = {c: name for c, name in atom14.items() if c in set(kept.tolist())}
        _require_residue_tokens(system, layout)
        rows = _atom_rows(system, layout, atom14)
        slots = _slots(system, layout, rows)
        truth = _truth_source(system, slots)
        accepted = gate(structure, slots, BOLTZGEN_GATING)
        references, sources, embedded = reference_conformers(
            structure,
            SlotMap(rows.reference, rows.token, layout, slots.unaddressed),
            embed=missing_reference(system),
            seed=self.seed,
        )
        tokens, losses = model_tokens(system, layout)
        plan = _msa_plan(msa, system, layout, cropped=cropper is not None)
        torch_generator = None if generator is None else torch.Generator().manual_seed(int(generator.integers(2**62)))
        features = self._features(
            structure,
            slots,
            rows,
            truth,
            tokens,
            references,
            plan,
            tuple(templates),
            conditions,
            generator=torch_generator,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        conformers[_owner(system.component_atom_ptr)[rows.reference]] = sources
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.components(system, designed),
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
        prepared: PreparedInput[BoltzGenInput],
    ) -> BoltzGenStructureTarget:
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
            symmetry=BOLTZGEN_SYMMETRY,
        )
        rows = align_target_components(
            prepared.context.structure, target, prepared.context.slots.layout.token_component
        )
        if (rows < 0).any():
            fail("value_invalid", RequestRef("target", "missing BoltzGen token component identities"))
        classes = token_classes(target.system).token[rows].astype(np.int64)
        return self._target_tensors(prepared, truth, classes)

    def _target_tensors(
        self, prepared: PreparedInput[BoltzGenInput], target: StructureTarget, classes: NDArray[np.int64]
    ) -> BoltzGenStructureTarget:
        r"""The model's tensor target: canonical truth with the per-row truth source, chain spans and equivalence, the
        model-row symmetry groups, the unmasked residue classes and the measured B factor of each token's centre."""
        system = prepared.context.structure.system
        truth = _truth_source(system, prepared.context.slots)
        observed = torch.from_numpy(target.truth == TruthState.OBSERVED)
        coordinates = project_model_coordinates(
            torch.from_numpy(target.coordinates.copy()), observed, feature="BoltzGen truth"
        )
        atom_ptr = system.component_atom_ptr[system.instance_component_ptr]
        rows, permutations = _symmetry_groups(prepared.context.slots, target)
        centre = truth[prepared.features.token_center_atom_index[0].cpu().numpy()]
        b_factor = torch.from_numpy(target.b_factor[centre].copy())
        b_factor_mask = torch.from_numpy(target.b_factor_mask[centre].copy())
        return BoltzGenStructureTarget(  # type: ignore[call-arg]
            coordinates=coordinates.to(self.device).unsqueeze(0).unsqueeze(0),
            coordinate_mask=observed.to(self.device).unsqueeze(0),
            coordinate_atom_index=torch.from_numpy(truth.copy()).to(self.device).unsqueeze(0),
            chain_span=torch.from_numpy(np.stack((atom_ptr[:-1], atom_ptr[1:]), axis=1)).to(self.device).unsqueeze(0),
            chain_equivalence_index=torch.from_numpy(system.instance_entity.astype(np.int64))
            .to(self.device)
            .unsqueeze(0),
            symmetry_atom_index=rows.to(self.device).unsqueeze(0),
            symmetry_permutation=permutations.to(self.device).unsqueeze(0),
            residue_type=torch.nn.functional.one_hot(torch.from_numpy(classes), len(TOKENS))
            .to(dtype=self.dtype, device=self.device)
            .unsqueeze(0),
            b_factor=project_model_features(b_factor, b_factor_mask, feature="BoltzGen measured B factors")
            .to(self.device)
            .unsqueeze(0),
            b_factor_mask=b_factor_mask.to(self.device).unsqueeze(0),
            batch_size=[1],
        )

    def _features(
        self,
        structure: Structure,
        slots: SlotMap,
        rows: _AtomRows,
        truth: NDArray[np.int64],
        tokens: NDArray[np.int64],
        references: NDArray[np.float64],
        plan: MsaRows | None,
        templates: tuple[Template, ...],
        conditions: Sequence[BoltzGenSchema],
        *,
        generator: torch.Generator | None,
    ) -> BoltzGenInput:
        r"""The BoltzGen design tensors of one prepared plan (upstream featurisation over the model-row axis)."""
        system = structure.system
        layout = slots.layout
        atoms = _atom_features(structure, rows, truth, references, generator=generator)
        token = _token_features(system, layout, rows, tokens)
        designed = torch.zeros(len(layout.token_component), dtype=torch.bool)
        designed[torch.from_numpy(rows.token[rows.designed]).long()] = True
        residue_type = token["residue_type_features"].clone()
        residue_type[designed] = 0
        residue_type[designed, TOKEN_INDEX["X"]] = 1
        values: dict[str, Tensor] = {
            **{name: value for name, value in token.items() if name != "residue_type_features"},
            **{name: value for name, value in atoms.items() if name != "raw_coordinates"},
            "residue_type_features": residue_type,
            "symmetry_index": _symmetry_ids(system, layout, conditions, token["symmetry_index"]),
            "center_coordinates": _centre_coordinates(atoms, token["token_center_atom_index"]),
            "token_distance_mask": _token_distance_mask(system, layout, conditions, designed),
            "binding_type": _binding_ids(system, layout, conditions),
            "ss_type": _ss_ids(system, layout, conditions),
            "design_mask": designed,
            "chain_design_mask": _chain_design_mask(system, layout, designed),
            **_contact_channels(system, layout, conditions),
            **_mask_designed_msa(_msa_features(plan, tokens), designed),
            **_template_features(system, layout, templates),
        }
        placed = {name: value.to(device=self.device) for name, value in values.items()}
        bonds = _bond_rows(system, rows).to(device=self.device)
        return cast(
            BoltzGenInput,
            normalize_boltzgen_input(
                BoltzGenInput(  # type: ignore[call-arg]
                    **{name: value.unsqueeze(0) for name, value in placed.items()},
                    atom_index=torch.from_numpy(rows.atom.copy()).to(device=self.device).unsqueeze(0),
                    bond_atom_index=bonds.unsqueeze(0),
                    bond_mask=torch.ones(1, len(bonds), dtype=torch.bool, device=self.device),
                    batch_size=[],
                ),
                dtype=self.dtype,
            ),
        )


class BoltzGenInverseFoldingProcessor(Processor):
    r"""
    Prepare one fixed backbone for BoltzGen inverse folding.

    Args:
        config: Input-processing configuration. Defaults to ``BoltzGenProcessorConfig()``.
        seed: Optional override for the configured reference-conformer embedding seed.
        device: Device for produced tensors.
        dtype: Floating-point dtype for produced tensors.
    """

    config_class = BoltzGenProcessorConfig
    config: BoltzGenProcessorConfig

    def __init__(
        self,
        config: BoltzGenProcessorConfig | None = None,
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
        design: PositionSelection = PositionSelection(),
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities] = (),
        global_omit_tokens: Sequence[str] = (),
        msa: Mapping[str, MSA] | MsaCollection | None = None,
        cropper: Cropper | None = None,
        generator: np.random.Generator | None = None,
    ) -> PreparedInput[BoltzGenInverseFoldingInput]:
        r"""
        Prepare one backbone for sequence design: the designed positions are addressed by their backbone alone and
        their identity is masked, the identity support follows the conditions, and tied groups share one identity.

        Coordinates are never regenerated: the scope is empty and the design producer builds the final target.

        Args:
            structure: The fixed backbone.
            design: The positions to design among the protein residues (default every one).
            conditions: Allowed-identity and tied-identity conditions.
            global_omit_tokens: Canonical letters excluded at every designed position.
            msa: MSA pools by instance, an exact plan, or ``None``.
            cropper: Optional training crop; needs ``generator``.
            generator: The numpy generator of the crop and of the reference-position augmentation.
        """

        if cropper is not None and generator is None:
            fail("request_invalid_generator", RequestRef("generator", "None"))
        system = structure.system
        monomer = CLASS_MONOMER[token_classes(system).token]
        base = token_layout(system, BOLTZGEN_LAYOUT)
        allowed = np.zeros(len(system.component_symbol), dtype=bool)
        allowed[base.token_component] = monomer[base.token_component] == _PROTEIN
        allowed &= _token_of_component(base, len(system.component_symbol)) >= 0
        positions = resolve_positions(structure, design, allowed=allowed)
        layout = token_layout(
            system, BOLTZGEN_LAYOUT, atom_mask=_addressed_atoms(system, {c: BOLTZGEN_BACKBONE for c in positions})
        )
        if cropper is not None:
            named = _named_components(system, conditions)
            required = np.isin(layout.token_component, named)
            layout = crop_token_layout(structure, layout, cropper, required=required, generator=generator)
            positions = positions[np.isin(positions, layout.token_component)]
        _require_residue_tokens(system, layout)
        rows = _atom_rows(system, layout, {})
        designed_component = np.zeros(len(system.component_symbol), dtype=bool)
        designed_component[positions] = True
        rows = _AtomRows(
            atom=rows.atom,
            token=rows.token,
            offset=rows.offset,
            reference=rows.reference,
            designed=designed_component[_owner(system.component_atom_ptr)[rows.reference]],
        )
        slots = _slots(system, layout, rows)
        accepted = gate(structure, slots, BOLTZGEN_GATING)
        references, sources, embedded = reference_conformers(
            structure,
            SlotMap(rows.reference, rows.token, layout, slots.unaddressed),
            embed=missing_reference(system),
            seed=self.seed,
        )
        tokens, losses = model_tokens(system, layout)
        plan = _msa_plan(msa, system, layout, cropped=cropper is not None)
        torch_generator = None if generator is None else torch.Generator().manual_seed(int(generator.integers(2**62)))
        features = self._features(
            structure,
            slots,
            rows,
            tokens,
            references,
            plan,
            positions,
            conditions,
            tuple(global_omit_tokens),
            generator=torch_generator,
        )
        conformers = np.zeros(len(system.component_symbol), dtype=np.uint8)
        conformers[_owner(system.component_atom_ptr)[rows.reference]] = sources
        return PreparedInput(
            features,
            PreparationContext(
                structure,
                slots,
                RegenerationScope.none(system),
                PreparationRecord(
                    Records.concat((component_records("encoding_loss", losses.component), accepted, embedded)),
                    conformers,
                ),
                spec=self.preparation_spec,
            ),
        )

    def prepare_target(
        self, target: Structure, *, prepared: PreparedInput[BoltzGenInverseFoldingInput]
    ) -> SequenceTarget:
        r"""Align sequence classes to the existing token axis without changing input masking.

        Missing or unmapped components have false target availability. Execution
        protocols decide whether their required teacher history is complete.
        """
        self._validate_prepared(prepared)
        components = torch.from_numpy(prepared.context.slots.layout.token_component).to(self.device).unsqueeze(0)
        return prepare_sequence_target(prepared.context.structure, target, components)

    def _features(
        self,
        structure: Structure,
        slots: SlotMap,
        rows: _AtomRows,
        tokens: NDArray[np.int64],
        references: NDArray[np.float64],
        plan: MsaRows | None,
        positions: NDArray[np.int64],
        conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
        global_omit_tokens: tuple[str, ...],
        *,
        generator: torch.Generator | None,
    ) -> BoltzGenInverseFoldingInput:
        r"""The BoltzGen inverse-folding tensors of one prepared backbone."""
        system = structure.system
        layout = slots.layout
        atoms = _atom_features(structure, rows, slots.atom, references, generator=generator)
        token = _token_features(system, layout, rows, tokens)
        designed = torch.zeros(len(layout.token_component), dtype=torch.bool)
        designed[torch.from_numpy(_token_of_component(layout, len(system.component_symbol))[positions]).long()] = True
        residue_type = token["residue_type_features"].clone()
        residue_type[designed] = 0
        residue_type[designed, TOKEN_INDEX["X"]] = 1
        centre = token["token_center_atom_index"].long()
        # the inverse folder reads the alignment as the profile and the deletion mean only
        msa = _mask_designed_msa(_msa_features(plan, tokens), designed)
        values: dict[str, Tensor] = {
            "native_token_index": torch.from_numpy(tokens.copy()),
            "token_mask": token["token_mask"],
            "residue_index": token["residue_index"],
            "chain_index": token["chain_index"],
            "atom_features": atoms["atom_features"],
            "atom_to_token": atoms["atom_to_token"],
            "atom_mask": atoms["atom_mask"],
            "reference_positions": atoms["reference_positions"],
            "reference_space_id": atoms["reference_space_id"],
            "residue_type_features": residue_type,
            "token_bonds": token["token_bonds"],
            "type_bonds": token["type_bonds"],
            "molecule_type": token["molecule_type"],
            "center_coordinates": _centre_coordinates(atoms, centre),
            "token_to_bb4_atoms": token["token_to_bb4_atoms"],
            "atom_coordinates": atoms["atom_coordinates"],
            "atom_coordinate_mask": atoms["atom_coordinate_mask"],
            "token_resolved_mask": atoms["atom_coordinate_mask"][centre],
            "method_feature": token["method_feature"],
            "modified": token["modified"],
            "cyclic_period": token["cyclic_period"],
            "design_mask": designed,
            "allowed_token_mask": _allowed_classes(system, layout, conditions, global_omit_tokens),
            "symmetric_group": _tied_groups(system, layout, conditions),
            "msa_profile": msa["msa_profile"],
            "deletion_mean": msa["deletion_mean"],
        }
        placed = {name: value.to(device=self.device) for name, value in values.items()}
        return cast(
            BoltzGenInverseFoldingInput,
            normalize_boltzgen_input(
                BoltzGenInverseFoldingInput(  # type: ignore[call-arg]
                    **{name: value.unsqueeze(0) for name, value in placed.items()},
                    atom_index=torch.from_numpy(rows.atom.copy()).to(device=self.device).unsqueeze(0),
                    batch_size=[],
                ),
                dtype=self.dtype,
            ),
        )


def _named_components(
    system: MolecularSystem, conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities]
) -> NDArray[np.int64]:
    r"""The components the inverse-folding conditions name (a crop must keep their tokens)."""
    positions = [
        position
        for condition in conditions
        for position in (
            condition.positions
            if isinstance(condition, AllowedResidueIdentities)
            else [member for group in condition.groups for member in group]
        )
    ]
    if not positions:
        return np.zeros(0, dtype=np.int64)
    found = system.component_index(positions)
    return np.unique(found[found >= 0])


def _allowed_classes(
    system: MolecularSystem,
    layout: TokenLayout,
    conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities],
    global_omit_tokens: tuple[str, ...],
) -> Tensor:
    r"""
    Per token, the canonical classes a designed position may take.

    Only the 20 trained residues are ever allowed; ``global_omit_tokens`` removes a canonical letter everywhere, and
    each allowed-identity condition intersects its position's support (overlapping conditions therefore agree).
    """

    count = len(layout.token_component)
    mask = torch.zeros(count, len(TOKENS), dtype=torch.bool)
    mask[:, list(_GENERATABLE)] = True
    for letter in global_omit_tokens:
        if letter not in TOKEN_INDEX:
            fail("request_invalid_positions", RequestRef("global_omit_tokens", letter))
        mask[:, TOKEN_INDEX[letter]] = False
    for condition in conditions:
        if not isinstance(condition, AllowedResidueIdentities):
            continue
        classes = [
            TOKEN_INDEX[CANONICAL_LETTER[name]] for name in sorted(condition.component_ids) if name in CANONICAL_LETTER
        ]
        allowed = torch.zeros(len(TOKENS), dtype=torch.bool)
        allowed[classes] = True
        for position in condition.positions:
            token = _component_token(system, layout, position, "allowed_identities")
            mask[token] &= allowed
    return mask


def _tied_groups(
    system: MolecularSystem, layout: TokenLayout, conditions: Sequence[AllowedResidueIdentities | TiedResidueIdentities]
) -> Tensor:
    r"""Per token, its sequence-symmetry group id; zero is untied and each group is numbered from one. Groups sharing
    a token are one group, as ProteinMPNN ties them."""
    parent: dict[int, int] = {}

    def find(token: int) -> int:
        while parent[token] != token:
            parent[token] = parent[parent[token]]
            token = parent[token]
        return token

    for condition in conditions:
        if not isinstance(condition, TiedResidueIdentities):
            continue
        for group in condition.groups:
            members = tuple(
                dict.fromkeys(_component_token(system, layout, position, "tied_identities") for position in group)
            )
            if len(members) < 2:
                continue
            for member in members:
                parent.setdefault(member, member)
            root = find(members[0])
            for member in members[1:]:
                parent[find(member)] = root
    ids = torch.zeros(len(layout.token_component), dtype=torch.long)
    numbers: dict[int, int] = {}
    for token in sorted(parent):
        ids[token] = numbers.setdefault(find(token), len(numbers) + 1)
    return ids
