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
Position selection and the one design producer.

A sequence-design family samples canonical classes; ``produce_design`` turns them into the final target once:
a class that encodes the source component retains it (MSE sampled as ``M`` stays MSE), every other designed position
is replaced by the one standard component of its class through ``chemistry.construction`` and one ``apply_edit``.
A guide atom's coordinate is the input's own, carried over unchanged, so it keeps the evidence, occupancy, su and B
the input states for it; an atom the design builds has none, and whatever later writes one labels it by its own rule.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ..chemistry.construction import (
    Admissibility,
    ReplacementCheck,
    instantiate_component,
    replacement_admissibility,
)
from ..chemistry.definitions import ComponentLibrary, get_component
from ..data.editing import ChemicalEdit, EdgeRows, EditReport, apply_edit
from ..data.molecular import ComponentRef, MolecularSystem, MonomerType
from ..data.stereo import ParityClears
from ..data.structure import Structure
from ..data.tokenization import CLASS_MONOMER, TOKEN_INDEX, TOKENS, token_classes
from ..errors import RequestRef, StructureRef, fail
from ..provenance import Records, known
from .layout import EncodingLoss, LayoutTable, model_tokens, token_layout
from .preparation import I, PreparedInput, component_records

_AMINO = dict(
    zip(
        "ARNDCQEGHILKMFPSTWYV",
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
    )
)
# the one standard component of each class that has exactly one; every other class is unmappable
_STANDARD_COMPONENT: Mapping[int, str] = {
    **{TOKEN_INDEX[letter]: name for letter, name in {**_AMINO, "U": "SEC", "O": "PYL"}.items()},
    **{TOKEN_INDEX[f"RNA_{letter}"]: letter for letter in "ACGUI"},
    **{TOKEN_INDEX[f"DNA_{letter}"]: f"D{letter}" for letter in "ACGTI"},
}


@dataclass(frozen=True)
class PositionSelection:
    r"""
    The positions of a request, in one form: component references, instances (their allowed components), component
    indices of the target, or author addresses ``(author chain, auth_seq_id text, insertion code or "")``; nothing
    named selects every allowed component. ``exclude`` removes components from any form.
    """

    components: tuple[ComponentRef, ...] | None = None
    instances: tuple[str, ...] | None = None
    exclude: tuple[ComponentRef, ...] = ()
    indices: NDArray[np.int64] | None = None
    author: tuple[tuple[str, str, str], ...] | None = None


def _invalid(argument: str, value: object, **detail: object) -> None:
    fail("request_invalid_positions", RequestRef(argument, str(value)), **detail)


def _author_matches(target: Structure, address: tuple[str, str, str]) -> NDArray[np.bool_]:
    r"""Per component, whether the author address names it (the source texts, never a parsed number)."""
    system = target.system
    component_instance = np.repeat(np.arange(len(system.instance_symbol)), np.diff(system.instance_component_ptr))
    chain = np.asarray((*target.symbols()["chain"], ""), dtype=object)[target.instance_author_chain]
    chain = np.where(known(target.instance_author_chain_state), chain, None)[component_instance]
    number = np.asarray((*target.symbols()["author_number"], ""), dtype=object)[target.component_author_text]
    insertion = np.asarray((*target.symbols()["insertion"], ""), dtype=object)[target.component_insertion]
    return (chain == address[0]) & (number == address[1]) & (insertion == address[2])


def resolve_positions(
    target: Structure, selection: PositionSelection, *, allowed: NDArray[np.bool_]
) -> NDArray[np.int64]:
    r"""
    The component indices a selection names among ``allowed`` (per component), sorted.

    Raises:
        InvalidRequestError: ``request_invalid_positions`` for several forms at once, a reference, instance or index
            that names nothing, an explicitly named position that is not allowed, or an author address matching zero
            or several allowed components.
    """

    system = target.system
    allowed = np.asarray(allowed, dtype=bool)
    forms = [form is not None for form in (selection.components, selection.instances, selection.indices)]
    if sum(forms) + (selection.author is not None) > 1:
        _invalid("selection", "several forms")
    if selection.components is not None:
        chosen = system.component_index(list(selection.components))
        if (chosen < 0).any():
            _invalid("components", selection.components[int(np.flatnonzero(chosen < 0)[0])])
    elif selection.instances is not None:
        instances = system.instance_index(list(selection.instances))
        if (instances < 0).any():
            _invalid("instances", selection.instances[int(np.flatnonzero(instances < 0)[0])])
        component_instance = np.repeat(np.arange(len(system.instance_symbol)), np.diff(system.instance_component_ptr))
        chosen = np.flatnonzero(np.isin(component_instance, instances) & allowed)
    elif selection.indices is not None:
        chosen = np.asarray(selection.indices, dtype=np.int64)
        if ((chosen < 0) | (chosen >= len(allowed))).any():
            _invalid("indices", chosen.tolist())
    elif selection.author is not None:
        found = []
        for address in selection.author:
            matches = np.flatnonzero(_author_matches(target, address) & allowed)
            if len(matches) != 1:
                _invalid("author", address, matches=len(matches))
            found.append(int(matches[0]))
        chosen = np.array(found, dtype=np.int64)
    else:
        chosen = np.flatnonzero(allowed)
    if not allowed[chosen].all():
        _invalid("positions", system.component_refs(chosen[~allowed[chosen]][:1])[0], reason="not_allowed")
    excluded = system.component_index(list(selection.exclude))
    if (excluded < 0).any():
        _invalid("exclude", selection.exclude[int(np.flatnonzero(excluded < 0)[0])])
    return np.setdiff1d(chosen, excluded).astype(np.int64)


@dataclass(frozen=True)
class GuideAtoms:
    r"""Per monomer type, the atom names whose coordinates guide a replaced residue."""

    names: Mapping[MonomerType, frozenset[str]]
    BACKBONE: ClassVar[GuideAtoms]


_NUCLEIC_BACKBONE = frozenset(("P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"))
GuideAtoms.BACKBONE = GuideAtoms(
    {
        MonomerType.PROTEIN: frozenset(("N", "CA", "C", "O")),
        MonomerType.RNA: _NUCLEIC_BACKBONE,
        MonomerType.DNA: _NUCLEIC_BACKBONE,
    }
)


@dataclass(frozen=True)
class DesignPolicy:
    r"""
    ``replace_encoded=False``: a sampled class that encodes the source component retains it. ``identity``: the
    author numbering of replaced components is inherited or withdrawn. ``break_connections``: edges a replacement may
    drop. ``allow_unknown`` permits a sampled protein ``X`` to produce the
    CCD ``UNK`` component instead of treating that class as unmappable.
    """

    replace_encoded: bool = False
    identity: Literal["withdraw", "inherit"] = "inherit"
    guide: GuideAtoms = GuideAtoms.BACKBONE
    break_connections: tuple[int, ...] = ()
    allow_unknown: bool = False


@dataclass(frozen=True)
class DesignedPosition:
    r"""One designed position: source and result ids, sampled class, basis, guide atoms kept, source atoms withdrawn."""

    component: int
    source: str
    result: str
    token: int
    basis: Literal["retained_encoded", "retained_unchanged", "replaced"]
    guide_atoms: tuple[str, ...]
    withdrawn: tuple[str, ...]
    admissibility: ReplacementCheck


@dataclass(frozen=True)
class DesignRecord:
    r"""What one design produced: per position its outcome, the edit's report, encoding losses and records."""

    positions: tuple[DesignedPosition, ...]
    edit: EditReport
    losses: EncodingLoss
    records: Records

    def replaced(self) -> NDArray[np.int64]:
        r"""The component indices whose basis is ``"replaced"`` (the packer's input)."""
        return np.array([p.component for p in self.positions if p.basis == "replaced"], dtype=np.int64)


def _no_edges() -> EdgeRows:
    skip = ("edge_index", "bond_connection_type")
    fields = MolecularSystem.SCHEMA.fields
    edge = {f.name: np.zeros((0, *f.trailing), f.dtype) for f in fields if f.axis == "edge" and f.name not in skip}
    return EdgeRows(np.zeros((2, 0), np.int64), edge, (), ())


def _empty_report() -> EditReport:
    nothing = np.zeros(0, dtype=np.int64)
    return EditReport(ParityClears(nothing, np.zeros(0, dtype=np.uint16)), {}, nothing, nothing)


def produce_design(
    prepared: PreparedInput[I],
    classes: NDArray[np.int64],
    *,
    positions: NDArray[np.int64],
    table: LayoutTable,
    library: ComponentLibrary,
    policy: DesignPolicy = DesignPolicy(),
) -> tuple[Structure, DesignRecord]:
    r"""
    The final target of one design: per position the sampled canonical class → retained or replaced.

    (1) A class with no single standard component is ``InvalidRequestError(design_token_unmappable)``; the class of the
    source component's encoding retains it unless ``policy.replace_encoded``. (2) Each replacement's connections to
    other components are checked (``chemistry.construction.replacement_admissibility``): an inadmissible one not in
    ``policy.break_connections`` is ``UnsupportedError(design_breaks_connection)``, an undetermined one is recorded.
    (3) The replacement's rows come from ``instantiate_component`` (Rule L at its linked ports). (4) One
    ``apply_edit`` replaces every replaced component; guide atoms and connection endpoints are retained by name. A
    guide atom's coordinate is the input's own, carried over by that name correspondence and unchanged, so it keeps
    the coordinate evidence, occupancy, su and B the input states for it: the provenance of those numbers, not a claim
    about the designed identity, which is the sampled class. An atom the design builds has no coordinate.
    """

    target = prepared.context.structure
    system = target.system
    s = system.arrays()
    names = system.text("atom", s["atom_symbol"])
    ids = system.text("component", s["component_symbol"])
    encoded = token_classes(system).token
    pointer = s["component_atom_ptr"]
    atom_component = np.repeat(np.arange(len(ids)), np.diff(pointer))
    first, second = s["edge_index"].astype(np.int64)
    designed, rows_of, retained_of, remove, keep = [], {}, {}, [], np.zeros(len(names), dtype=bool)
    undetermined = []
    for k, c in zip(np.asarray(classes, dtype=np.int64).tolist(), np.asarray(positions, dtype=np.int64).tolist()):
        result = "UNK" if policy.allow_unknown and k == TOKEN_INDEX["X"] else _STANDARD_COMPONENT.get(k)
        if result is None:
            fail("design_token_unmappable", StructureRef(component=c), token=TOKENS[k])
        source = str(ids[c])
        atoms = np.arange(pointer[c], pointer[c + 1])
        nothing = ReplacementCheck(np.zeros(0, np.int64), np.zeros(0, np.uint8), np.zeros(0, np.uint16))
        if source == result or (encoded[c] == k and not policy.replace_encoded):
            basis: Literal["retained_unchanged", "retained_encoded"] = (
                "retained_unchanged" if source == result else "retained_encoded"
            )
            designed.append(DesignedPosition(c, source, source, k, basis, (), (), nothing))
            continue
        definition = get_component(result, library=library)
        check = replacement_admissibility(system, c, definition)
        broken = (check.outcome == Admissibility.INADMISSIBLE) & ~np.isin(check.edge, policy.break_connections)
        if broken.any():
            fail("design_breaks_connection", StructureRef(component=c), edge=int(check.edge[broken][0]))
        if (check.outcome == Admissibility.UNDETERMINED).any():
            undetermined.append(c)
        cross = np.flatnonzero((atom_component[first] == c) != (atom_component[second] == c))
        port = s["component_port"][c].astype(np.int64)
        first_linked, second_linked = (
            bool(p >= 0 and ((first[cross] == p) | (second[cross] == p)).any()) for p in port.tolist()
        )
        rows = instantiate_component(definition, linked=(first_linked, second_linked))
        guide = policy.guide.names.get(MonomerType(int(CLASS_MONOMER[encoded[c]])), frozenset())
        kept_edges = cross[~np.isin(cross, check.edge[check.outcome == Admissibility.INADMISSIBLE])]
        ends = np.where(atom_component[first[kept_edges]] == c, first[kept_edges], second[kept_edges])
        externals = s["external_atom"][atom_component[s["external_atom"]] == c].astype(np.int64)
        missing = externals[~np.isin(names[externals], rows.names)]
        if len(missing):
            fail("design_breaks_connection", StructureRef(component=c, atom=int(missing[0])), external=True)
        wanted = set(guide) | set(names[ends].tolist()) | set(names[externals].tolist())
        by_name = {str(names[a]): int(a) for a in atoms}
        retained = np.array([by_name.get(n, -1) if n in wanted else -1 for n in rows.names], dtype=np.int64)
        gone = cross[~np.isin(np.where(atom_component[first[cross]] == c, first[cross], second[cross]), retained)]
        remove += gone.tolist()
        guide_atoms = atoms[np.isin(names[atoms], list(guide)) & np.isin(atoms, retained)]
        keep[guide_atoms] = True
        rows_of[c], retained_of[c] = rows, retained
        withdrawn = tuple(names[atoms[~np.isin(atoms, retained)]].tolist())
        designed.append(
            DesignedPosition(c, source, result, k, "replaced", tuple(names[guide_atoms].tolist()), withdrawn, check)
        )
    if not rows_of:
        structure, report = target, _empty_report()
    else:
        edit = ChemicalEdit(
            remove_atoms=np.zeros(0, dtype=np.int64),
            replace_components=rows_of,
            retained_atoms=retained_of,
            remove_edges=np.unique(np.array(remove, dtype=np.int64)),
            add_edges=_no_edges(),
            set_atoms={},
            claims={c: rows.claims for c, rows in rows_of.items()},
            keep_coordinates=keep,
            identity=policy.identity,
        )
        structure, _atom_map, report = apply_edit(target, edit)
    _, losses = model_tokens(structure.system, token_layout(structure.system, table))
    lossy = np.isin(losses.component, np.asarray(positions, dtype=np.int64))
    losses = EncodingLoss(losses.component[lossy], losses.canonical[lossy], losses.encoding[lossy])
    undetermined_at = np.array(undetermined, dtype=np.int64)
    record = DesignRecord(
        tuple(designed), report, losses, component_records("replacement_endpoint_undetermined", undetermined_at)
    )
    return structure, record
