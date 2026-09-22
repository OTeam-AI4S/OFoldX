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
The one read driver: source → typed tables → definitions → residue table → chemical state →
symmetry → attachment inputs → attachment chemistry → the one lowering → instances, observations, identity and copies →
one construction → ``(Structure, ReadReport)``.

Each stage returns an immutable value and never re-decides a predecessor's output. Read options decide only which
molecule is read (model, assembly, alternative conformers, extra component definitions, conflict policy); chains, water
and hydrogens are a projection after the read (``Structure.select``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, NoReturn, cast

import numpy as np
from numpy.typing import NDArray

from ..chemistry.attachments import AttachmentResult, resolve_attachments
from ..chemistry.declarations import AssumptionSet, AttachmentInputs
from ..chemistry.definitions import (
    ArtifactProvenance,
    ChemicalComponent,
    ComponentLibrary,
    configured_library,
    resolve_definitions,
)
from ..chemistry.realize import GraphArrays, OccurrenceLayout, graph_records, realize_graph
from ..data.assembly import replicate_arrays
from ..data.molecular import EndpointState, MolecularSystem, SystemArrays
from ..data.structure import (
    CoordinateEvidence,
    CopyId,
    CopyKind,
    OperationNumbering,
    ResolutionKind,
    Structure,
    StructureArrays,
)
from ..errors import InvalidRequestError, RequestRef, SourceRef, StructureRef, fail
from ..provenance import ALLOWED_STATES, Basis, Records, Resolution, SourceRefs, SubjectKind, known, state
from .assemblies import assembly_copies
from .binding import NO_CODE, AltlocPolicy, ResidueTable, chemical_state, residue_table, stated_identity
from .connections import attachment_inputs
from .mmcif.decode import private_rows
from .report import AppliedSelection, ReadReport, SourceInfo
from .source import Source, read_source
from .symmetry import STABILISER_RULE, CrystalFrame, Symmetry, cell_state, symmetry
from .tables import SourceTables

#: The reader's rule version, recorded in ``SourceInfo.reader_version`` (with the stabiliser rule version and the
#: assumption rule version).
READER_VERSION = f"read/1 {STABILISER_RULE} {AssumptionSet.VERSION}"

_KNOWN_DECLARED = state(Resolution.KNOWN, Basis.DECLARED)
_UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
_INAPPLICABLE_DEFINED = state(Resolution.INAPPLICABLE, Basis.DEFINED)
_DETERMINED = (
    EndpointState.DECLARED_BY_SOURCE,
    EndpointState.DERIVED_FROM_REFERENCE,
    EndpointState.DERIVED_FROM_VALENCE,
    EndpointState.INAPPLICABLE,
)
# the readings of rows naming one atom pair that the source cannot tell apart
_READINGS = (StructureRef(item="one physical edge, restated"), StructureRef(item="distinct physical edges"))


def read_structure(
    source: str | os.PathLike[str] | Source,
    *,
    model: int | None = None,
    assembly: str | None = None,
    altloc: AltlocPolicy = "highest_occupancy",
    components: Mapping[str, ChemicalComponent] | None = None,
    on_conflict: Literal["raise", "record"] = "raise",
    library: ComponentLibrary | None = None,
    assumptions: AssumptionSet = AssumptionSet.EMPTY,
) -> tuple[Structure, ReadReport]:
    r"""
    Read one structure.

    Args:
        source: A path or a ``Source`` (parse-once reuse; an ensemble is one read per ``source.models`` entry).
        model: The model number; the first model when None.
        assembly: The ``_pdbx_struct_assembly`` id (``REMARK 350`` biomolecule) whose copies to build; the deposited
            native instances when None.
        altloc: ``"highest_occupancy"`` or one altloc label (``io.binding.chemical_state``).
        components: Caller definitions at archive scope, e.g. ``read_smiles(...).component``.
        on_conflict: ``"raise"``: any Conflict record raises ``ContradictionError(read_conflicts)`` carrying every
            Conflict record and the report, then any ``ambiguous_connection`` record whose rows can be covalent (its
            readings can differ in the covalent graph or a budget; a coordination-only pair stays a record) raises
            ``InsufficientInformationError(read_ambiguous_connections)`` naming the atoms, the source rows and the
            readings and carrying the records and the report; ``"record"``: the conflicting and ambiguous parts are
            undetermined and recorded in the report.
        library: The versioned CCD artifact; ``configured_library()`` when None.
        assumptions: Consumer assumption rules.

    Raises:
        InvalidRequestError: ``value_invalid`` for an ``on_conflict`` that is neither ``"raise"`` nor ``"record"``.
    """

    if on_conflict not in ("raise", "record"):
        fail("value_invalid", RequestRef("on_conflict", str(on_conflict)))
    source = source if isinstance(source, Source) else read_source(source)
    library = configured_library() if library is None else library
    caller = MappingProxyType(dict(components or {}))
    model = source.models[0] if model is None else model
    structure, report = _read(source, model, assembly, altloc, assumptions, library, caller)
    conflicts, ambiguous = report.conflicts(), report.records.where(code="ambiguous_connection")
    if on_conflict == "raise" and len(conflicts):
        fail("read_conflicts", conflicts=tuple(conflicts[k] for k in range(len(conflicts))), report=report)
    found = [ambiguous[k] for k in range(len(ambiguous)) if ambiguous[k].detail["covalent"]]
    if on_conflict == "raise" and found:
        atoms = sorted(
            {atom for record in found for atom in cast(tuple[int, ...], record.detail["atoms"]) if atom >= 0}
        )
        sources = dict.fromkeys(source for record in found for source in record.sources)
        refs = [StructureRef(atom=atom) for atom in atoms] + list(sources)
        fail("read_ambiguous_connections", *refs, candidates=_READINGS, connections=tuple(found), report=report)
    return structure, report


def _read(
    source: Source,
    model: int,
    assembly: str | None,
    altloc: AltlocPolicy,
    assumptions: AssumptionSet,
    library: ComponentLibrary,
    caller: Mapping[str, ChemicalComponent],
) -> tuple[Structure, ReadReport]:
    tables = source.tables(model)
    residues = residue_table(tables, model)
    comp = tables.atoms.symbols.get("comp", ())
    ids = {comp[k] for k in np.unique(residues.component).tolist()}
    definitions = resolve_definitions(ids, tables.definitions, library, caller)
    state_ = chemical_state(residues, tables.atoms, altloc=altloc)
    crystal = symmetry(tables.crystal, tables.experiment, tables.refs)
    # the attachment inputs, the declaration row of each struct_conn/SSBOND/LINK row and the records of the stage
    inputs, layout, declaration, stage = attachment_inputs(tables, residues, state_, crystal, definitions, assumptions)
    result = resolve_attachments(inputs)
    graph = realize_graph(inputs, result, layout)
    _configurations(source, graph.system)
    *copies, open_, restored = _copies(source, layout.instance_symbol)
    complete = graph.system.columns["instance_connections_complete"]
    graph.system.columns["instance_connections_complete"] = complete & ~open_

    # observations, identity, frame, copies
    observed = _structure(source, tables, inputs, layout, graph, residues, crystal, model, copies)
    log = _leaving_flags(tables, inputs, result, graph, declaration)
    records = Records.concat(
        [
            tables.records,
            residues.records,
            state_.records,
            crystal.records,
            _in_block(definitions.records, tables.refs.block),
            graph_records(stage, graph),
            graph.records,
            Records.from_rows(log, inputs.refs),
            restored,
            _library(source, cast(ArtifactProvenance, definitions.library)),  # The public read uses a versioned CCD.
        ]
    )
    system_arrays, structure_arrays = graph.system, observed
    if assembly is not None:
        system_arrays, structure_arrays, records = _assembly(tables, crystal, graph.system, observed, assembly, records)

    try:  # a value the read builds that construction refuses is a reader defect
        system = MolecularSystem.from_arrays(system_arrays)
        structure = Structure.from_arrays(system, structure_arrays)
    except InvalidRequestError as error:
        raise AssertionError(f"the read built an invalid value: {error}") from error

    report = ReadReport(
        source=SourceInfo(source.path, source.sha256, source.format, source.block, READER_VERSION),
        selection=AppliedSelection(model, assembly, altloc, assumptions),
        metadata=source.metadata,
        definitions=definitions,
        structure_digest=structure.digest(),
        records=records,
        bulk=MappingProxyType({"standard_junction_endpoint": int(result.leaving.standard.sum())}),
        refs=inputs.refs,
    )
    return structure, report


def _in_block(records: Records, block: str | None) -> Records:
    r"""``records`` with their reference table in ``block``: artifact and caller references name no block of their own
    (``SourceRefs`` holds one block per table)."""
    refs = records.refs
    if refs.block == block or not len(refs):
        return records
    moved = SourceRefs(refs.category, refs.row, refs.item, refs.categories, refs.items, block)
    return Records(
        records.code,
        records.subject_kind,
        records.subject,
        records.ref_ptr,
        records.ref,
        moved,
        records.payload,
        records.list_ptr,
        records.list_value,
        records.texts,
    )


def _rows(tables: SourceTables, inputs: AttachmentInputs, layout: OccurrenceLayout, graph: GraphArrays, n: int):
    r"""Per graph atom its ``AtomTable`` position (-1 for an atom with no row: materialised from the definition)."""
    atoms = tables.atoms
    order = np.argsort(atoms.row, kind="stable")
    position = np.full(n, -1, dtype=np.int64)
    refs = inputs.refs.row
    placed = (graph.occurrence_atom >= 0) & (inputs.occurrences.atom_ref >= 0)
    rows = refs[inputs.occurrences.atom_ref[placed]]
    position[graph.occurrence_atom[placed]] = order[np.searchsorted(atoms.row[order], rows)]
    extra = refs[layout.extra_ref]
    position[graph.extra_atom] = order[np.searchsorted(atoms.row[order], extra)]
    return position


def _structure(
    source: Source,
    tables: SourceTables,
    inputs: AttachmentInputs,
    layout: OccurrenceLayout,
    graph: GraphArrays,
    residues: ResidueTable,
    crystal: Symmetry,
    model: int,
    copies: list,
) -> StructureArrays:
    r"""
    The structure columns of the native value: coordinates OBSERVED where all three are stated, with
    their printed unit where they stay OBSERVED in a deposited instance (``_ofoldx_coordinate_evidence`` names other
    evidence, whose atoms carry no occupancy or B), occupancy, B factor, the raw ``pdbx_formal_charge`` cell and the
    altloc of each atom's row (materialised atoms carry none: evidence NONE, occupancy and B INAPPLICABLE×DEFINED);
    the residue identity binding decided (sequence number, author text with its integer reading, insertion,
    author chain; PDB label asyms unknown); the deposited instances or the copies ``_ofoldx_instance``
    names; the model, the source digest, method, resolution, the cell (with its printed unit) and the frame (a DECLARED
    one with the unit its bounds used: printed, or for a frame inverted from a ``Cartn_transf`` group propagated), the
    space group and the numbering in force with the file's own table (an ITA list is not stored).
    """

    a, s = tables.atoms, graph.system.columns
    n = len(s["atom_symbol"])
    row = _rows(tables, inputs, layout, graph, n)
    has = row >= 0
    r = row[has]
    columns: dict[str, NDArray] = {}
    coordinates, evidence, quantum = np.zeros((n, 3)), np.zeros(n, dtype=np.uint8), np.zeros(n)
    stated = np.flatnonzero(has)[known(a.coordinate_state[r])]
    coordinates[stated], evidence[stated] = a.coordinates[row[stated]], CoordinateEvidence.OBSERVED
    quantum[stated] = 2 * a.coordinate_quantum[row[stated]].astype(np.float64)  # the printed unit
    _evidence(source, np.append(a.row, -1)[row], evidence)  # an atom with no row (-1): the appended -1
    copy = np.repeat(np.repeat(copies[0], np.diff(s["instance_component_ptr"])), np.diff(s["component_atom_ptr"]))
    quantum[(evidence != CoordinateEvidence.OBSERVED) | (copy != CopyKind.DEPOSITED)] = 0  # values no source printed
    columns["coordinates"], columns["coordinate_evidence"] = coordinates, evidence
    columns["coordinate_quantum"] = quantum
    unobserved = evidence > CoordinateEvidence.OBSERVED  # no occupancy or B of an unobserved atom
    for name in ("occupancy", "occupancy_uncertainty", "b_factor"):
        values, states = np.zeros(n), np.full(n, _INAPPLICABLE_DEFINED, dtype=np.uint8)
        values[has] = np.where(known(getattr(a, f"{name}_state")[r]), getattr(a, name)[r], 0)
        states[has] = getattr(a, f"{name}_state")[r]
        values[unobserved], states[unobserved] = 0, _INAPPLICABLE_DEFINED
        columns[name], columns[f"{name}_state"] = values, states
    charge, charge_state = np.zeros(n, dtype=np.int8), np.full(n, _UNKNOWN, dtype=np.uint8)
    charge[has], charge_state[has] = np.where(known(a.charge_state[r]), a.charge[r], 0), a.charge_state[r]
    columns["atom_site_charge"], columns["atom_site_charge_state"] = charge, charge_state
    altloc = np.full(n, -1, dtype=np.int32)
    altloc[has] = np.where(known(a.label_alt_state[r]), a.label_alt[r], -1)
    columns["atom_altloc"] = altloc

    ordinal = inputs.occurrences.ordinal
    number_state = residues.number_state[ordinal]
    columns["component_sequence_number"] = np.where(known(number_state), residues.number[ordinal], 0)
    columns["component_sequence_number_state"] = number_state
    columns["component_author_text"] = residues.author_text[ordinal]
    columns["component_author_number"] = residues.author_number[ordinal]
    columns["component_author_number_state"] = residues.author_state[ordinal]
    columns["component_insertion"] = residues.insertion[ordinal]
    columns["instance_author_chain"] = residues.instance_author_chain
    columns["instance_author_chain_state"] = residues.instance_author_chain_state
    columns["instance_copy_kind"], columns["instance_copy_operators"], columns["instance_source_asym"] = copies[:3]

    columns["source_sha256"] = np.frombuffer(bytes.fromhex(source.sha256), dtype=np.uint8)[None]
    columns["system_model_number"] = np.array([model])
    columns["system_model_number_state"] = np.array([tables.model_state])
    columns["system_model_number"] *= known(columns["system_model_number_state"])
    stated_cell = cell_state(tables.crystal)  # the cell and the frame are two statements
    stated = known(np.uint8(stated_cell))
    columns["cell"] = (tables.crystal.cell if stated else np.zeros(6))[None]
    columns["cell_state"] = np.array([stated_cell])
    columns["cell_quantum"] = (2 * tables.crystal.cell_quantum if stated else np.zeros(6))[None]
    frame = crystal.frame
    frame_state = frame.state if frame is not None else _UNKNOWN
    framed = frame is not None and bool(np.isfinite(frame.fractionalisation).all())  # KNOWN, or INAPPLICABLE on [A | u]
    columns["frame"] = (cast(CrystalFrame, frame).fractionalisation if framed else np.zeros((3, 4)))[None]
    columns["frame_state"] = np.array([frame_state])
    printed = framed and frame_state in (_KNOWN_DECLARED, _INAPPLICABLE_DEFINED)  # the source's transform: its unit
    columns["frame_quantum"] = (2 * cast(CrystalFrame, frame).quantum if printed else np.zeros((3, 4)))[None]
    columns["space_group"] = np.array([crystal.space_group])
    columns["space_group_state"] = np.array([crystal.space_group_state])
    numbering = {"file": OperationNumbering.FILE, "ita": OperationNumbering.ITA}.get(crystal.numbering)
    operations = crystal.operations if numbering == OperationNumbering.FILE else ()
    columns["operation_numbering"] = np.array([numbering or OperationNumbering.NONE])
    columns["operation_ptr"] = np.array([0, len(operations)])
    columns["operation_id"] = np.arange(len(operations))
    columns["operation_rotation"] = np.array([op.rotation for op in operations]).reshape(-1, 3, 3)
    columns["operation_translation"] = np.array([op.translation for op in operations]).reshape(-1, 3)

    experiment = tables.experiment
    methods = np.flatnonzero((experiment.resolution_kind == ResolutionKind.NONE) & (experiment.method >= 0))
    resolutions = np.flatnonzero(
        (experiment.resolution_kind != ResolutionKind.NONE) & (experiment.resolution_state != _UNKNOWN)
    )
    texts = a.symbols.get("text", ())
    method_texts = tuple(dict.fromkeys(texts[k] for k in experiment.method[methods].tolist()))
    columns["system_method_ptr"] = np.array([0, len(method_texts)])
    columns["method"] = np.arange(len(method_texts))
    columns["system_resolution_ptr"] = np.array([0, len(resolutions)])
    resolution_state = experiment.resolution_state[resolutions]
    columns["resolution"] = np.where(known(resolution_state), experiment.resolution[resolutions], 0)
    columns["resolution_kind"] = experiment.resolution_kind[resolutions]
    columns["resolution_state"] = resolution_state
    symbols = {
        "asym": copies[3],
        "chain": a.symbols.get("chain", ()),
        "author_number": a.symbols.get("number", ()),
        "insertion": a.symbols.get("ins", ()),
        "altloc": a.symbols.get("alt", ()),
        "operator": copies[4],
        "method": method_texts,
        "space_group": tables.crystal.symbols.get("space_group", ()),
        "operation": crystal.operation_ids if numbering == OperationNumbering.FILE else (),
    }
    return StructureArrays(symbols, columns)


def _private(source: Source, name: str) -> list[dict]:
    return private_rows(source.categories, name, source.block)


def _invalid(name: str, row: int, source: Source) -> NoReturn:
    fail("private_row_invalid", SourceRef(name, row, None, source.block))


def _evidence(source: Source, source_row: NDArray[np.int64], evidence: NDArray[np.uint8]) -> None:
    r"""``_ofoldx_coordinate_evidence``: the evidence of each named atom whose coordinates are stated."""
    rows = _private(source, "ofoldx_coordinate_evidence")
    if not rows:
        return
    ids = source.categories["atom_site"].text("id")[0].tolist()
    source_atoms = {int(i): atom for atom, i in enumerate(source_row.tolist()) if i >= 0}
    atom_of = {ids[row].decode(): atom for row, atom in source_atoms.items()}
    for k, entry in enumerate(rows):
        atom = atom_of.get(str(entry["atom_site_id"]), -1)
        if atom < 0 or evidence[atom] != CoordinateEvidence.OBSERVED or not (entry["evidence"] or 0) > 0:
            _invalid("ofoldx_coordinate_evidence", k, source)
        evidence[atom] = entry["evidence"]


def _library(source: Source, library: ArtifactProvenance) -> Records:
    r"""``_ofoldx_library``: the CCD artifact the file was written against. One of other content
    than ``library`` is a finding naming both; what the file leaves to the library is derived from ``library``."""
    rows = _private(source, "ofoldx_library")
    for k, entry in enumerate(rows):
        if k or None in entry.values():
            _invalid("ofoldx_library", k, source)
    if not rows or rows[0]["sha256"] == library.sha256:
        return Records.empty()
    written = rows[0]
    payload = (written["content_version"], written["sha256"], library.content_version, library.sha256)
    refs = SourceRefs(
        np.zeros(1, np.int16),
        np.zeros(1, np.int64),
        np.full(1, -1, np.int16),
        ("ofoldx_library",),
        (),
        source.block,
    )
    return Records.from_rows([("written_library_differs", SubjectKind.NONE, -1, [0], payload)], refs)


def _copies(source: Source, instances: tuple[str, ...]) -> tuple:
    r"""
    Copy kind, operator expression and source asym of each instance with the ``asym`` and ``operator`` symbols, and
    the instances whose connections a row states open with their records: an mmCIF instance is the deposited copy of
    its asym, PDB names none; ``_ofoldx_instance`` rows restore the copies that are not and the connections a
    written value did not all represent (a row stating no ``connections`` is malformed; without a row completeness is
    derived from the file's own rows).
    """
    mmcif = source.format == "mmcif"
    kind = np.full(len(instances), CopyKind.DEPOSITED, dtype=np.uint8)
    operators = np.full(len(instances), -1, dtype=np.int32)
    asym = np.arange(len(instances), dtype=np.int32) if mmcif else np.full(len(instances), -1, dtype=np.int32)
    asyms = {symbol: k for k, symbol in enumerate(instances)} if mmcif else {}  # symbol -> index
    expressions: dict[str, int] = {}
    kinds = {k.name.lower(): k for k in CopyKind}
    index, restored = {symbol: k for k, symbol in enumerate(instances)}, set()
    open_, log = np.zeros(len(instances), dtype=bool), []
    rows = _private(source, "ofoldx_instance")
    for k, entry in enumerate(rows):
        i, copy = index.get(entry["asym_id"]), kinds.get(entry["copy_kind"])
        if i is None or copy is None or i in restored or entry["connections"] is None:
            _invalid("ofoldx_instance", k, source)
        restored.add(i)
        if entry["connections"] == 2:  # open
            open_[i] = True
            log.append(("connections_open_restored", SubjectKind.INSTANCE, i, [k], ()))
        kind[i] = copy
        if entry["operators"] is not None:
            symbol = CopyId(copy, tuple(entry["operators"].split(",")), None).operator_symbol
            operators[i] = expressions.setdefault(symbol, len(expressions))
        source_asym = entry["source_asym"]
        asym[i] = -1 if source_asym is None else asyms.setdefault(source_asym, len(asyms))
    n = len(rows)
    refs = SourceRefs(
        np.zeros(n, np.int16),
        np.arange(n),
        np.full(n, -1, np.int16),
        ("ofoldx_instance",),
        (),
        source.block,
    )
    return kind, operators, asym, tuple(asyms), tuple(expressions), open_, Records.from_rows(log, refs)


def _configurations(source: Source, system: SystemArrays) -> None:
    r"""``_ofoldx_atom_parity`` and ``_ofoldx_bond_stereo``: the stated configuration of each named stereocentre
    and stereobond of the value, which its written definition does not reproduce."""
    parities, bonds = _private(source, "ofoldx_atom_parity"), _private(source, "ofoldx_bond_stereo")
    if not parities and not bonds:
        return
    c = system.columns
    instance = np.repeat(np.arange(len(c["instance_component_ptr"]) - 1), np.diff(c["instance_component_ptr"]))
    component = np.repeat(np.arange(len(c["component_atom_ptr"]) - 1), np.diff(c["component_atom_ptr"]))
    ordinal = np.arange(len(instance)) - c["instance_component_ptr"][instance]
    names, instances = system.symbols["atom"], system.symbols["instance"]
    sites = {
        (instances[c["instance_symbol"][instance[k]]], int(ordinal[k]), names[symbol]): atom
        for atom, (k, symbol) in enumerate(zip(component.tolist(), c["atom_symbol"].tolist()))
    }

    def atom(entry: dict, prefix: str, name: str, row: int) -> int:
        site = (entry[f"{prefix}asym_id"], entry[f"{prefix}component"], entry[f"{prefix}atom_id"])
        if site == (None, None, None) and prefix:  # the implicit slot
            return -1
        if site not in sites:
            _invalid(name, row, source)
        return sites[site]

    centres = {a: k for k, a in enumerate(c["stereocentre_atom"].tolist())}
    for name in ("stereocentre_parity", "stereocentre_neighbors", "stereocentre_state"):
        c[name] = c[name].copy()
    restored: set[int] = set()  # each centre once
    for k, entry in enumerate(parities):
        at = centres.get(atom(entry, "", "ofoldx_atom_parity", k))
        if at is None or at in restored or entry["state"] not in ALLOWED_STATES:
            _invalid("ofoldx_atom_parity", k, source)
        restored.add(at)
        c["stereocentre_neighbors"][at] = [atom(entry, f"neighbor_{j}_", "ofoldx_atom_parity", k) for j in (1, 2, 3, 4)]
        c["stereocentre_parity"][at], c["stereocentre_state"][at] = entry["parity"] or 0, entry["state"]

    first, second = c["edge_index"].astype(np.int64)
    edge = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(first, second))}
    stereo = {int(e): k for k, e in enumerate(c["stereobond_edge"].tolist())}
    for name in ("stereobond_value", "stereobond_reference", "stereobond_state"):
        c[name] = c[name].copy()
    restored = set()  # each bond once
    for k, entry in enumerate(bonds):
        ends = [atom(entry, f"end_{j}_", "ofoldx_bond_stereo", k) for j in (1, 2)]
        references = [atom(entry, f"reference_{j}_", "ofoldx_bond_stereo", k) for j in (1, 2)]
        at = stereo.get(edge.get((min(ends), max(ends)), -1))
        if at is None or at in restored or entry["state"] not in ALLOWED_STATES:
            _invalid("ofoldx_bond_stereo", k, source)
        restored.add(at)
        c["stereobond_reference"][at] = references if ends[0] < ends[1] else references[::-1]
        c["stereobond_value"][at], c["stereobond_state"][at] = entry["value"] or 0, entry["state"]


def _leaving_flags(tables, inputs, result: AttachmentResult, graph: GraphArrays, declarations) -> list[tuple]:
    r"""``leaving_flag_inconsistent``: a ``struct_conn`` row whose advisory ``pdbx_leaving_atom_flag``
    (none / one / both) disagrees with the number of its endpoints at which atoms leave, where both are determined."""
    c, decl, leaving = tables.connections, inputs.declarations, result.leaving
    stated = {int(e): k for k, e in enumerate(leaving.endpoint.tolist())}
    log = []
    for k in np.flatnonzero(c.leaving_flag > 0).tolist():
        declaration = int(declarations[k])
        if declaration < 0:
            continue
        rows = [stated.get(int(e)) for e in (decl.first[declaration], decl.second[declaration])]
        if any(r is None or leaving.state[r] not in _DETERMINED for r in rows):
            continue
        determined_rows = cast(list[int], rows)  # The preceding check rejected absent or undetermined endpoints.
        sides = sum(int(leaving.atoms_ptr[r + 1] > leaving.atoms_ptr[r]) for r in determined_rows)
        if sides != int(c.leaving_flag[k]) - 1:
            target = int(graph.edge_of_declaration[declaration])
            kind, subject = (SubjectKind.EXTERNAL, -2 - target) if target <= -2 else (SubjectKind.EDGE, target)
            log.append(("leaving_flag_inconsistent", kind, subject, [int(c.ref[k])], ()))
    return log


def _assembly(tables, crystal, system: SystemArrays, observed: StructureArrays, assembly, records):
    r"""
    The requested assembly: its ordered copies (``io.assemblies.assembly_copies``) realised by
    the one copy join (``data.assembly.replicate_arrays``). The native instances are kept only as the copy under an
    identity operator; subjects of native parts move to that copy, and keep -1 where it does not exist.
    """

    a = tables.atoms
    if tables.refs.block is not None:  # mmCIF: an instance is its asym
        keys = [(symbol, k) for k, symbol in enumerate(system.symbols["instance"])]
    else:  # PDB: the instances of an author chain; a blank chain's id is blank (a REMARK 350 list naming no chain)
        chains, columns = a.symbols.get("chain", ()), observed.columns
        author = stated_identity(columns["instance_author_chain"], columns["instance_author_chain_state"])
        keys = [(chains[code] if code >= 0 else "", k) for k, code in enumerate(author.tolist()) if code != NO_CODE]
    instances: dict[str, list[int]] = {}
    for key, k in keys:
        instances.setdefault(key, []).append(k)
    copies, operator_records = assembly_copies(
        crystal,
        tables.operators,
        tables.assemblies,
        a.symbols,
        tables.refs,
        assembly,
        {k: np.array(v) for k, v in instances.items()},
    )
    out_system, out_structure, replica = replicate_arrays(system, observed, copies, keep_source=False)

    identity = np.array([np.allclose(copy.matrix, np.eye(4), atol=1e-5) for copy in copies] + [False])
    native, out = system.columns, out_system.columns
    atom_map = np.full(len(native["atom_symbol"]), -1, dtype=np.int64)
    kept = identity[replica.atom_copy]
    atom_map[replica.atom_source[kept]] = np.flatnonzero(kept)
    instance_map = np.full(len(native["instance_symbol"]), -1, dtype=np.int64)
    kept = identity[replica.instance_copy]
    instance_map[replica.instance_source[kept]] = np.flatnonzero(kept)
    component_map = np.full(len(native["component_symbol"]), -1, dtype=np.int64)
    native_ptr, out_ptr = native["instance_component_ptr"], out["instance_component_ptr"]
    for source, target in zip(np.flatnonzero(instance_map >= 0).tolist(), instance_map[instance_map >= 0].tolist()):
        count = native_ptr[source + 1] - native_ptr[source]
        component_map[native_ptr[source] : native_ptr[source + 1]] = out_ptr[target] + np.arange(count)
    first, second = native["edge_index"].astype(np.int64)
    width = max(len(out["atom_symbol"]), 1)
    keys_out = out["edge_index"][0].astype(np.int64) * width + out["edge_index"][1]
    lo, hi = np.minimum(atom_map[first], atom_map[second]), np.maximum(atom_map[first], atom_map[second])
    found = np.clip(np.searchsorted(keys_out, lo * width + hi), 0, max(len(keys_out) - 1, 0))
    hit = (lo >= 0) & (keys_out[found] == lo * width + hi) if len(keys_out) else np.zeros(len(lo), bool)
    edge_map = np.where(hit, found, -1)
    external_map = _external_map(native, out, out_system.symbols, system.symbols, atom_map)

    maps = {
        SubjectKind.ATOM: atom_map,
        SubjectKind.COMPONENT: component_map,
        SubjectKind.INSTANCE: instance_map,
        SubjectKind.EDGE: edge_map,
        SubjectKind.EXTERNAL: external_map,
    }
    for kind, old_to_new in maps.items():
        records = records.remap(kind, old_to_new)
    return out_system, out_structure, Records.concat([records, operator_records])


def _external_map(native, out, out_symbols, native_symbols, atom_map) -> NDArray[np.int64]:
    r"""Native external row → the external row of the identity copy that restates it exactly (atom, partner and
    transform), -1 where none or several do (a join made it an edge, or the copy does not exist)."""

    def keys(columns, symbols, atoms):
        rotation, translation = columns["transform_rotation"], columns["transform_translation"]
        transform = columns["external_transform"]
        return [
            (
                int(atom),
                symbols["instance"][instance],
                int(component),
                symbols["atom"][name],
                rotation[t].tobytes(),
                translation[t].tobytes(),
            )
            for atom, instance, component, name, t in zip(
                atoms,
                columns["external_partner_instance"].tolist(),
                columns["external_partner_component"].tolist(),
                columns["external_partner_atom"].tolist(),
                transform.tolist(),
            )
        ]

    found: dict[tuple, list[int]] = {}
    for k, key in enumerate(keys(out, out_symbols, out["external_atom"].tolist())):
        found.setdefault(key, []).append(k)
    mapped = atom_map[native["external_atom"].astype(np.int64)].tolist()
    return np.array(
        [
            found[key][0] if len(found.get(key, ())) == 1 and key[0] >= 0 else -1
            for key in keys(native, native_symbols, mapped)
        ],
        dtype=np.int64,
    )
