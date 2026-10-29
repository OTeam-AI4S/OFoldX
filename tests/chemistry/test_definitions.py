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

from __future__ import annotations

import hashlib
import json
import pickle
import tomllib
from dataclasses import fields, replace
from types import MappingProxyType

import numpy as np
import pytest

from ofoldx.chemistry.definitions import (
    ComponentItems,
    DefinitionRows,
    StereoBasis,
    configured_library,
    get_component,
    load_library,
    resolve_definitions,
    write_artifact,
)
from ofoldx.chemistry.fragments import PortCondition, PortStatus, ports
from ofoldx.chemistry.rdkit import StereoArbitration
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import DefinitionScope, ReferenceSource
from ofoldx.errors import ContradictionError, InsufficientInformationError, InvalidRequestError, UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

from .builders import SUBSET, edit_atom, provenance

STATED, NULL = state(Resolution.KNOWN, Basis.REFERENCE), state(Resolution.UNKNOWN, Basis.REFERENCE)
DEFAULT, ABSENT = state(Resolution.KNOWN, Basis.DEFAULT), state(Resolution.UNKNOWN, Basis.NONE)
CONFLICT = state(Resolution.CONFLICT, Basis.NONE)
SUBSET_IDS = tomllib.loads((SUBSET.parent / "MANIFEST.toml").read_text())["blocks"]


def rows(**components: ComponentItems) -> DefinitionRows:
    return DefinitionRows(MappingProxyType(components))


def records(definitions) -> list:
    return [definitions.records[i] for i in range(len(definitions.records))]


def codes(definitions) -> list[str]:
    return sorted(record.code for record in records(definitions))


def withheld(definitions, component_id: str) -> bool:
    return component_id not in definitions.by_id


class TestDefine:
    @pytest.mark.parametrize(
        ("token", "charge", "charge_state"),
        [(1.0, 1, STATED), ("?", 0, NULL), (".", 0, NULL), (("default", 0.0), 0, DEFAULT), (None, 0, ABSENT)],
        ids=["stated", "unknown", "inapplicable", "default", "absent"],
    )
    def test_charge_states(self, items, glycine, define, token, charge, charge_state):
        columns = glycine()
        columns["atoms"]["charge"] = [token] * 10
        definition = define(items(**columns))
        assert definition.charge.tolist() == [charge] * 10
        assert definition.charge_state.tolist() == [charge_state] * 10

    @pytest.mark.parametrize(
        ("ideal", "model", "source"),
        [
            ("all", "none", ReferenceSource.IDEAL),
            ("one-missing", "all", ReferenceSource.MODEL),
            ("one-missing", "one-missing", ReferenceSource.NONE),
        ],
    )
    def test_coherent_conformer(self, items, glycine, define, ideal, model, source):
        columns = glycine()
        atoms = columns["atoms"]
        for axis in "xyz":
            atoms[f"model_Cartn_{axis}"] = list(atoms[f"pdbx_model_Cartn_{axis}_ideal"])
            for item, mode in ((f"pdbx_model_Cartn_{axis}_ideal", ideal), (f"model_Cartn_{axis}", model)):
                if mode == "one-missing":
                    atoms[item][3] = "?"
                if mode == "none":
                    del atoms[item]
        assert define(items(**columns)).reference_source == source

    def test_stereocentre_rows(self, archive):
        alanine, sulfoxide = archive("ALA"), archive("SME")
        assert [alanine.atom_names[a] for a in alanine.stereo_centre] == ["CA"]
        assert alanine.stereo_parity.tolist() != [0] and alanine.stereo_basis.tolist() == [StereoBasis.CONFORMER]
        row = sulfoxide.stereo_centre.tolist().index(sulfoxide.atom_names.index("S"))
        assert sulfoxide.stereo_neighbors[row, 3] == -1  # degree 3: the implicit slot is appended

    def test_parity_needs_conformer(self, items, glycine, define):
        columns = glycine()
        columns["atoms"]["pdbx_stereo_config"][1] = "R"
        for axis in "xyz":
            del columns["atoms"][f"pdbx_model_Cartn_{axis}_ideal"]
        definition = define(items(**columns))
        assert (definition.stereo_centre.tolist(), definition.stereo_parity.tolist()) == ([1], [0])

    @pytest.mark.parametrize(
        ("token", "order", "order_state"),
        [("doub", 2, STATED), ("AROM", 0, NULL), ("?", 0, NULL), (None, 0, ABSENT)],
    )
    def test_bond_orders(self, items, define, token, order, order_state):
        bonds = {"atom_id_1": ["C1"], "atom_id_2": ["C2"], "value_order": [token], "pdbx_aromatic_flag": ["Y"]}
        definition = define(items({"id": "CC"}, atoms={"atom_id": ["C1", "C2"]}, bonds=bonds))
        assert (definition.bond_order.tolist(), definition.bond_order_state.tolist()) == ([order], [order_state])

    def test_aromaticity_not_inferred(self, items, define):
        bonds = {"atom_id_1": ["C1"], "atom_id_2": ["C2"], "value_order": ["AROM"], "pdbx_aromatic_flag": ["Y"]}
        definition = define(items({"id": "CC"}, atoms={"atom_id": ["C1", "C2"]}, bonds=bonds))
        assert definition.aromatic_state.tolist() == [ABSENT, ABSENT]

    def test_parents_split(self, items, define):
        definition = define(items({"id": "X", "mon_nstd_parent_comp_id": "ASP, GLY"}))
        assert (definition.parents, definition.parents_state) == (("ASP", "GLY"), STATED)

    def test_digest_ignores_scope(self, items, glycine, define):
        built = items(**glycine())
        assert define(built).digest == define(built, DefinitionScope.CALLER).digest

    @pytest.mark.parametrize(
        ("category", "item", "row", "token"),
        [("atoms", "charge", 0, 1.0), ("bonds", "value_order", 0, "DOUB"), ("component", "one_letter_code", None, "X")],
    )
    def test_digest_tracks_items(self, items, glycine, define, category, item, row, token):
        columns = glycine()
        if row is None:
            columns[category][item] = token
        else:
            columns[category][item][row] = token
        assert define(items(**columns)).digest != define(items(**glycine())).digest

    def test_dangling_bond_refused(self, items, glycine, define):
        columns = glycine()
        columns["bonds"]["atom_id_2"][0] = "ZZ"
        with pytest.raises(InvalidRequestError):
            define(items(**columns))


class TestResolveDefinitions:
    @pytest.fixture
    def gly(self, items, glycine, define, library):
        return library([define(items(**glycine()))])

    def test_file_overrides_archive(self, items, glycine, gly):
        columns = glycine()
        columns["component"]["one_letter_code"] = "X"
        definition = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly).by_id["GLY"]
        assert (definition.one_letter_code, definition.scope) == ("X", DefinitionScope.FILE)

    def test_file_null_keeps_archive(self, items, glycine, gly):
        columns = glycine()
        columns["component"]["one_letter_code"] = "?"
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        assert definitions.by_id["GLY"].one_letter_code == "G"
        (record,) = records(definitions)
        assert (record.code, dict(record.detail)) == (
            "file_scope_null_over_archive_value",
            {"id": "GLY", "item": "_chem_comp.one_letter_code"},
        )

    @pytest.mark.parametrize(
        ("item", "field"), [("pdbx_leaving_atom_flag", "leaving_flag"), ("type_symbol", "isotope")]
    )
    def test_stateless_null_recorded(self, items, glycine, gly, item, field):
        columns = glycine()
        columns["atoms"][item][4] = "?"  # OXT
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        assert np.array_equal(getattr(definitions.by_id["GLY"], field), getattr(gly.definition("GLY"), field))
        assert [(r.code, dict(r.detail)) for r in records(definitions)] == [
            ("file_scope_null_over_archive_value", {"id": "GLY", "item": f"_chem_comp_atom.{item}"})
        ]

    @pytest.mark.parametrize(
        ("category", "item"),
        [("atoms", "charge"), ("atoms", "pdbx_leaving_atom_flag"), ("component", "one_letter_code")],
    )
    def test_absent_item_unrecorded(self, items, glycine, gly, category, item):  # absence is no stated null
        columns = glycine()
        del columns[category][item]
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        found, archived = definitions.by_id["GLY"], gly.definition("GLY")
        assert (found.charge_state.tolist(), found.leaving_flag.tolist(), found.one_letter_code) == (
            archived.charge_state.tolist(),
            archived.leaving_flag.tolist(),
            archived.one_letter_code,
        )
        assert codes(definitions) == []

    def test_hydrogen_overrides_deuterium(self, items, glycine, define, library):
        deuterated = glycine()
        deuterated["atoms"]["type_symbol"][5] = "D"
        archive = library([define(items(**deuterated))])
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**glycine())), archive)
        assert (int(definitions.by_id["GLY"].isotope[5]), codes(definitions)) == (0, [])

    def test_definitions_pickle(self, items, glycine, gly):  # a read report carries them across processes
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**glycine())), gly)
        again = pickle.loads(pickle.dumps(definitions))
        assert again.by_id["GLY"].digest == definitions.by_id["GLY"].digest
        assert dict(again.refs) == dict(definitions.refs)

    @pytest.mark.parametrize("smiles", ["C[C@H](N)O", "F/C=C/F"], ids=["centre", "double-bond"])
    @pytest.mark.parametrize("path", ["chem-comp-row", "unified"])
    def test_caller_stated_stereo_kept(self, items, library, smiles, path):
        caller = read_smiles(smiles, component_id="LIG").component
        if path == "chem-comp-row":  # file scope states the _chem_comp row only (most wwPDB entries)
            found = resolve_definitions(
                ["LIG"], rows(LIG=items({"id": "LIG", "type": "NON-POLYMER"})), library([]), {"LIG": caller}
            )
        else:  # the library's entry and the caller's, one evidence class
            found = resolve_definitions(["LIG"], rows(), library([caller]), {"LIG": caller})
        definition = found.by_id["LIG"]
        assert definition.stereo_parity.tolist() == caller.stereo_parity.tolist()
        assert definition.bond_stereo_value.tolist() == caller.bond_stereo_value.tolist()

    @pytest.mark.parametrize(
        ("archived", "given"), [("C[C@H](N)O", "C[C@@H](N)O"), ("F/C=C/F", "F/C=C\\F")], ids=["centre", "double-bond"]
    )
    def test_stated_stereo_disagreement_withholds(self, library, archived, given):
        archive = library([read_smiles(archived, component_id="LIG").component])
        found = resolve_definitions(["LIG"], rows(), archive, {"LIG": read_smiles(given, component_id="LIG").component})
        assert withheld(found, "LIG") and "definition_conflict" in codes(found)

    def test_file_only_atom_merged(self, items, glycine, gly):
        columns = glycine()
        for values in columns["atoms"].values():
            values.append(values[-1])
        columns["atoms"]["atom_id"][-1] = "HX"
        for item, value in zip(columns["bonds"], ("OXT", "HX", "SING", "N", "N")):
            columns["bonds"][item].append(value)
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        assert definitions.by_id["GLY"].atom_names[-1] == "HX"
        assert "definition_atom_sets_differ" in codes(definitions)

    @pytest.mark.parametrize("stated", ["component", "atoms"])
    def test_file_states_own_tables(self, items, glycine, gly, stated):
        columns = glycine()
        file = items(columns["component"]) if stated == "component" else items(columns["component"], columns["atoms"])
        definitions = resolve_definitions(["GLY"], rows(GLY=file), gly)
        definition, archived = definitions.by_id["GLY"], gly.definition("GLY")
        assert (definition.atom_names, definition.bond_atoms.tolist()) == (
            archived.atom_names,
            archived.bond_atoms.tolist(),
        )
        assert (ports(definition), codes(definitions)) == (ports(archived), [])

    def test_type_case_insensitive(self, items, glycine, gly):  # wwPDB entries state "L-peptide linking"
        columns = glycine()
        columns["component"]["type"] = "L-peptide linking"
        definition = resolve_definitions(["GLY"], rows(GLY=items(columns["component"])), gly).by_id["GLY"]
        assert definition.digest == gly.definition("GLY").digest

    @pytest.mark.parametrize(("archived", "stated", "centres"), [("R", "N", []), ("N", "R", [1])])
    def test_stated_label_overrides(self, items, glycine, define, library, archived, stated, centres):
        archive, file = glycine(), glycine()
        archive["atoms"]["pdbx_stereo_config"][1], file["atoms"]["pdbx_stereo_config"][1] = archived, stated
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**file)), library([define(items(**archive))]))
        definition = definitions.by_id["GLY"]
        assert (definition.cip_declared[1], definition.stereo_centre.tolist()) == (stated, centres)
        assert codes(definitions) == []

    def test_caller_fills_library_gap(self, items, glycine, define, gly):
        columns = glycine()
        columns["component"]["id"] = "GLX"
        caller = {"GLX": define(items(**columns))}
        definitions = resolve_definitions(["GLX"], rows(), gly, caller)
        assert definitions.by_id["GLX"].scope == DefinitionScope.CALLER

    def test_caller_disagreement_conflicts(self, items, glycine, define, gly):
        columns = glycine()
        columns["atoms"]["charge"][0] = 1.0
        caller = {"GLY": define(items(**columns), DefinitionScope.CALLER)}
        definitions = resolve_definitions(["GLY"], rows(), gly, caller)
        assert definitions.by_id["GLY"].charge_state[0] == CONFLICT
        assert "definition_conflict" in codes(definitions)

    def test_duplicate_rows_conflict(self, items, glycine, gly):
        columns = glycine()
        for values in columns["atoms"].values():
            values.append(values[0])
        columns["atoms"]["charge"][-1] = 1.0
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        assert definitions.by_id["GLY"].charge_state[0] == CONFLICT
        (record,) = (r for r in records(definitions) if r.code == "definition_conflict")
        assert [(ref.category, ref.row) for ref in record.sources] == [("chem_comp_atom", 0), ("chem_comp_atom", 10)]

    def test_default_never_conflicts(self, items, glycine, define, library):
        stated, defaulted = glycine(), glycine()
        stated["atoms"]["charge"][0] = 1.0
        defaulted["atoms"]["charge"] = [("default", 0.0)] * 10
        caller = {"GLY": define(items(**defaulted), DefinitionScope.CALLER)}
        definitions = resolve_definitions(["GLY"], rows(), library([define(items(**stated))]), caller)
        definition = definitions.by_id["GLY"]
        assert (int(definition.charge[0]), int(definition.charge_state[0])) == (1, STATED)
        assert "definition_conflict" not in codes(definitions)

    @pytest.mark.parametrize("channel", ["file", "caller"])
    def test_isotope_disagreement_withholds(self, items, glycine, define, gly, channel):
        columns = glycine()
        if channel == "file":  # a duplicate H row stated as D
            for values in columns["atoms"].values():
                values.append(values[5])
            columns["atoms"]["type_symbol"][-1] = "D"
            definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        else:
            columns["atoms"]["type_symbol"][5] = "D"
            caller = {"GLY": define(items(**columns), DefinitionScope.CALLER)}
            definitions = resolve_definitions(["GLY"], rows(), gly, caller)
        assert withheld(definitions, "GLY") and codes(definitions) == ["definition_conflict"]

    def test_dangling_bond_withholds(self, items, glycine, gly):
        columns = glycine()
        columns["bonds"]["atom_id_2"][0] = "ZZ"
        definitions = resolve_definitions(["GLY"], rows(GLY=items(**columns)), gly)
        assert withheld(definitions, "GLY") and codes(definitions) == ["definition_conflict"]

    @pytest.mark.parametrize("channel", ["file", "caller"])
    @pytest.mark.parametrize("outcome", ["inverted", "refused"])
    def test_restatement_keeps_arbitration(self, ccd, define, library, outcome, channel):
        first = define(ccd["ALA"])
        neighbors = first.stereo_neighbors.copy()
        if outcome == "inverted":
            basis = StereoBasis.DESCRIPTOR_VERIFIED
            parity = (3 - first.stereo_parity).astype(np.int8)
            arbitration = StereoArbitration(parity, neighbors, np.array([basis], np.uint8), ("CA",), (), True)
        else:
            basis = StereoBasis.REFUSED
            arbitration = StereoArbitration(
                np.zeros(1, np.int8), neighbors, np.array([basis], np.uint8), (), ("CA",), False
            )
        archived = library([define(ccd["ALA"], stereo=arbitration)]).definition("ALA")
        if channel == "file":
            definitions = resolve_definitions(["ALA"], rows(ALA=ccd["ALA"]), library([archived]))
        else:
            caller = {"ALA": define(ccd["ALA"], DefinitionScope.CALLER)}
            definitions = resolve_definitions(["ALA"], rows(), library([archived]), caller)
        definition = definitions.by_id["ALA"]
        assert definition.reference_source == archived.reference_source
        assert definition.stereo_parity.tolist() == archived.stereo_parity.tolist()
        assert definition.stereo_basis.tolist() == [basis]
        assert "definition_conflict" not in codes(definitions)

    def test_changed_graph_refuses_verification(self, ccd, define, library):  # verified on the archive graph only
        first = define(ccd["ALA"])
        basis = np.array([StereoBasis.DESCRIPTOR_VERIFIED], np.uint8)
        parity = (3 - first.stereo_parity).astype(np.int8)
        arbitration = StereoArbitration(parity, first.stereo_neighbors.copy(), basis, ("CA",), (), True)
        archived = library([define(ccd["ALA"], stereo=arbitration)]).definition("ALA")
        charge, charge_state, quantum = ccd["ALA"].numbers["charge"]
        charge = np.where(np.arange(len(charge)) == 0, 1, charge)  # N: +1
        numbers = {**ccd["ALA"].numbers, "charge": (charge, charge_state, quantum)}
        file = replace(ccd["ALA"], numbers=MappingProxyType(numbers))
        definition = resolve_definitions(["ALA"], rows(ALA=file), library([archived])).by_id["ALA"]
        assert (definition.stereo_parity.tolist(), definition.stereo_basis.tolist()) == ([0], [StereoBasis.REFUSED])

    def test_missing_and_atomless_incomplete(self, archive, library):
        definitions = resolve_definitions(["UNL", "QQQ"], rows(), library([archive("UNL")]))
        assert (sorted(definitions.by_id), definitions.by_id["UNL"].atom_names) == (["UNL"], ())

    def test_type_only_ports_undefined(self, items, gly):
        definitions = resolve_definitions(["NEW"], rows(NEW=items({"id": "NEW", "type": "L-PEPTIDE LINKING"})), gly)
        assert ports(definitions.by_id["NEW"]).status == (PortStatus.UNDEFINED,) * 2
        assert definitions.by_id["NEW"].atom_names == ()

    def test_file_change_reverifies_ports(self, ccd, archive, library):
        file = edit_atom(ccd["ALA"], "pdbx_leaving_atom_flag", "OXT", b"N")
        definitions = resolve_definitions(["ALA"], rows(ALA=file), library([archive("ALA")]))
        record = ports(definitions.by_id["ALA"])
        assert (record.status[1], record.condition[1]) == (
            PortStatus.UNVERIFIED,
            PortCondition.LEAVING_GROUP,
        )
        assert "port_unverified" in codes(definitions)

    def test_findings_become_records(self, items, glycine, define, library):
        empty = (np.zeros(0, np.int8), np.zeros((0, 4), np.int16), np.zeros(0, np.uint8))
        contradicted = define(items(**glycine()), stereo=StereoArbitration(*empty, (), (), True))
        definitions = resolve_definitions(["GLY"], rows(), library([contradicted]))
        assert codes(definitions) == ["reference_coordinates_contradicted"]

    def test_order_and_duplicates_irrelevant(self, archive, library):
        alanine_glycine = library([archive("ALA"), archive("GLY")])
        first = resolve_definitions(["GLY", "ALA", "GLY"], rows(), alanine_glycine)
        second = resolve_definitions(("ALA", "GLY"), rows(), alanine_glycine)
        assert list(first.by_id) == list(second.by_id) == ["ALA", "GLY"]
        assert [c.digest for c in first.by_id.values()] == [c.digest for c in second.by_id.values()]


def negative_offsets(header, payload) -> None:
    header["element"]["data_offsets"] = [offset - len(payload) for offset in header["element"]["data_offsets"]]


def shifted(name: str, index: int, by: int):
    r"""An edit adding ``by`` to entry ``index`` of the int64 pointer ``name`` in the payload."""

    def edit(header, payload) -> None:
        start, stop = header[name]["data_offsets"]
        pointer = np.frombuffer(payload[start:stop], "<i8").copy()
        pointer[index] += by
        payload[start:stop] = pointer.tobytes()

    return edit


ARTIFACT_EDITS = {
    "dtype": lambda header, payload: header["component_id"].update(dtype="I8"),
    "shape": lambda header, payload: header["record.offset"]["shape"].insert(0, 1),
    "rank": lambda header, payload: header["record.offset"]["shape"].extend([1] * 64),  # past what NumPy builds
    "offsets": negative_offsets,
    "missing": lambda header, payload: header.pop("atom.ptr"),
    "pointer-end": shifted("atom.ptr", -1, 5),
    "pointer-interior": shifted("record.offset", 1, 1),
}


@pytest.fixture(scope="module")
def subset(archive, tmp_path_factory):
    path = tmp_path_factory.mktemp("library") / "subset.safetensors"
    write_artifact([archive(x) for x in SUBSET_IDS], path, provenance())
    return path


class TestLibrary:
    def test_round_trip(self, archive, subset):
        loaded = load_library(subset)
        for component_id in SUBSET_IDS:
            written, read = archive(component_id), loaded.definition(component_id)
            for field in fields(written):
                mine, theirs = getattr(written, field.name), getattr(read, field.name)
                same = np.array_equal(mine, theirs) if isinstance(mine, np.ndarray) else mine == theirs
                assert same, (component_id, field.name)

    def test_writes_deterministic(self, archive, subset, tmp_path):
        again = tmp_path / "again.safetensors"
        write_artifact([archive(x) for x in reversed(SUBSET_IDS)], again, provenance())
        assert again.read_bytes() == subset.read_bytes()

    def test_lookup(self, subset):
        loaded = load_library(subset)
        assert (loaded.definition("QQQ"), "QQQ" in loaded, "ALA" in loaded) == (None, False, True)

    def test_older_snapshot_loads(self, archive, library):
        assert library([archive("ALA")], "protenix_20.01.01").provenance.content_version == "protenix_20.01.01"

    def test_unsupported_schema(self, subset, tmp_path):
        other = tmp_path / "other.safetensors"
        other.write_bytes(subset.read_bytes().replace(b'"schema":"ofoldx.ccd/7"', b'"schema":"ofoldx.ccd/9"', 1))
        with pytest.raises(UnsupportedError) as raised:
            load_library(other)
        assert raised.value.code == "artifact_schema_unsupported"

    def test_corruption_detected(self, subset, tmp_path):
        data = bytearray(subset.read_bytes())
        data[-1] ^= 1
        (tmp_path / "flipped.safetensors").write_bytes(bytes(data))
        with pytest.raises(ContradictionError) as raised:
            load_library(tmp_path / "flipped.safetensors")
        assert raised.value.code == "native_corrupt"

    @pytest.mark.parametrize("edit", ARTIFACT_EDITS.values(), ids=ARTIFACT_EDITS)
    def test_forged_layout_refused(self, edit, subset, tmp_path):
        data = subset.read_bytes()
        size = int.from_bytes(data[:8], "little")
        header, payload = json.loads(data[8 : 8 + size]), bytearray(data[8 + size :])
        edit(header, payload)
        header["__metadata__"]["sha256"] = hashlib.sha256(payload).hexdigest()  # the payload digest stays valid
        text = json.dumps(header).encode()
        path = tmp_path / "forged.safetensors"
        path.write_bytes(len(text).to_bytes(8, "little") + text + payload)
        with pytest.raises(ContradictionError) as raised:
            library = load_library(path)
            for component_id in SUBSET_IDS:
                library.definition(component_id)
        assert raised.value.code == "native_corrupt"


class TestGetComponent:
    def test_hit(self, archive, library):
        loaded = library([archive("ALA")])
        assert get_component("ALA", library=loaded) is loaded.definition("ALA")

    def test_miss_names_version(self, archive, library):
        with pytest.raises(InsufficientInformationError) as raised:
            get_component("QQQ", library=library([archive("ALA")], "v1"))
        error = raised.value
        assert (error.code, error.detail["id"], error.detail["content_version"]) == (
            "component_not_in_library",
            "QQQ",
            "v1",
        )

    def test_configured_library(self, monkeypatch, archive, tmp_path):
        path = tmp_path / "configured.safetensors"
        write_artifact([archive("ALA")], path, provenance())
        monkeypatch.setenv("OFOLDX_CCD_ARTIFACT", str(path))  # the environment variable is the interface
        assert get_component("ALA").digest == archive("ALA").digest

    @pytest.mark.parametrize("value", [None, ""])
    def test_default_download_is_cached(self, monkeypatch, archive, tmp_path, value):
        from ofoldx.chemistry import definitions

        path = tmp_path / "default.safetensors"
        write_artifact([archive("ALA")], path, provenance())
        monkeypatch.setattr(definitions, "_CONFIGURED", {})
        monkeypatch.delenv("OFOLDX_CCD_ARTIFACT", raising=False)
        if value is not None:
            monkeypatch.setenv("OFOLDX_CCD_ARTIFACT", value)
        calls = []

        def download():
            calls.append(True)
            return path

        monkeypatch.setattr("ofoldx.chemistry.definitions.download_library", download)
        first = configured_library()
        assert configured_library() is first
        assert get_component("ALA").digest == archive("ALA").digest
        assert len(calls) == 1

    def test_missing_local_override_does_not_download(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OFOLDX_CCD_ARTIFACT", str(tmp_path / "missing.safetensors"))

        def download():
            pytest.fail("an explicit local path must not fall back to the Hub")

        monkeypatch.setattr("ofoldx.chemistry.definitions.download_library", download)
        with pytest.raises(FileNotFoundError):
            configured_library()
