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

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from ofoldx.data.molecular import EntityType
from ofoldx.errors import ContradictionError, InsufficientInformationError
from ofoldx.io.binding import (
    POLYMER_TYPES,
    BindingOutcome,
    QuerySource,
    chemical_state,
    definition_atoms,
    endpoint_queries,
    pdb_polymer_types,
    residue_table,
    resolve_endpoints,
)
from ofoldx.io.source import parse_source
from ofoldx.provenance import Basis, Resolution, SourceRefs, known, state

KNOWN = state(Resolution.KNOWN, Basis.DECLARED)
DOT = state(Resolution.INAPPLICABLE, Basis.DECLARED)
NOT_INTEGER = state(Resolution.UNKNOWN, Basis.DERIVED)
UNKNOWN = state(Resolution.UNKNOWN, Basis.NONE)
POLY, BRANCH, NONPOLY, WATER = EntityType.POLYMER, EntityType.BRANCHED, EntityType.NON_POLYMER, EntityType.WATER
ENDPOINT_ITEMS = ("label_asym", "label_seq", "label_comp", "label_atom", "alt")
ENDPOINT_ITEMS += ("auth_asym", "auth_seq", "auth_ins", "auth_comp", "auth_atom")
FAMILY = {"label_asym": "chain", "label_comp": "comp", "label_atom": "atom", "alt": "alt", "auth_asym": "chain"}
FAMILY |= {"auth_ins": "ins", "auth_comp": "comp", "auth_atom": "atom"}


class Source:
    """A SourceTables stand-in: binding reads only these columns. Values are symbols; None is unknown, "." is `.`."""

    def __init__(self, format="mmcif"):
        self.format = format
        self.families = defaultdict(dict)
        self.atoms, self.entities, self.asyms, self.sequences, self.schemes = [], [], [], [], []
        self.connections, self.links, self.conects = [], [], []
        self.unobserved, self.modified, self.terminators = [], [], []

    def code(self, family, value):
        if value is None or value == ".":
            return -1, UNKNOWN if value is None else DOT
        return self.families[family].setdefault(value, len(self.families[family])), KNOWN

    def number(self, value):
        """An author number as the decoders read it: the integer reading, its state and the text code."""
        text, text_state = self.code("number", None if value is None else str(value))
        if text < 0:
            return 0, text_state, text
        integer = str(value).lstrip("+-").isdigit()
        return (int(value) if integer else 0), (KNOWN if integer else NOT_INTEGER), text

    def atom(
        self,
        asym,
        comp,
        seq,
        name,
        *,
        entity="1",
        alt=".",
        auth_seq=None,
        chain=None,
        ins=None,
        occupancy=1.0,
        model=1,
        hetatm=False,
        auth_name=None,
    ):
        """An mmCIF row; ``asym=None`` gives a PDB row (author namespace only)."""
        pdb = asym is None
        self.atoms.append(
            {
                "model": model,
                "group": 2 if hetatm else 1,
                "label_asym": asym,
                "label_entity": None if pdb else entity,
                "label_comp": None if pdb else comp,
                "label_atom": None if pdb else name,
                "label_alt": alt,
                "label_seq": None if pdb else seq,
                "auth_asym": chain if chain is not None else asym,
                "auth_comp": comp,
                "auth_atom": auth_name or name,
                "auth_seq": auth_seq if auth_seq is not None or pdb else (seq if isinstance(seq, int) else 1),
                "insertion": ins,
                "occupancy": occupancy,
            }
        )
        return self

    def pdb_atom(self, chain, comp, number, name, **items):
        return self.atom(None, comp, None, name, chain=chain, auth_seq=number, **items)

    def ter(self):
        """A PDB ``TER`` record before the next row."""
        self.terminators.append(len(self.atoms))
        return self

    def ref(self):
        return len(self.entities) + len(self.asyms) + len(self.sequences) + len(self.schemes) + len(self.connections)

    def tables(self):
        connections = self._connections()  # codes its symbols before the atom table snapshots them
        atoms = self._atom_table()
        refs = SourceRefs(
            np.zeros(256, np.int16),
            np.arange(256),
            np.full(256, -1, np.int16),
            ("row",),
            (),
            "test" if self.format == "mmcif" else None,
        )
        return SimpleNamespace(
            atoms=atoms,
            entities=self._table(self.entities, ("id", "type", "type_state", "ref")),
            asyms=self._table(self.asyms, ("id", "entity", "order", "ref")),
            sequences=self._table(self.sequences, ("entity", "num", "mon_id", "ref")),
            schemes=self._table(
                self.schemes,
                (
                    "kind",
                    "asym",
                    "entity",
                    "number",
                    "mon_id",
                    "pdb_seq_num",
                    "pdb_seq_state",
                    "pdb_seq_text",
                    "pdb_ins_code",
                    "pdb_ins_state",
                    "pdb_strand_id",
                    "pdb_strand_state",
                    "ref",
                ),
            ),
            branch_links=SimpleNamespace(
                entity=np.array([e for e, *_ in self.links], np.int32),
                comp=np.array([c for _, c, _, _, _ in self.links], np.int32).reshape(-1, 2),
                num=np.array([n for _, _, n, _, _ in self.links], np.int32).reshape(-1, 2),
                atom=np.array([a for _, _, _, a, _ in self.links], np.int32).reshape(-1, 2),
                ref=np.array([r for *_, r in self.links], np.int64),
            ),
            connections=connections,
            conect=SimpleNamespace(
                serial=np.array(self.conects, np.int64).reshape(-1, 2), ref=np.arange(len(self.conects), dtype=np.int64)
            ),
            unobserved=self._table(self.unobserved, ("model", "chain", "comp", "seq", "seq_text", "ins")),
            modified=self._table(self.modified, ("asym", "asym_state", "seq", "seq_state", "seq_text", "ins", "comp")),
            terminators=np.array(self.terminators, np.int64),
            refs=refs,
        )

    # declarations

    def entity(self, id_, type_):
        self.entities.append((self.code("entity", id_)[0], type_, KNOWN, self.ref()))
        return self

    def asym(self, id_, entity="1"):
        self.asyms.append((self.code("chain", id_)[0], self.code("entity", entity)[0], len(self.asyms), self.ref()))
        return self

    def sequence(self, *components, entity="1", chain=None):
        """``_entity_poly_seq`` of ``entity``, or the SEQRES of ``chain``."""
        key = self.code("chain", chain)[0] if chain is not None else self.code("entity", entity)[0]
        for num, comp in enumerate(components, 1):
            for candidate in comp.split("|"):  # "ALA|GLY": microheterogeneity
                self.sequences.append((key, num, self.code("comp", candidate)[0], self.ref()))
        return self

    def scheme(self, kind, asym, comp, number, author, entity="1", strand=""):
        """A scheme row; its ``strand`` is ``asym`` unless stated (None: unknown)."""
        value, value_state, text = self.number(author)
        strand_code, strand_state = self.code("chain", asym if strand == "" else strand)
        self.schemes.append(
            (
                kind,
                self.code("chain", asym)[0],
                self.code("entity", entity)[0],
                number,
                self.code("comp", comp)[0],
                value,
                value_state,
                text,
                -1,
                DOT,
                strand_code,
                strand_state,
                self.ref(),
            )
        )
        return self

    def modres(self, chain, number, comp):
        value, value_state, text = self.number(number)
        self.modified.append((*self.code("chain", chain), value, value_state, text, -1, self.code("comp", comp)[0]))
        return self

    def connection(self, first, second, origin=QuerySource.STRUCT_CONN):
        self.connections.append((origin, first, second))
        return self

    def link(self, entity, comp1, num1, atom1, comp2, num2, atom2):
        comps = [self.code("comp", comp1)[0], self.code("comp", comp2)[0]]
        atoms = [self.code("atom", atom1)[0], self.code("atom", atom2)[0]]
        self.links.append((self.code("entity", entity)[0], comps, [num1, num2], atoms, 100 + len(self.links)))
        return self

    # tables

    @staticmethod
    def _table(rows, names):
        columns = list(zip(*rows)) if rows else [()] * len(names)
        return SimpleNamespace(**{name: np.array(column, dtype=np.int64) for name, column in zip(names, columns)})

    def _atom_table(self):
        columns = defaultdict(list)
        for index, row in enumerate(self.atoms):
            columns["row"].append(index)
            columns["serial"].append(index + 1)
            columns["model"].append(row["model"])
            columns["group"].append(row["group"])
            for item, family in (
                ("label_asym", "chain"),
                ("label_entity", "entity"),
                ("label_comp", "comp"),
                ("label_atom", "atom"),
                ("label_alt", "alt"),
                ("auth_asym", "chain"),
                ("auth_comp", "comp"),
                ("auth_atom", "atom"),
                ("insertion", "ins"),
            ):
                code, item_state = self.code(family, row[item])
                columns[item].append(code)
                columns[f"{item}_state"].append(item_state)
            value = row["label_seq"]
            columns["label_seq"].append(value if isinstance(value, int) else 0)
            columns["label_seq_state"].append(KNOWN if isinstance(value, int) else (DOT if value == "." else UNKNOWN))
            for item, value in zip(("auth_seq", "auth_seq_state", "auth_seq_text"), self.number(row["auth_seq"])):
                columns[item].append(value)
            columns["occupancy"].append(row["occupancy"] if row["occupancy"] is not None else np.nan)
            columns["occupancy_state"].append(KNOWN if row["occupancy"] is not None else UNKNOWN)
        table = SimpleNamespace(
            **{
                name: np.array(values, dtype=np.float64 if name == "occupancy" else np.int64)
                for name, values in columns.items()
            }
        )
        for name in [name for name in vars(table) if name.endswith("_state")]:
            setattr(table, name, getattr(table, name).astype(np.uint8))
        if not self.atoms:
            for name in ("row", "serial", "model", "group"):
                setattr(table, name, np.zeros(0, np.int64))
        table.symbols = {family: tuple(values) for family, values in self.families.items()}
        for family in ("chain", "entity", "comp", "atom", "alt", "ins", "number"):
            table.symbols.setdefault(family, ())
        return table

    def _connections(self):
        n = len(self.connections)
        values = {item: np.full((n, 2), -1, np.int64) for item in (*ENDPOINT_ITEMS, "auth_seq_text")}
        field_state = np.full((n, 2, 10), UNKNOWN, np.uint8)
        for i, (_, *sides) in enumerate(self.connections):
            for side, endpoint in enumerate(sides):
                for k, item in enumerate(ENDPOINT_ITEMS):
                    if item not in endpoint:
                        continue
                    value = endpoint[item]
                    if item == "auth_seq":
                        number, field_state[i, side, k], values["auth_seq_text"][i, side] = self.number(value)
                        values[item][i, side] = number
                    elif item in FAMILY:
                        values[item][i, side], field_state[i, side, k] = self.code(FAMILY[item], value)
                    else:
                        values[item][i, side] = value if isinstance(value, int) else 0
                        field_state[i, side, k] = KNOWN if isinstance(value, int) else DOT
        return SimpleNamespace(
            origin=np.array([o for o, *_ in self.connections], np.uint8),
            label_alt=values.pop("alt"),
            field_state=field_state,
            ref=np.arange(n, dtype=np.int64),
            **values,
        )


def definitions(**components):
    """Definitions stand-in: id -> atom names (None: no atoms in force); chem_comp type via ``id=(type, names)``."""
    by_id, incomplete = {}, set()
    for cid, value in components.items():
        type_, names = value if isinstance(value, tuple) else ("L-PEPTIDE LINKING", value)
        if names is None:
            incomplete.add(cid)
            continue
        by_id[cid] = SimpleNamespace(chem_comp_type=type_, chem_comp_type_state=KNOWN, atom_names=tuple(names))
    return SimpleNamespace(by_id=by_id, incomplete=frozenset(incomplete))


def tripeptide():
    """Entity 1 = ALA GLY SER as chain A; residue 3 unobserved."""
    source = Source().entity("1", POLY).asym("A").sequence("ALA", "GLY", "SER")
    for seq, comp in ((1, "ALA"), (2, "GLY")):
        source.atom("A", comp, seq, "N").atom("A", comp, seq, "CA")
    return source


def components(table, tables):
    return [tables.atoms.symbols["comp"][c] for c in table.component]


class TestResidueTable:
    def test_declared_unobserved_residue(self):
        tables = tripeptide().tables()
        table = residue_table(tables, 1)
        assert components(table, tables) == ["ALA", "GLY", "SER"]
        assert table.observed.tolist() == [True, True, False]
        assert table.declared.all()
        assert table.ordinal_of_row.tolist() == [0, 0, 1, 1]

    def test_ordinals_ignore_row_order(self):
        source = tripeptide()
        source.atoms.reverse()
        table = residue_table(source.tables(), 1)
        assert table.ordinal_of_row.tolist() == [1, 1, 0, 0]

    def test_microheterogeneity_shares_position(self):
        source = Source().entity("1", POLY).asym("A").sequence("ALA", "GLY|SER")
        source.atom("A", "ALA", 1, "CA").atom("A", "GLY", 2, "CA", occupancy=0.6).atom(
            "A", "SER", 2, "CA", occupancy=0.4
        )
        table = residue_table(source.tables(), 1)
        assert table.position.tolist() == [0, 1, 1]

    @pytest.mark.parametrize(
        "row, code",
        [
            ({"asym": "A", "comp": "ALA", "seq": 7, "name": "CA"}, "residue_unbound"),
            ({"asym": "A", "comp": "ALA", "seq": 1, "name": "N"}, "duplicate_atom"),
            (
                {"asym": "A", "comp": "ALA", "seq": 1, "name": "N", "alt": "A"},
                "duplicate_atom",
            ),  # beside the altloc-less N
        ],
    )
    def test_contradiction_raises(self, row, code):
        source = tripeptide().atom(**row)
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == code

    @pytest.mark.parametrize("declared", [False, True])  # B: no _struct_asym row, or one stating no entity
    @pytest.mark.parametrize("first", ["atom", "scheme"])
    def test_asym_two_entities(self, declared, first):  # every statement of an instance's entity agrees
        source = Source().asym("B", None) if declared else Source()
        if first == "atom":
            source.atom("B", "ZN", ".", "ZN", entity="2", auth_seq=1)
        else:
            source.scheme(NONPOLY, "B", "ZN", 0, 1, entity="2")
        source.atom("B", "ZN", ".", "ZN", entity="3")
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "asym_entity_mismatch"

    def test_other_model_entity_recorded(self):  # 2q4o: every model is checked, only the selected one refuses
        source = Source().asym("B", "2").atom("B", "ZN", ".", "ZN", entity="2", auth_seq=1)
        source.atom("B", "ZN", ".", "ZN", entity="3", auth_seq=1, model=2)
        (found,) = residue_table(source.tables(), 1).records.where(code="entry_level_violation")
        assert found.detail == {"model": 2, "check": "asym_entity_mismatch", "asym": "B"}
        assert [(ref.category, ref.row, ref.item) for ref in found.sources] == [
            ("atom_site", 1, "label_entity_id"),
            ("row", 0, None),  # the _struct_asym row of B
        ]
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 2)
        assert error.value.code == "asym_entity_mismatch"

    @pytest.mark.parametrize(
        "declared, stated",
        [((10, "A"), (11, "A")), (("A1", "A"), ("A2", "A")), ((10, "X"), (10, ".")), ((10, "."), (10, "X"))],
    )
    def test_scheme_address_conflict(self, declared, stated):
        source = Source().entity("1", POLY).asym("A").sequence("ALA")
        source.scheme(POLY, "A", "ALA", 1, author=declared[0], strand=declared[1])
        source.atom("A", "ALA", 1, "CA", auth_seq=stated[0], chain=stated[1])
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "label_auth_conflict"

    @pytest.mark.parametrize("item", ["auth_asym", "auth_seq", "auth_comp"])
    def test_unknown_author_part_compatible(self, item):
        source = Source().entity("1", POLY).asym("A").sequence("ALA")
        source.atom("A", "ALA", 1, "N", chain="X", auth_seq=5).atom("A", "ALA", 1, "CA", chain="X", auth_seq=5)
        source.atoms[0][item] = None  # the first row leaves it unknown
        tables = source.tables()
        table = residue_table(tables, 1)
        assert tables.atoms.symbols["chain"][table.author_chain[0]] == "X" and table.author_number.tolist() == [5]
        assert table.author_component.tolist() == [tables.atoms.symbols["comp"].index("ALA")]

    @pytest.mark.parametrize("unknown", ["strand", "author"])
    def test_declared_unknown_takes_rows(self, unknown):  # the scheme leaves it `?`, the rows state it
        source = Source().entity("1", POLY).asym("A").sequence("GLY")
        source.scheme(POLY, "A", "GLY", 1, **({"strand": "P", "author": 7} | {unknown: None}))
        tables = source.atom("A", "GLY", 1, "CA", chain="P", auth_seq=7).tables()
        table = residue_table(tables, 1)
        assert (tables.atoms.symbols["number"][table.author_text[0]], table.author_state[0]) == ("7", KNOWN)
        chain = tables.atoms.symbols["chain"][table.instance_author_chain[0]]
        assert (chain, table.instance_author_chain_state[0]) == ("P", KNOWN)

    def test_rows_state_two_components(self):  # the second row says SEM
        source = (
            Source().entity("1", POLY).asym("A").sequence("MSE").atom("A", "MSE", 1, "CA").atom("A", "MSE", 1, "SE")
        )
        source.atoms[1]["auth_comp"] = "SEM"
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "label_auth_conflict"

    @pytest.mark.parametrize("kind, authors", [(NONPOLY, ("A1",)), (BRANCH, ("A1", "A2"))])
    def test_token_scheme_address_binds(self, kind, authors):
        source = Source().entity("2", kind).asym("B", "2")
        for number, author in enumerate(authors, 1):
            source.scheme(kind, "B", "NAG", number, author=author, entity="2")
            source.atom("B", "NAG", ".", "C1", entity="2", auth_seq=author)
        table = residue_table(source.tables(), 1)
        assert table.observed.all() and table.ordinal_of_row.tolist() == list(range(len(authors)))

    @pytest.mark.parametrize("kind, comp", [(BRANCH, "NAG"), (WATER, "HOH")])
    def test_token_addresses_without_scheme(self, kind, comp):
        source = Source().entity("2", kind).asym("B", "2")
        for author in ("A1", "A2"):
            source.atom("B", comp, ".", "O", entity="2", auth_seq=author)
        assert residue_table(source.tables(), 1).ordinal_of_row.tolist() == [0, 1]

    def test_one_molecule_two_tokens(self):
        source = Source().entity("2", NONPOLY).asym("B", "2")
        for name, author in (("C", "A1"), ("O", "A2")):
            source.atom("B", "ACT", ".", name, entity="2", auth_seq=author)
        with pytest.raises(InsufficientInformationError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "nonpolymer_address_ambiguous"

    @pytest.mark.parametrize("unknown", [0, 1])
    def test_unknown_number_one_molecule(self, unknown):
        source = Source().entity("2", NONPOLY).asym("B", "2")
        source.atom("B", "ACT", ".", "C", entity="2", auth_seq=201).atom("B", "ACT", ".", "O", entity="2", auth_seq=201)
        source.atoms[unknown]["auth_seq"] = None  # `?`
        table = residue_table(source.tables(), 1)
        assert table.ordinal_of_row.tolist() == [0, 0] and table.author_number.tolist() == [201]

    @pytest.mark.parametrize("chains", [("Y", "X"), (".", "X")])
    def test_asym_two_author_chains(self, chains):
        source = Source().entity("1", POLY).asym("A").sequence("ALA", "GLY")
        for seq, (comp, chain) in enumerate(zip(("ALA", "GLY"), chains), 1):
            source.atom("A", comp, seq, "CA", chain=chain)
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "asym_two_chains"

    def test_nonpolymer_without_scheme(self):
        source = Source().entity("2", NONPOLY).asym("B", "2")
        source.atom("B", "HEM", ".", "FE", entity="2", auth_seq=201).atom(
            "B", "HEM", ".", "NA", entity="2", auth_seq=201
        )
        table = residue_table(source.tables(), 1)
        assert len(table.kind) == 1 and table.observed.all()
        assert table.records.counts() == {"scheme_absent": 1}

    def test_unstated_component_unbound(self):  # label_comp_id ? names no component
        source = Source().entity("2", NONPOLY).asym("B", "2").atom("B", None, ".", "FE", entity="2", auth_seq=201)
        with pytest.raises(ContradictionError) as error:
            residue_table(source.tables(), 1)
        assert error.value.code == "residue_unbound"

    def test_unresolved_asym_has_no_residues(self):
        source = tripeptide().entity("2", NONPOLY).asym("B", "2")
        table = residue_table(source.tables(), 1)
        assert table.instance_symbol == ("A", "B")
        assert 1 not in table.instance.tolist()
        assert table.records.counts() == {"instance_unobserved": 1}

    @pytest.mark.parametrize(("text", "format"), [("data_X\n", "mmcif"), ("HEADER    TEST\nEND\n", "pdb")])
    def test_no_molecule_refused(self, text, format):  # a bare data block, a component definition file
        with pytest.raises(InsufficientInformationError, match="molecule_absent"):
            residue_table(parse_source(text, format=format).tables(1), 1)


class TestPdbResidues:
    def seqres(self, *components):
        return Source(format="pdb").sequence(*components, chain="A")

    def test_seqres_places_gap_and_water(self):
        source = self.seqres("ALA", "GLY", "SER")
        source.pdb_atom("A", "ALA", 1, "CA").pdb_atom("A", "SER", 3, "CA").pdb_atom("A", "HOH", 101, "O", hetatm=True)
        tables = source.tables()
        table = residue_table(tables, 1)
        assert components(table, tables) == ["ALA", "GLY", "SER", "HOH"]
        assert table.observed.tolist() == [True, False, True, True]
        assert table.kind.tolist() == [POLY, POLY, POLY, WATER]
        assert table.instance_symbol == ("A", "A_water")

    def test_ter_ends_polymer(self):  # a free amino acid after TER is no SEQRES residue (6iot ARG A 501)
        source = self.seqres("GLY", "SER", "ARG").pdb_atom("A", "GLY", 1, "CA").pdb_atom("A", "SER", 2, "CA")
        table = residue_table(source.ter().pdb_atom("A", "ARG", 501, "CA", hetatm=True).tables(), 1)
        assert table.instance_symbol == ("A", "A_ARG_501")
        assert table.observed.tolist() == [True, True, False, True]

    @pytest.mark.parametrize(
        "seqres, numbered, embeddings",
        [(("ALA", "ALA", "ALA"), [False] * 3, 3), (("ALA", "GLY", "ALA"), [False, True, False], 2)],
    )
    def test_ambiguous_correspondence(self, seqres, numbered, embeddings):
        source = self.seqres(*seqres).pdb_atom("A", "ALA", 5, "CA")
        tables = source.tables()
        table = residue_table(tables, 1)
        record = table.records.where(code="seqres_correspondence_ambiguous")[0]
        assert record.detail == {"embeddings": embeddings}
        assert components(table, tables) == list(seqres)
        assert known(table.number_state).tolist() == numbered

    def test_microheterogeneity_one_position(self):
        source = self.seqres("ALA", "PRO", "GLY").pdb_atom("A", "ALA", 1, "CA")
        source.pdb_atom("A", "PRO", 2, "CA", alt="A", occupancy=0.6).pdb_atom(
            "A", "SER", 2, "CA", alt="B", occupancy=0.4
        )
        tables = source.pdb_atom("A", "GLY", 3, "CA").tables()
        table = residue_table(tables, 1)
        assert table.position.tolist() == [0, 1, 1, 2]
        assert table.number.tolist() == [1, 2, 2, 3]
        assert chemical_state(table, tables.atoms, altloc="highest_occupancy").selected.tolist() == [1, 1, 0, 1]

    def test_number_compared_as_integer(self):  # resSeq "1" and "01" name one residue
        source = Source(format="pdb").pdb_atom("A", "GLY", "1", "N").pdb_atom("A", "GLY", "01", "CA")
        assert residue_table(source.tables(), 1).ordinal_of_row.tolist() == [0, 0]

    def test_blank_chain_stated_none(self):  # its rows and its SEQRES alike
        source = Source(format="pdb").sequence("GLY", "GLY", chain=".").pdb_atom(".", "GLY", 1, "CA")
        table = residue_table(source.tables(), 1)
        assert table.author_chain_state.tolist() == [DOT, DOT] and table.instance_author_chain_state.tolist() == [DOT]

    def test_remark_465_places(self):
        source = self.seqres("ALA", "ALA", "ALA").pdb_atom("A", "ALA", 1, "CA").pdb_atom("A", "ALA", 3, "CA")
        source.unobserved.append(
            (0, source.code("chain", "A")[0], source.code("comp", "ALA")[0], 2, source.code("number", "2")[0], -1)
        )
        table = residue_table(source.tables(), 1)
        assert table.number.tolist() == [1, 2, 3]
        assert table.author_number.tolist() == [1, 2, 3]
        assert len(table.records) == 0

    @pytest.mark.parametrize(("number", "found"), [(0, 0), (None, 1)], ids=["numbered", "unnumbered"])
    def test_modres_names_numbered_residue(self, number, found):  # the MODRES row states resSeq 0
        source = self.seqres("MSE").pdb_atom("A", "MSE", number, "CA", hetatm=True).modres("A", 0, "MSE")
        table = residue_table(source.tables(), 1)
        assert len(table.records.where(code="group_vs_seqres")) == found


class TestChemicalState:
    def altlocs(self, a, b):
        source = Source().entity("1", POLY).asym("A").sequence("SER")
        source.atom("A", "SER", 1, "CA")
        source.atom("A", "SER", 1, "OG", alt="A", occupancy=a).atom("A", "SER", 1, "OG", alt="B", occupancy=b)
        tables = source.tables()
        return residue_table(tables, 1), tables.atoms

    def test_highest_occupancy(self):
        table, atoms = self.altlocs(0.4, 0.6)
        chosen = chemical_state(table, atoms, altloc="highest_occupancy")
        assert chosen.row_selected.tolist() == [True, False, True]

    def test_tie_goes_to_lower_label(self):
        table, atoms = self.altlocs(0.5, 0.5)
        chosen = chemical_state(table, atoms, altloc="highest_occupancy")
        assert chosen.row_selected.tolist() == [True, True, False]
        assert chosen.records.counts() == {"altloc_tie_by_label": 1}

    def test_label_policy(self):
        table, atoms = self.altlocs(0.9, 0.1)
        chosen = chemical_state(table, atoms, altloc="B")
        assert chosen.row_selected.tolist() == [True, False, True]

    def test_unknown_occupancy_raises(self):
        table, atoms = self.altlocs(None, 0.5)
        with pytest.raises(InsufficientInformationError) as error:
            chemical_state(table, atoms, altloc="highest_occupancy")
        assert error.value.code == "altloc_occupancy_unknown"

    def test_unknown_altloc_beside_letter(self):  # `?` is not the altloc-less `.`: unknown, not a duplicate
        source = Source().entity("1", POLY).asym("A").sequence("ALA")
        source.atom("A", "ALA", 1, "CA", alt=None, occupancy=0.5).atom("A", "ALA", 1, "CA", alt="A", occupancy=0.5)
        tables = source.tables()
        with pytest.raises(InsufficientInformationError) as error:
            chemical_state(residue_table(tables, 1), tables.atoms, altloc="highest_occupancy")
        assert error.value.code == "altloc_unknown"

    def test_microheterogeneity_by_occupancy(self):
        source = Source().entity("1", POLY).asym("A").sequence("GLY|SER")
        source.atom("A", "GLY", 1, "CA", occupancy=0.3).atom("A", "SER", 1, "CA", occupancy=0.7)
        tables = source.tables()
        chosen = chemical_state(residue_table(tables, 1), tables.atoms, altloc="highest_occupancy")
        assert chosen.selected.tolist() == [False, True]


class TestPdbPolymerTypes:
    @pytest.mark.parametrize(
        "types, expected",
        [
            (("L-PEPTIDE LINKING", "PEPTIDE LINKING"), "polypeptide(L)"),
            (("DNA LINKING", "DNA LINKING"), "polydeoxyribonucleotide"),
            (("L-PEPTIDE LINKING", "D-PEPTIDE LINKING"), None),
        ],
    )
    def test_derived_type(self, types, expected):
        source = Source(format="pdb").sequence("XA", "XB", chain="A").pdb_atom("A", "XA", 1, "CA")
        tables = source.tables()
        codes, states, records = pdb_polymer_types(
            tables, residue_table(tables, 1), definitions(XA=(types[0], ["CA"]), XB=(types[1], ["CA"]))
        )
        if expected is None:  # derived, none determined: a re-read of the same SEQRES derives the same
            assert codes.tolist() == [-1] and records.counts() == {"polymer_type_unstated": 1}
            assert states.tolist() == [state(Resolution.UNKNOWN, Basis.DERIVED)]
        else:
            assert POLYMER_TYPES[codes[0]] == expected and known(states).all()


class TestEndpoints:
    DEFINITIONS = definitions(ALA=["N", "CA", "C", "O", "CB"], GLY=["N", "CA", "C", "O"], SER=["N", "CA", "OG"])

    def resolve(self, source, definitions_=DEFINITIONS, altloc="highest_occupancy"):
        tables = source.tables()
        table = residue_table(tables, 1)
        chosen = chemical_state(table, tables.atoms, altloc=altloc)
        return resolve_endpoints(
            table, tables.atoms, state=chosen, definitions=definitions_, query=endpoint_queries(tables, table)
        )

    @pytest.mark.parametrize(
        "endpoint, outcome",
        [
            ({"label_asym": "A", "label_seq": 2, "label_comp": "GLY", "label_atom": "CA"}, BindingOutcome.RESOLVED),
            ({"auth_asym": "A", "auth_seq": 2, "auth_comp": "GLY", "auth_atom": "CA"}, BindingOutcome.RESOLVED),
            (
                {"label_asym": "A", "label_seq": 1, "label_atom": "CA", "auth_asym": "A", "auth_seq": 2},
                BindingOutcome.LABEL_AUTH_CONFLICT,
            ),
            ({"label_asym": "A", "label_seq": 9, "label_atom": "CA"}, BindingOutcome.DANGLING),
            ({"label_asym": "A", "label_seq": 2**63 - 1, "label_atom": "CA"}, BindingOutcome.DANGLING),  # a 64-bit key
            (
                {"label_asym": "A", "label_seq": 3, "label_comp": "SER", "label_atom": "OG"},
                BindingOutcome.ABSENT_IN_MODEL,
            ),
            ({"label_asym": "A", "label_seq": 2, "label_atom": "C"}, BindingOutcome.RESOLVED),
            ({"label_asym": "A", "label_seq": 2, "label_atom": "CB"}, BindingOutcome.ATOM_NOT_IN_REFERENCE),
        ],
    )
    def test_outcome(self, endpoint, outcome):
        source = tripeptide().connection(endpoint, {"label_asym": "A", "label_seq": 1, "label_atom": "N"})
        resolution = self.resolve(source)
        assert resolution.outcome[0] == outcome

    def test_resolved_atom_and_extra(self):
        source = tripeptide().atom("A", "GLY", 2, "XX")
        source.connection(
            {"label_asym": "A", "label_seq": 2, "label_atom": "CA"},
            {"label_asym": "A", "label_seq": 2, "label_atom": "XX"},
        )
        resolution = self.resolve(source)
        assert resolution.outcome.tolist() == [BindingOutcome.RESOLVED] * 2
        assert resolution.atom.tolist() == [1, -1]
        assert resolution.extra.tolist() == [-1, 4]

    @pytest.mark.parametrize(
        "names, outcome",
        [
            ({"label_atom": "O", "auth_atom": "O1"}, BindingOutcome.RESOLVED),
            ({"auth_atom": "O1"}, BindingOutcome.RESOLVED),
            ({"label_atom": "O", "auth_atom": "O"}, BindingOutcome.LABEL_AUTH_CONFLICT),
        ],
    )
    def test_author_atom_name(self, names, outcome):
        source = tripeptide().atom("A", "GLY", 2, "O", auth_name="O1")
        source.connection(
            dict(label_asym="A", label_seq=2, **names), {"label_asym": "A", "label_seq": 1, "label_atom": "N"}
        )
        resolution = self.resolve(source)
        assert resolution.outcome[0] == outcome
        assert resolution.atom[0] == (3 if outcome == BindingOutcome.RESOLVED else -1)

    @pytest.mark.parametrize(
        ("rows", "atom", "extra", "bound"),
        [(("DA",), 2, -1, [0, 1, 2]), (("HA", "DA"), -1, 3, [0, 1, 2, -1])],
        ids=["D", "H+D"],
    )
    def test_isotope_name(self, rows, atom, extra, bound):  # DA reads as HA unless a row is HA, in both bindings
        source = Source().entity("1", POLY).asym("A").sequence("ALA").atom("A", "ALA", 1, "N").atom("A", "ALA", 1, "CA")
        for name in rows:
            source.atom("A", "ALA", 1, name)
        source.connection(
            {"label_asym": "A", "label_seq": 1, "label_atom": "DA"},
            {"label_asym": "A", "label_seq": 1, "label_atom": "N"},
        )
        alanine = definitions(ALA=["N", "CA", "HA"])
        resolution = self.resolve(source, alanine)
        assert resolution.outcome[0] == BindingOutcome.RESOLVED
        assert (resolution.atom[0], resolution.extra[0]) == (atom, extra)
        tables = source.tables()
        table = residue_table(tables, 1)
        chosen = chemical_state(table, tables.atoms, altloc="highest_occupancy")
        assert definition_atoms(table, tables.atoms, chosen, alanine)[0].tolist() == bound

    def test_definition_missing_unbound(self):
        source = tripeptide().connection(
            {"label_asym": "A", "label_seq": 2, "label_atom": "CB"},
            {"label_asym": "A", "label_seq": 1, "label_atom": "N"},
        )
        resolution = self.resolve(source, definitions(ALA=["N"], GLY=None))
        assert resolution.outcome[0] == BindingOutcome.ATOM_UNBOUND

    def test_unselected_alternative(self):
        source = (
            tripeptide()
            .atom("A", "GLY", 2, "O", alt="A", occupancy=0.7)
            .atom("A", "GLY", 2, "O", alt="B", occupancy=0.3)
        )
        source.connection(
            {"label_asym": "A", "label_seq": 2, "label_atom": "O", "alt": "B"},
            {"label_asym": "A", "label_seq": 1, "label_atom": "N"},
        )
        assert self.resolve(source).outcome[0] == BindingOutcome.UNSELECTED_ALTERNATIVE

    def test_branch_link_per_asym(self):
        source = Source().entity("2", BRANCH).asym("C", "2").asym("D", "2")
        for asym in "CD":
            source.scheme(BRANCH, asym, "NAG", 1, author=1, entity="2").scheme(
                BRANCH, asym, "NAG", 2, author=2, entity="2"
            )
            for number in (1, 2):
                source.atom(asym, "NAG", ".", "C1", entity="2", auth_seq=number).atom(
                    asym, "NAG", ".", "O4", entity="2", auth_seq=number
                )
        source.link("2", "NAG", 2, "C1", "NAG", 1, "O4")
        resolution = self.resolve(source, definitions(NAG=["C1", "O4"]))
        assert resolution.outcome.tolist() == [BindingOutcome.RESOLVED] * 4
        assert resolution.ordinal.tolist() == [1, 0, 3, 2]

    def test_blank_chain_names_its_residue(self):
        source = Source(format="pdb")
        for chain in (".", "B"):  # "." reads as a blank PDB chain
            source.sequence("CYS", "CYS", chain=chain)
            for number in (1, 2):
                source.pdb_atom(chain, "CYS", number, "SG")
        first, second = (
            {"auth_asym": ".", "auth_seq": number, "auth_comp": "CYS", "auth_atom": "SG"} for number in (1, 2)
        )
        resolution = self.resolve(source.connection(first, second, QuerySource.SSBOND), definitions(CYS=["SG"]))
        assert resolution.outcome.tolist() == [BindingOutcome.RESOLVED] * 2
        assert resolution.ordinal.tolist() == [0, 1]

    @pytest.mark.parametrize("numbers", [("A1", "A2"), ("1", "01")])
    def test_author_number_compares_token(self, numbers):
        source = Source().entity("1", POLY).asym("A").sequence("CYS", "CYS")
        for seq, number in enumerate(numbers, 1):
            source.atom("A", "CYS", seq, "SG", auth_seq=number)
        first, second = (
            {"auth_asym": "A", "auth_seq": number, "auth_comp": "CYS", "auth_atom": "SG"} for number in numbers
        )
        resolution = self.resolve(source.connection(first, second), definitions(CYS=["SG"]))
        assert resolution.outcome.tolist() == [BindingOutcome.RESOLVED] * 2
        assert resolution.ordinal.tolist() == [0, 1]

    def test_conect_names_row(self):
        source = (
            Source(format="pdb").pdb_atom("A", "LIG", 1, "C1", hetatm=True).pdb_atom("A", "LIG", 1, "C2", hetatm=True)
        )
        source.conects.append((1, 2))
        resolution = self.resolve(source, definitions(LIG=None))
        assert resolution.outcome.tolist() == [BindingOutcome.RESOLVED] * 2
        assert resolution.extra.tolist() == [0, 1]
