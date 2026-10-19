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

import pytest

from ofoldx.chemistry.definitions import StereoBasis
from ofoldx.chemistry.smiles import canonical_atom_ranks, read_smiles, standardize_smiles
from ofoldx.data.molecular import DefinitionScope, ReferenceSource
from ofoldx.data.stereo import StereoGroupKind
from ofoldx.errors import ContradictionError, UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

STATED, DERIVED = state(Resolution.KNOWN, Basis.REFERENCE), state(Resolution.KNOWN, Basis.DERIVED)
# RDKit's CXSMILES of alanine drawn in 2-D: its coordinates and the wedge it draws for the tag
WEDGED = "C[C@H](N)C(=O)O |(1.5,1.29904,;0.75,0,;1.5,-1.29904,;-0.75,0,;-1.5,1.29904,;-1.5,-1.29904,),wD:1.0|"


def component(text: str):
    return read_smiles(text, component_id="LIG").component


def bond(c, first: str, second: str) -> int:
    pair = {c.atom_names.index(first), c.atom_names.index(second)}
    return next(k for k, atoms in enumerate(c.bond_atoms.tolist()) if set(atoms) == pair)


class TestDefinition:
    def test_element_count_names(self):
        c = component("OCC")
        assert c.atom_names == ("O1", "C1", "C2", "H1", "H2", "H3", "H4", "H5", "H6")
        assert (c.scope, c.reference_source) == (DefinitionScope.SMILES, ReferenceSource.NONE)

    def test_stated_and_kekule_orders(self):
        stated, aromatic = component("C1=CC=CC=C1"), component("c1ccccc1")
        assert stated.bond_order[bond(stated, "C1", "C2")] == 2
        assert set(stated.bond_order_state[:6].tolist()) == {STATED}
        assert sorted(aromatic.bond_order[:6].tolist()) == [1, 1, 1, 2, 2, 2]
        assert set(aromatic.bond_order_state[:6].tolist()) == {DERIVED}
        assert aromatic.bond_aromatic[:6].all()

    def test_nitro_charge_separated(self):
        c = component("CN(=O)=O")
        oxygens = sorted((c.charge[c.atom_names.index(o)], c.bond_order[bond(c, "N1", o)]) for o in ("O1", "O2"))
        assert oxygens == [(-1, 1), (0, 2)]
        assert c.charge[c.atom_names.index("N1")] == 1
        assert c.charge_state[c.atom_names.index("N1")] == DERIVED


class TestStereo:
    def test_enantiomers_opposite_parity(self):
        left, right = component("N[C@@H](C)C(=O)O"), component("N[C@H](C)C(=O)O")
        assert left.stereo_centre.tolist() == right.stereo_centre.tolist() == [1]
        assert left.stereo_parity[0] == 3 - right.stereo_parity[0]
        assert left.stereo_basis[0] == StereoBasis.STATED

    def test_lone_pair_last_slot(self):
        c = component("C[S@](=O)CC")
        assert c.stereo_neighbors[0].tolist()[-1] == -1

    @pytest.mark.parametrize(
        ("text", "value"),
        [("F/C=C/F", 2), ("F/C=C\\F", 1), ("F/C=C/1.C1", 2), ("C1.F/C=C/1", 2)],
        ids=["trans", "cis", "opening-digit", "closing-digit"],
    )
    def test_marks_give_cis_trans(self, text, value):
        assert component(text).bond_stereo_value.tolist() == [value]

    def test_bare_r_one_and_group(self):
        c = component("N[C@H](F)[C@H](Cl)Br |r|")
        assert c.stereo_group_kind.tolist() == [StereoGroupKind.AND]
        assert sorted(c.stereo_group_member.tolist()) == [0, 1]

    @pytest.mark.parametrize(
        ("text", "records"), [("CC/C=C(C)C", 1), ("F/C=[CH2]", 0)], ids=["stereogenic", "terminal"]
    )
    def test_lone_mark_records(self, text, records):
        imported = read_smiles(text, component_id="LIG")
        assert imported.records.counts().get("cx_incomplete_declaration", 0) == records
        assert len(imported.component.bond_stereo_bond) == 0


class TestCxDeclarations:
    @pytest.mark.parametrize(
        ("text", "value", "reference"),
        [("C1=CCCCCCC1 |c:0|", 1, [7, 2]), ("C1=CCCCCCCCC1 |t:0|", 2, [9, 2]), ("C1CCCCC/C=C\\1 |c:6|", 1, [5, 0])],
        ids=["cis", "trans", "agreeing-marks"],
    )
    def test_ring_geometry(self, text, value, reference):  # along the ring: the reference pair are ring neighbours
        c = component(text)
        assert (c.bond_stereo_value.tolist(), c.bond_stereo_reference.tolist()) == ([value], [reference])

    @pytest.mark.parametrize(
        "text", ["C/C=C/C |ctu:1|", "C/C=C/C |w:0.0|", "C[C@H](N)O |w:1.0|"], ids=["ctu", "wavy-bond", "wavy-centre"]
    )
    def test_unknown_configuration(self, text):  # either configuration, as RDKit reads it
        c = component(text)
        assert (c.stereo_centre.tolist(), c.bond_stereo_bond.tolist()) == ([], [])

    def test_wedge_draws_tag(self):  # a depiction: checked against the tag, never a configuration or conformer
        c, plain = component(WEDGED), component("C[C@H](N)C(=O)O")
        assert (c.stereo_parity.tolist(), c.reference_source) == (plain.stereo_parity.tolist(), ReferenceSource.NONE)


class TestRefusal:
    @pytest.mark.parametrize(
        ("text", "error", "code"),
        [
            ("C1CC", ContradictionError, "smiles_syntax"),
            ("C/C(\\F)=C/F", ContradictionError, "cx_conflicting_declaration"),
            ("F[C@SP1](Cl)(Br)I", ContradictionError, "invalid_stereo_class"),
            ("c1cccc1", ContradictionError, "smiles_chemistry"),
            ("CC |TLB:1|", UnsupportedError, "cx_unsupported_declaration"),
            ("*C", UnsupportedError, "smiles_unsupported"),
            ("[CH4:1]", UnsupportedError, "smiles_atom_mapping_unsupported"),
            ("C1CCCCC/C=C\\1 |t:6|", ContradictionError, "cx_conflicting_declaration"),
            ("C/C=C/C |c:1|", ContradictionError, "cx_conflicting_declaration"),
            (WEDGED.replace("wD", "wU"), ContradictionError, "cx_conflicting_declaration"),
            ("C[C@H](N)O |(0,0,;1,0,),wU:1.0|", ContradictionError, "cx_conflicting_declaration"),
        ],
        ids=[
            "syntax", "conflicting-marks", "stereo-class", "kekule", "cx-key", "dummy-atom", "atom-mapping",
            "ring-against-marks", "ring-off-ring", "wedge-against-tag", "points-per-atom",
        ],  # fmt: skip
    )
    def test_typed_refusal(self, text, error, code):
        with pytest.raises(error) as caught:
            read_smiles(text, component_id="LIG")
        assert caught.value.code == code

    @pytest.mark.parametrize(
        ("text", "refused"),
        [
            *((text, True) for text in ("[CH3]", "[OH]", "[N]=O", "C |^1:0|", "[Zn+2] |^1:0|", "[CH3].[Fe+3]")),
            *((text, False) for text in ("C", "[NH4+]", "[Fe+3]", "[Cu+2]", "[K]", "[Ca]", "[Zn+2]", "[Cl-].[Fe+3]")),
            ("[CH4]", False),  # a bracket atom at its valence; with an atom-map number it is refused as unsupported
        ],
    )
    def test_radical_refused(self, text, refused):  # a metal's RDKit count is not consulted
        if refused:
            with pytest.raises(UnsupportedError) as caught:
                component(text)
            assert caught.value.code == "unsupported_radical"
            assert caught.value.detail == {"atoms": (0,), "electrons": (1,)}
        else:
            component(text)


class TestStandardize:
    def test_largest_fragment_parent(self):
        assert standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"

    def test_ranks_cover_hydrogens(self):
        assert sorted(canonical_atom_ranks("CO").tolist()) == list(range(6))
