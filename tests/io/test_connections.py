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

import numpy as np
import pytest

from ofoldx.chemistry.declarations import AssumptionRule, AssumptionSet
from ofoldx.data.molecular import ELEMENTS, BondKind, BondOrigin, EndpointState
from ofoldx.errors import ContradictionError, InsufficientInformationError
from ofoldx.io import parse_source, read_structure, write_structure
from ofoldx.io.connections import glycosylation_order
from ofoldx.provenance import Basis, Resolution, known, state

N = ELEMENTS.index("N")
C = ELEMENTS.index("C")
NON_POLYMER = """_entity.id 1
_entity.type non-polymer
_struct_asym.id A
_struct_asym.entity_id 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_asym_id
"""
# an IAS (CCD parent ?) whose _pdbx_struct_mod_residue row leaves its parent unstated
UNSTATED_PARENT = f"""data_IAS
{NON_POLYMER}HETATM 1 C CA IAS A 1 . 1.0 2.0 3.0 1 A
_pdbx_struct_mod_residue.id 1
_pdbx_struct_mod_residue.auth_asym_id A
_pdbx_struct_mod_residue.auth_seq_id 1
_pdbx_struct_mod_residue.auth_comp_id IAS
_pdbx_struct_mod_residue.parent_comp_id ?
"""
# a label MSE at author address {residue} (number chain component), a → MET row at {row} (chain number component)
ADDRESSED_MSE = f"""data_MSE
{NON_POLYMER}_atom_site.auth_comp_id
HETATM 1 C CA MSE A 1 . 1.0 2.0 3.0 {{residue}}
loop_
_pdbx_struct_mod_residue.id
_pdbx_struct_mod_residue.auth_asym_id
_pdbx_struct_mod_residue.auth_seq_id
_pdbx_struct_mod_residue.auth_comp_id
_pdbx_struct_mod_residue.parent_comp_id
1 {{row}} MET
"""
# two MSE copies of one entity (asyms B, C) at one author address A 1, one MSE → MET row naming it
TWO_COPIES = (
    ADDRESSED_MSE.format(residue="1 A MSE\nHETATM 2 C CA MSE C 1 . 4.0 2.0 3.0 1 A MSE", row="A 1 MSE")
    .replace("_struct_asym.id A\n_struct_asym.entity_id 1", "loop_\n_struct_asym.id\n_struct_asym.entity_id\nB 1\nC 1")
    .replace("MSE A 1 .", "MSE B 1 .")
)
# position A 1 is MSE or ALA (microheterogeneity); the MSE → MET annotation names the MSE
MICROHETEROGENEOUS = """data_MH
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type polypeptide(L)
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
_entity_poly_seq.hetero
1 1 MSE y
1 1 ALA y
_struct_asym.id A
_struct_asym.entity_id 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.auth_comp_id
HETATM 1 C CA MSE A 1 1 1.0 2.0 3.0 {mse} 1 A MSE
ATOM 2 C CA ALA A 1 1 1.0 2.0 3.0 {ala} 1 A ALA
_pdbx_struct_mod_residue.id 1
_pdbx_struct_mod_residue.auth_asym_id A
_pdbx_struct_mod_residue.auth_seq_id 1
_pdbx_struct_mod_residue.auth_comp_id MSE
_pdbx_struct_mod_residue.parent_comp_id MET
"""


class TestGlycosylationOrder:
    @pytest.fixture
    def link(self, archive):
        asn, nag = archive("ASN"), archive("NAG")
        atoms = np.array([[asn.atom_names.index("ND2"), nag.atom_names.index("C1")]], dtype=np.int32)
        return asn, nag, atoms

    def test_role_defines_order(self, link):
        asn, nag, atoms = link
        defined, findings = glycosylation_order(["N-Glycosylation"], [(asn, nag)], atoms, np.array([[N, C]]))
        assert defined.tolist() == [True] and findings == (None,)

    def test_incompatible_element(self, link):
        asn, nag, atoms = link
        defined, findings = glycosylation_order(["O-Glycosylation"], [(asn, nag)], atoms, np.array([[N, C]]))
        assert defined.tolist() == [False] and findings == ("role_atoms_incompatible",)

    def test_untestable_saccharide(self, link):
        asn, _, atoms = link
        defined, findings = glycosylation_order(["N-Glycosylation"], [(asn, None)], atoms, np.array([[N, C]]))
        assert defined.tolist() == [False] and findings == ("role_environment_unknown",)

    def test_no_role(self, link):
        asn, nag, atoms = link
        defined, findings = glycosylation_order([None], [(asn, nag)], atoms, np.array([[N, C]]))
        assert defined.tolist() == [False] and findings == (None,)


# a metalc between the two atoms of ``unknown_element``'s LIG
METALC = """loop_
_struct_conn.id
_struct_conn.conn_type_id
_struct_conn.ptnr1_label_asym_id
_struct_conn.ptnr1_label_comp_id
_struct_conn.ptnr1_label_seq_id
_struct_conn.ptnr1_label_atom_id
_struct_conn.ptnr2_label_asym_id
_struct_conn.ptnr2_label_comp_id
_struct_conn.ptnr2_label_seq_id
_struct_conn.ptnr2_label_atom_id
_struct_conn.ptnr1_symmetry
_struct_conn.ptnr2_symmetry
metalc1 metalc A LIG . C1 A LIG . X1 1_555 1_555
"""


def unknown_element(observed: str) -> str:
    r"""A file-scope LIG whose X1 states no element (``type_symbol ?``), its ``atom_site`` row typed ``observed``."""
    return f"""data_LIG
_chem_comp.id LIG
_chem_comp.type NON-POLYMER
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
LIG C1 C
LIG X1 ?
{NON_POLYMER}HETATM 1 C C1 LIG A 1 . 1.0 2.0 3.0 1 A
HETATM 2 {observed} X1 LIG A 1 . 2.4 2.0 3.0 1 A
"""


class TestUnknownReferenceElement:
    def test_observed_element_fills_unknown(self, library):
        source = parse_source(unknown_element("N"), format="mmcif")
        structure, report = read_structure(source, library=library, on_conflict="record")
        system = structure.system
        x1 = system.text("atom", system.atom_symbol).tolist().index("X1")
        assert "element_differs_from_reference" not in report.summary()
        assert system.atomic_number[x1] == N
        assert system.atomic_number_state[x1] == state(Resolution.KNOWN, Basis.DECLARED)

    # a `?`, `.` or absent type is missing information, the row kept with its kind undetermined in the
    # type field's own state, never a Conflict, never filled by link_metal_coordination (a PDB LINK's rule)
    @pytest.mark.parametrize(
        ("conn_type", "kind", "kind_state"),
        [
            ("metalc", BondKind.COORDINATION, state(Resolution.KNOWN, Basis.DEFINED)),
            ("?", 0, state(Resolution.UNKNOWN, Basis.DECLARED)),
            (".", 0, state(Resolution.INAPPLICABLE, Basis.DECLARED)),
            ("", 0, state(Resolution.UNKNOWN, Basis.NONE)),
        ],
        ids=["metalc", "unknown", "inapplicable", "absent"],
    )
    def test_observed_metal_kind(self, library, conn_type, kind, kind_state):
        rows = METALC.replace("metalc1 metalc", f"metalc1 {conn_type}")
        rows = rows if conn_type else rows.replace("_struct_conn.conn_type_id\n", "")
        source = parse_source(unknown_element("ZN") + rows, format="mmcif")
        metal = AssumptionSet(frozenset({AssumptionRule.LINK_METAL_COORDINATION}))
        structure, report = read_structure(source, library=library, on_conflict="record", assumptions=metal)
        assert not {"kind_element_mismatch", "conn_type_enumeration"} & set(report.summary())
        assert structure.system.bond_kind.tolist() == [kind]
        assert structure.system.bond_kind_state.tolist() == [kind_state]


# an ALA whose observed H3 (no CCD counterpart) lies 0.99 Å from N and 1.99 Å from CA
ALANINE = {
    "N": "1.000 2.000 3.000",
    "CA": "2.450 2.000 3.000",
    "C": "3.000 3.400 3.000",
    "O": "2.300 4.400 3.000",
    "CB": "3.000 1.200 4.200",
    "H3": "0.700 2.900 3.300",
}


def alanine(*unknown: str) -> str:
    r"""``ALANINE`` as an mmCIF entry, the ``unknown`` atoms' coordinates ``?``."""
    rows = [
        f"HETATM {k} {name[0]} {name} ALA A 1 . {'? ? ?' if name in unknown else xyz} 1 A"
        for k, (name, xyz) in enumerate(ALANINE.items(), 1)
    ]
    return f"data_ALA\n{NON_POLYMER}" + "\n".join(rows) + "\n"


class TestModifiedResidues:
    def test_unstated_parent_annotates_none(self, library):
        structure, _ = read_structure(parse_source(UNSTATED_PARENT, format="mmcif"), library=library)
        assert not len(structure.system.component_parent)

    @pytest.mark.parametrize(
        ("residue", "row", "parent"),
        [
            ("0 A MSE", "A 0 MSE", ["MET"]),
            ("0 A MSE", "A ? MSE", []),
            ("? A MSE", "A 0 MSE", []),
            ("0 ? MSE", "? 0 MSE", []),
            ("0 . MSE", ". 0 MSE", ["MET"]),
            ("1A A MSE", "A 1A MSE", ["MET"]),
            ("1A A MSE", "A 2B MSE", []),
            ("01 A MSE", "A 1 MSE", []),
            ("1 A SEM", "A 1 SEM", ["MET"]),
            ("1 A SEM", "A 1 MSE", []),
            ("1 A ?", "A 1 MSE", []),
        ],
        ids=[
            "stated",
            "row-number",
            "residue-number",
            "chain",
            "chain-none",
            "token",
            "other-token",
            "padded-token",
            "author-component",
            "label-component",
            "residue-component",
        ],
    )
    def test_address_matches_as_stated(self, library, residue, row, parent):
        source = parse_source(ADDRESSED_MSE.format(residue=residue, row=row), format="mmcif")
        structure, report = read_structure(source, library=library)
        system = structure.system
        assert system.text("parent", system.component_parent).tolist() == parent
        assert system.text("parent", system.component_definition_parent).tolist() == ["MET"]
        assert report.summary().get("modified_residue_unbound", 0) == (not parent)  # declared but unrealised

    @pytest.fixture
    def labelled(self):
        r"""The A 0 MSE → MET row also stating ``label_asym_id`` (the MSE's is A)."""
        text = ADDRESSED_MSE.format(residue="0 A MSE", row="A 0 MSE")
        row = "\n1 A 0 MSE MET"
        return lambda label: parse_source(
            text.replace(row, f"\n_pdbx_struct_mod_residue.label_asym_id{row} {label}"), format="mmcif"
        )

    def test_stated_label_binds(self, library, labelled):
        system = read_structure(labelled("A"), library=library)[0].system
        assert system.text("parent", system.component_parent).tolist() == ["MET"]

    def test_other_label_conflicts(self, library, labelled):  # the label part is one more statement
        with pytest.raises(ContradictionError, match="label_auth_conflict"):
            read_structure(labelled("B"), library=library)

    @pytest.mark.parametrize("chain", ["A", " "], ids=["named", "blank"])
    def test_pdb_blank_chain_matches(self, library, atom_line, chain):
        lines = [f"MODRES 1ABC MSE {chain}    1  MET", atom_line(1, "CA", "MSE", chain, 1, record="HETATM"), "END"]
        system = read_structure(parse_source("\n".join(lines), format="pdb"), library=library)[0].system
        assert system.text("parent", system.component_parent).tolist() == ["MET"]

    def test_shared_address_is_ambiguous(self, library):
        with pytest.raises(InsufficientInformationError) as raised:
            read_structure(parse_source(TWO_COPIES, format="mmcif"), library=library)
        assert raised.value.code == "modified_residue_ambiguous" and len(raised.value.candidates) == 2

    def test_label_address_binds(self, library):  # the copies of one author chain differ only there
        items = "_pdbx_struct_mod_residue.label_asym_id\n_pdbx_struct_mod_residue.label_seq_id\n"
        text = TWO_COPIES.replace("\n1 A 1 MSE MET", f"\n{items}1 A 1 MSE MET C .")
        system = read_structure(parse_source(text, format="mmcif"), library=library)[0].system
        assert np.diff(system.component_parent_ptr).tolist() == [0, 1]

    @pytest.mark.parametrize(("mse", "ala", "parent"), [(0.7, 0.3, ["MET"]), (0.3, 0.7, [])], ids=["MSE", "ALA"])
    def test_annotation_names_its_component(self, library, mse, ala, parent):
        source = parse_source(MICROHETEROGENEOUS.format(mse=mse, ala=ala), format="mmcif")
        system = read_structure(source, library=library)[0].system
        assert system.text("parent", system.component_parent).tolist() == parent


class TestPolymerJunctions:
    @pytest.mark.parametrize(("linked", "junctions"), [(False, [2, 2]), (True, [1, 1])])
    def test_pdb_junction_in_mmcif(self, library, atom_line, tmp_path, linked, junctions):
        rows = [atom_line(1, "C", "ALA", "A", 1), atom_line(2, "N", "GLY", "A", 2, (2.33, 2.0, 3.0), element="N")]
        rows += [atom_line(3, "CA", "GLY", "A", 2, (1.0, 3.5, 3.0)), atom_line(4, "C", "GLY", "A", 2, (5.0, 5.0, 5.0))]
        rows += [atom_line(5, "N", "ALA", "A", 3, (6.33, 5.0, 5.0), element="N")]
        link = f"LINK{'':8}C    ALA A   1{'':16}CA   GLY A   2"  # pair 1-2 joined otherwise: the derived `yes`
        text = "\n".join(["SEQRES   1 A    3  ALA GLY ALA", *([link] if linked else []), *rows, "END"])
        pdb, report = read_structure(parse_source(text, format="pdb"), library=library)
        write_structure(pdb, tmp_path / "x.cif", definitions=report.definitions)
        again, _ = read_structure(tmp_path / "x.cif", library=library)
        assert [(value.system.bond_origin == BondOrigin.JUNCTION).sum() for value in (pdb, again)] == junctions


# ALA C on the two-fold axis of P 1 2 1 in ITA numbering (no table, the name stated by _space_group alone) and GLY N
# off it: -x,y,-z fixes C only within a tolerance, which never overrides the junction's one neighbour
JUNCTION_ON_AXIS = """data_PDN
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 GLY
_struct_asym.id A
_struct_asym.entity_id 1
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 C C ALA A 1 1 0.000 5.000 0.000
ATOM 2 N N GLY A 1 2 1.330 5.000 0.000
_cell.length_a 10.0
_cell.length_b 20.0
_cell.length_c 30.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_space_group.name_H-M_alt 'P 1 2 1'
"""


class TestSiteSymmetry:
    def test_unstated_coordinates_decide_nothing(self, library, zinc_site):  # `?` reads 0, on the two-fold axis
        text = zinc_site().replace("0.000 5.000 0.000", "? ? ?")
        _, report = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        summary = report.summary()
        codes = ("special_position_unplaced", "ambiguous_connection", "contact_implausible")
        assert [summary.get(code) for code in codes] == [1, 1, None]

    def test_axis_never_conflicts_topology(self, library):  # the junction's one neighbour at C stands
        system = read_structure(parse_source(JUNCTION_ON_AXIS, format="mmcif"), library=library)[0].system
        carbon = system.text("atom", system.atom_symbol).tolist().index("C")
        assert system.atom_endpoint_state[carbon] == EndpointState.DERIVED_FROM_REFERENCE


class TestZeroOccupancy:
    @pytest.mark.parametrize(("occupancy", "kept"), [("0.00", False), ("0.01", True)], ids=["zero", "small"])
    def test_zero_row_may_leave(self, library, disulfide, occupancy, kept):  # no evidence the atom stays
        text = disulfide.replace("1.300 0.000 0.000 1.00", f"1.300 0.000 0.000 {occupancy}")
        structure, report = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        system = structure.system
        assert ("HG" in system.text("atom", system.atom_symbol).tolist()) is kept
        assert ("placed_leaving_atom" in report.summary()) is kept


# a disulf between the SG of two definition-less ZZZ, B's of element {element}; its order item absent (the default)
LOOSE_DISULFIDE = (
    "data_SS\n"
    + NON_POLYMER.replace(
        "_struct_asym.id A\n_struct_asym.entity_id 1", "loop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\nB 1"
    )
    + "HETATM 1 S SG ZZZ A 1 . 0.000 0.000 0.000 1 A\nHETATM 2 {element} SG ZZZ B 1 . 0.000 2.050 0.000 1 B\n"
    + METALC.replace("metalc1 metalc A LIG . C1 A LIG . X1", "disulf1 disulf A ZZZ . SG B ZZZ . SG")
)


class TestDisulfide:
    @pytest.mark.parametrize(
        ("element", "findings", "order"),
        [("S", [], 1), ("SE", ["disulf_selenium_endpoint"], 0), ("X", [], 0), ("C", ["type_vs_element"], 0)],
    )
    def test_ssbond_reads_as_disulf(self, library, atom_line, element, findings, order):  # one rule for both
        ssbond = f"{'SSBOND   1 ZZZ A    1    ZZZ B    1':<59}{'1555':>6} {'1555':>6}"
        ends = [atom_line(1, "SG", "ZZZ", "A", 1, (0.0, 0.0, 0.0), element="S", record="HETATM")]
        ends.append(atom_line(2, "SG", "ZZZ", "B", 1, (0.0, 2.05, 0.0), element=element, record="HETATM"))
        pdb = parse_source("\n".join([ssbond, *ends, "END"]), format="pdb")
        for source in (pdb, parse_source(LOOSE_DISULFIDE.format(element=element), format="mmcif")):
            structure, report = read_structure(source, library=library, on_conflict="record")
            system, summary = structure.system, report.summary()
            assert [code for code in ("disulf_selenium_endpoint", "type_vs_element") if code in summary] == findings
            assert (system.bond_order.tolist(), known(system.bond_order_state).tolist()) == ([order], [bool(order)])


class TestAltlocExclusion:
    @pytest.mark.parametrize(
        ("partner", "exclusive"), [("ZN  B ZN A 101", False), ("SG  BCYS A   1", True)], ids=["other", "same"]
    )
    def test_exclusive_within_one_residue(self, library, pdb_text, atom_line, partner, exclusive):
        link = next(line for line in pdb_text.splitlines() if line.startswith("LINK"))
        first = atom_line(1, "SG", "CYS", "A", 1, element="S", charge="1-")
        second = atom_line(4, "SG", "CYS", "A", 1, element="S").replace("SG   CYS", "SG  BCYS")
        text = pdb_text.replace(link, link.replace("ZN    ZN A 101", partner)).replace(first, f"{first}\n{second}")
        text = text.replace("SG   CYS A   1", "SG  ACYS A   1").replace("ZN    ZN A 101", "ZN  B ZN A 101")
        _, report = read_structure(parse_source(text, format="pdb"), library=library, on_conflict="record")
        assert ("altloc_exclusive_endpoints" in report.summary()) is exclusive


class TestHydrogenAttribution:
    @pytest.mark.parametrize(
        ("unknown", "hydrogens"),
        [
            ((), {"HA", "HB1", "HB2", "HB3", "HXT", "H3"}),
            (("H3",), {"H3"}),
            (("N",), {"HA", "HB1", "HB2", "HB3", "H3"}),
        ],
        ids=["located", "hydrogen-unlocated", "host-unlocated"],
    )
    def test_undecided_parent_keeps_observed(self, library, unknown, hydrogens):  # only a located H attributes
        structure, _ = read_structure(parse_source(alanine(*unknown), format="mmcif"), library=library)
        system = structure.system
        assert set(system.text("atom", system.atom_symbol[system.atomic_number == 1]).tolist()) == hydrogens

    def test_parent_unknown_cites_hydrogen(self, library):  # every heavy atom of ALA may host the unlocated H3
        _, report = read_structure(parse_source(alanine("H3"), format="mmcif"), library=library)
        found = report.records.where(code="hydrogen_count_unknown")
        cited = [(found[k].detail["cause"], tuple(ref.row for ref in found[k].sources)) for k in range(len(found))]
        assert cited == [("hydrogen_parent_unknown", (5,))] * 6


def zinc_alanine(name: str) -> str:
    r"""``alanine()`` with a deuterium row named ``name`` at its HA site and a ZN (asym B) that a ``metalc`` binds to
    the ALA atom ``name``."""
    asyms = "loop_\n_entity.id\n_entity.type\n1 non-polymer\n2 non-polymer\n"
    asyms += "loop_\n_struct_asym.id\n_struct_asym.entity_id\nA 1\nB 2\n"
    text = alanine().replace(
        "_entity.id 1\n_entity.type non-polymer\n_struct_asym.id A\n_struct_asym.entity_id 1\n", asyms
    )
    text += f"HETATM 7 D {name} ALA A 1 . 2.800 1.300 2.200 1 A\nHETATM 8 ZN ZN ZN B 2 . 4.500 0.500 1.500 2 A\n"
    return text + METALC.replace("A LIG . C1 A LIG . X1", f"A ALA . {name} B ZN . ZN")


class TestObservedIsotope:
    @pytest.mark.parametrize("name", ["HA", "DA"])
    def test_connection_binds_where_row_binds(self, library, name):  # DA is HA's row, so its endpoint is HA
        source = parse_source(zinc_alanine(name), format="mmcif")
        structure, report = read_structure(source, library=library, on_conflict="record")
        system = structure.system
        names = system.text("atom", system.atom_symbol).tolist()
        zinc = names.index("ZN")
        partners = [names[a + b - zinc] for a, b in system.edge_index.T.tolist() if zinc in (a, b)]
        assert partners == ["HA"] and "edge_endpoint_unrealised" not in report.summary()

    def test_hydrogen_beside_deuterium_kept(self, library):  # one site, an H and a D row: neither is dropped
        rows = "HETATM 7 H HA ALA A 1 . 2.800 1.300 2.200 1 A\nHETATM 8 D DA ALA A 1 . 2.800 1.300 2.200 1 A\n"
        source = parse_source(alanine() + rows, format="mmcif")
        structure, _ = read_structure(source, library=library, on_conflict="record")
        observed = structure.system.text("atom", structure.system.atom_symbol[structure.coordinate_mask])
        assert {"HA", "DA"} <= set(observed.tolist())

    @pytest.mark.parametrize(
        ("row", "change"),
        [("D HA", ("", "")), ("D DA", ("", "")), ("D HA", ("C CB", "N CB"))],
        ids=["reference", "mapped-name", "unusable"],
    )
    def test_stated_deuterium_is_kept(self, library, row, change):
        text = (alanine() + f"HETATM 7 {row} ALA A 1 . 2.800 1.300 2.200 1 A\n").replace(*change)
        structure, _ = read_structure(parse_source(text, format="mmcif"), library=library, on_conflict="record")
        system = structure.system
        stated = system.isotope > 0
        assert system.text("atom", system.atom_symbol[stated]).tolist() == ["HA"]
        assert system.isotope[stated].tolist() == [2]

    @pytest.mark.parametrize(("symbol", "differs"), [("D", False), ("T", True)])
    def test_other_mass_number_is_unusable(self, library, symbol, differs):  # DOD's D1 states mass number 2
        rows = f"HETATM 1 O O DOD A 1 . 0.0 0.0 0.0 1 A\nHETATM 2 {symbol} D1 DOD A 1 . 0.96 0.0 0.0 1 A\n"
        source = parse_source(f"data_DOD\n{NON_POLYMER}{rows}", format="mmcif")
        _, report = read_structure(source, library=library, on_conflict="record")
        assert ("element_differs_from_reference" in report.summary()) is differs
