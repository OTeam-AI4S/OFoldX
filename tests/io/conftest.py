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

r"""Small mmCIF and PDB sources shared by the io tests; line builders keep the PDB columns exact; builders of observed
structures for the writer and native tests."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.structure import CoordinateEvidence, CopyKind, Structure, StructureArrays
from ofoldx.provenance import Basis, Resolution, state

MMCIF = """data_TEST
_entry.id TEST
_struct.title 'Two models of a tiny complex'
loop_
_pdbx_audit_revision_history.ordinal
_pdbx_audit_revision_history.revision_date
1 2001-02-03
2 2005-06-07
loop_
_entity.id
_entity.type
_entity.pdbx_description
1 polymer 'tiny peptide'
2 non-polymer 'zinc ion'
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 ALA
1 2 GLY
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_mon_id
_pdbx_poly_seq_scheme.auth_mon_id
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 1 ALA 10 10 ALA ALA P ?
A 1 2 GLY 11 11 GLY GLY P ?
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 1 ? 1.000 2.000 3.000 1.00 10.0 10 ALA P N 1
ATOM 2 D CA . ALA A 1 1 ? 1.500 2.500 3.500 1.00 10.0 10 ALA P CA 1
HETATM 3 ZN ZN . ZN B 2 . ? 5.0 5.0 5.0 . 20.0 101 ZN P ZN 1
ATOM 4 N N . ALA A 1 1 ? 1.100 2.100 3.100 1.00 10.0 10 ALA P N 2
ATOM 5 D CA . ALA A 1 1 ? 1.600 2.600 3.600 1.00 10.0 10 ALA P CA 2
HETATM 6 ZN ZN . ZN B 2 . ? 5.1 5.1 5.1 . 20.0 101 ZN P ZN 2
loop_
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
metalc1 metalc A ALA 1 N B ZN . ZN 1_555
_cell.length_a 10.0
_cell.length_b 20.0
_cell.length_c 30.00
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 1 21 1'
_symmetry.Int_Tables_number 4
loop_
_space_group_symop.id
_space_group_symop.operation_xyz
1 x,y,z
2 -x,y+1/2,-z
"""


@pytest.fixture
def mmcif_text() -> str:
    r"""A two-model polymer + zinc entry: ``pdbx_formal_charge`` and ``pdbx_value_order`` absent, a ``D`` atom."""
    return MMCIF


@pytest.fixture
def atom_line():
    def make(serial, name, residue, chain, number, xyz=(1.0, 2.0, 3.0), *, element="C", charge="", record="ATOM"):
        x, y, z = xyz
        return (
            f"{record:<6}{serial:>5} {name:<4} {residue:>3} {chain:1}{number:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{10.0:6.2f}          {element:>2}{charge:2}"
        )

    return make


@pytest.fixture
def pdb_text(atom_line) -> str:
    r"""A PDB file with COMPND, SEQRES, CRYST1, REMARK 290, SSBOND and LINK (blank symmetry), CONECT, MODRES."""
    lines = [
        "HEADER    HYDROLASE                               12-JAN-98   1ABC              ",
        "TITLE     A TINY TEST",
        "COMPND    MOL_ID: 1;",
        "COMPND   2 MOLECULE: TINY PEPTIDE;",
        "COMPND   3 CHAIN: A, B;",
        "AUTHOR    A.AUTHOR, B.AUTHOR",
        "REVDAT   1   15-MAR-99 1ABC    0",
        "REMARK   2 RESOLUTION.    2.10 ANGSTROMS.",
        "REMARK 290     NNNMMM   OPERATOR",
        "REMARK 290       1555   X,Y,Z",
        "REMARK 290       2555   -X,Y+1/2,-Z",
        "SEQRES   1 A    2  CYS GLY",
        "SEQRES   1 B    2  CYS GLY",
        "MODRES 1ABC MSE A    2  MET  SELENOMETHIONINE",
        "SSBOND   1 CYS A    1    CYS B    1",
        f"LINK{'':8}{'SG':<4} CYS A   1{'':16}{'ZN':<4}  ZN A 101{'':3}{'':>6} {'1555':>6}",
        "CRYST1   10.000   20.000   30.000  90.00  90.00  90.00 P 1 21 1      2",
        atom_line(1, "SG", "CYS", "A", 1, element="S", charge="1-"),
        atom_line(2, "SG", "CYS", "B", 1, element="S"),
        atom_line(3, "ZN", "ZN", "A", 101, element="ZN", record="HETATM"),
        "CONECT    1    2",
        "CONECT    3    1",
        "END",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def placed(polymers):
    r"""
    A deposited, observed ``Structure`` of ``polymers(*chains)`` with carbon atoms: distinct coordinates printed to
    0.01, occupancy 1 (no su) and B 20 stated, a blank charge field, author chains the instance symbols and author
    numbers the positions 1, 2, ... ``pdb=True``: as a PDB source states it, which PDB carries whole (no source asym,
    the linkage INAPPLICABLE×DEFINED, the peptide type KNOWN×DERIVED, coordinates printed to 0.001).
    """

    def build(*chains, pdb: bool = False) -> Structure:
        carbon = polymers(*chains).build()
        carbon.columns["atomic_number"][:] = 6
        carbon.columns["atomic_number_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        if pdb:
            carbon.columns["entity_nstd_linkage_state"][:] = state(Resolution.INAPPLICABLE, Basis.DEFINED)
            carbon.symbols["polymer_type"] = ("polypeptide(L)",)  # the type its SEQRES derives
            carbon.columns["entity_polymer_type"][:] = 0
            carbon.columns["entity_polymer_type_state"][:] = state(Resolution.KNOWN, Basis.DERIVED)
        system = MolecularSystem.from_arrays(carbon)
        arrays = Structure.unobserved(system).build()
        c, symbols = dict(arrays.columns), dict(arrays.symbols)
        n, instances = len(c["coordinates"]), len(c["instance_copy_kind"])
        stated = state(Resolution.KNOWN, Basis.DECLARED)
        c["coordinates"] = np.arange(3 * n, dtype=np.float64).reshape(n, 3) / 4
        c["coordinate_evidence"][:] = CoordinateEvidence.OBSERVED
        c["coordinate_quantum"][:] = 1e-3 if pdb else 0.01
        c["source_sha256"][:] = 1
        for name, value in (("occupancy", 1.0), ("b_factor", 20.0)):
            c[name][:], c[f"{name}_state"][:] = value, stated
        c["occupancy_uncertainty_state"][:] = state(Resolution.UNKNOWN, Basis.NONE)
        c["atom_site_charge_state"][:] = state(Resolution.UNKNOWN, Basis.DECLARED)
        c["instance_copy_kind"][:] = CopyKind.DEPOSITED
        c["instance_author_chain"] = np.arange(instances, dtype=np.int32)
        c["instance_source_asym"] = np.full(instances, -1, np.int32) if pdb else c["instance_author_chain"].copy()
        c["instance_author_chain_state"][:] = stated
        symbols["asym"] = symbols["chain"] = system.symbols()["instance"]
        pointer = system.instance_component_ptr
        ordinal = np.arange(pointer[-1]) - np.repeat(pointer[:-1], np.diff(pointer)) + 1
        symbols["author_number"] = tuple(str(k) for k in range(1, int(ordinal.max(initial=0)) + 1))
        c["component_author_text"] = (ordinal - 1).astype(np.int32)
        c["component_author_number"], c["component_author_number_state"][:] = ordinal, stated
        return Structure.from_arrays(system, StructureArrays(symbols, c))

    return build


@pytest.fixture
def parented(placed):
    r"""``placed(*chains)`` whose first component the source annotates with the parent ``MET``."""

    def build(*chains, pdb: bool = False) -> Structure:
        structure = placed(*chains, pdb=pdb)
        arrays = structure.system.build()
        c = arrays.columns
        c["component_parent_ptr"] = np.minimum(np.arange(len(c["component_parent_ptr"])), 1)
        c["component_parent"] = np.zeros(1, dtype=np.int32)
        arrays.symbols["parent"] = ("MET",)
        return Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())

    return build


# GLY atoms of a deposited dipeptide (no hydrogens), per residue: name, element, x, y, z
GLYCINE = (
    ("N", "N", 0.0, 0.0, 0.0),
    ("CA", "C", 1.46, 0.0, 0.0),
    ("C", "C", 2.0, 1.42, 0.0),
    ("O", "O", 1.25, 2.4, 0.0),
)


@pytest.fixture
def dipeptide():
    r"""``dipeptide(models=1, nstd_linkage="no")``: an mmCIF GLY-GLY chain A, OXT on the second residue."""

    def make(models: int = 1, nstd_linkage: str = "no") -> str:
        rows = []
        for model in range(1, models + 1):
            for number in (1, 2):
                atoms = GLYCINE + ((("OXT", "O", 3.3, 1.5, 0.0),) if number == 2 else ())
                for name, element, x, y, z in atoms:
                    x += 3.8 * (number - 1) + 0.1 * (model - 1)
                    serial = len(rows) + 1
                    rows.append(
                        f"ATOM {serial} {element} {name} . GLY A 1 {number} ? {x:.3f} {y:.3f} {z:.3f} 1.00 10.0 ? "
                        f"{number} GLY A {name} {model}"
                    )
        return DIPEPTIDE.format(nstd_linkage=nstd_linkage, rows="\n".join(rows))

    return make


DIPEPTIDE = """data_GG
_entry.id GG
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage {nstd_linkage}
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 GLY
1 2 GLY
_struct_asym.id A
_struct_asym.entity_id 1
loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.entity_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
_pdbx_poly_seq_scheme.pdb_seq_num
_pdbx_poly_seq_scheme.auth_seq_num
_pdbx_poly_seq_scheme.pdb_mon_id
_pdbx_poly_seq_scheme.auth_mon_id
_pdbx_poly_seq_scheme.pdb_strand_id
_pdbx_poly_seq_scheme.pdb_ins_code
A 1 1 GLY 1 1 GLY GLY A .
A 1 2 GLY 2 2 GLY GLY A .
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
{rows}
"""


@pytest.fixture
def disulfide() -> str:
    r"""An mmCIF of two CYS instances (CB, SG) joined by a disulfide; A models its HG (Conflict
    placed_leaving_atom; the placed CB keeps the residue body out of SG's candidates)."""
    return DISULFIDE


DISULFIDE = """data_SS
_entity.id 1
_entity.type polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
_entity_poly_seq.entity_id 1
_entity_poly_seq.num 1
_entity_poly_seq.mon_id CYS
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 1
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
_atom_site.pdbx_formal_charge
_atom_site.pdbx_PDB_model_num
ATOM 1 C CB CYS A 1 1 -1.810 0.000 0.000 1.00 ? 1
ATOM 2 S SG CYS A 1 1 0.000 0.000 0.000 1.00 ? 1
ATOM 3 H HG CYS A 1 1 1.300 0.000 0.000 1.00 ? 1
ATOM 4 C CB CYS B 1 1 -1.810 2.050 0.000 1.00 ? 1
ATOM 5 S SG CYS B 1 1 0.000 2.050 0.000 1.00 ? 1
loop_
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
disulf1 disulf A CYS 1 SG B CYS 1 SG 1_555 1_555
"""


@pytest.fixture
def zinc_site():
    r"""``zinc_site(reverse=False, ids=("1", "2"))``: an mmCIF of a CYS SG bound to a ZN on the two-fold axis of
    P 1 2 1, stated from SG (``1_555``) and from its mate (``2_555``), one physical edge; ``reverse`` lists the
    atom_site and struct_conn rows backwards, ``ids`` names x,y,z and -x,y,-z (and every code naming them)."""

    def make(reverse: bool = False, ids: tuple[str, str] = ("1", "2")) -> str:
        identity, axis = ids
        rows = [f"metalc{k} metalc A CYS 1 SG B ZN . ZN {code}_555 {identity}_555" for k, code in enumerate(ids, 1)]
        step = -1 if reverse else 1
        return ZINC_SITE.format(
            atoms="\n".join(ZINC_ATOMS[::step]), connections="\n".join(rows[::step]), identity=identity, axis=axis
        )

    return make


ZINC_ATOMS = (
    "ATOM 1 C CB CYS A 1 1 2.900 5.000 2.800 1.00 ? 1",
    "ATOM 2 S SG CYS A 1 1 1.400 5.000 1.800 1.00 ? 1",
    "HETATM 3 ZN ZN ZN B 2 . 0.000 5.000 0.000 1.00 ? 1",
)
ZINC_SITE = """data_ZS
loop_
_entity.id
_entity.type
1 polymer
2 non-polymer
_entity_poly.entity_id 1
_entity_poly.type 'polypeptide(L)'
_entity_poly.nstd_linkage no
_entity_poly_seq.entity_id 1
_entity_poly_seq.num 1
_entity_poly_seq.mon_id CYS
loop_
_struct_asym.id
_struct_asym.entity_id
A 1
B 2
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
_atom_site.pdbx_formal_charge
_atom_site.pdbx_PDB_model_num
{atoms}
loop_
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
{connections}
_cell.length_a 10.0
_cell.length_b 20.0
_cell.length_c 30.0
_cell.angle_alpha 90.0
_cell.angle_beta 90.0
_cell.angle_gamma 90.0
_symmetry.space_group_name_H-M 'P 1 2 1'
loop_
_space_group_symop.id
_space_group_symop.operation_xyz
{identity} x,y,z
{axis} -x,y,-z
"""
