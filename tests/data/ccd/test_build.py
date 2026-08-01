from __future__ import annotations

import pytest

from ofoldx.data.ccd.reader import decode_component
from ofoldx.io.cif import parse_cif


class TestDecodeComponent:
    def test_polymer_attachments(self) -> None:
        block = parse_cif("""data_BETA
_chem_comp.id BETA
_chem_comp.type 'D-beta-peptide, C-gamma linking'
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_leaving_atom_flag
N N N
H2 H Y
CG C N
OD2 O Y
HD2 H N
loop_
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.value_order
_chem_comp_bond.pdbx_aromatic_flag
N H2 SING N
CG OD2 SING N
OD2 HD2 SING N
""").block
        row = decode_component(block)

        assert row["leaving_atom_ids"] == ["H2", "OD2"]
        assert row["non_leaving_atom_ids"] == ["N", "CG", "HD2"]
        assert row["previous_attachment_atom"] == "N"
        assert row["previous_leaving_atoms"] == ["H2"]
        assert row["next_attachment_atom"] == "CG"
        assert row["next_leaving_atoms"] == ["OD2", "HD2"]

    @pytest.mark.parametrize(
        ("flag", "leaving", "non_leaving"),
        [("N", [], ["C"]), ("Y", ["C"], [])],
    )
    def test_partial_leaving_flags(
        self,
        flag: str,
        leaving: list[str],
        non_leaving: list[str],
    ) -> None:
        block = parse_cif(f"""data_PART
_chem_comp.id PART
_chem_comp.type non-polymer
loop_
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.pdbx_leaving_atom_flag
C C {flag}
O O ?
""").block
        row = decode_component(block)

        assert row["leaving_atom_ids"] == leaving
        assert row["non_leaving_atom_ids"] == non_leaving
