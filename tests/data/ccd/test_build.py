from __future__ import annotations

import pytest

from ofoldx.data.ccd.build import row_from_block


class TestRowFromBlock:
    def test_polymer_attachments(self) -> None:
        row = row_from_block(
            {
                "_chem_comp.id": ["BETA"],
                "_chem_comp.type": ["D-beta-peptide, C-gamma linking"],
                "_chem_comp_atom.atom_id": ["N", "H2", "CG", "OD2", "HD2"],
                "_chem_comp_atom.type_symbol": ["N", "H", "C", "O", "H"],
                "_chem_comp_atom.pdbx_leaving_atom_flag": ["N", "Y", "N", "Y", "N"],
                "_chem_comp_bond.atom_id_1": ["N", "CG", "OD2"],
                "_chem_comp_bond.atom_id_2": ["H2", "OD2", "HD2"],
                "_chem_comp_bond.value_order": ["SING", "SING", "SING"],
                "_chem_comp_bond.pdbx_aromatic_flag": ["N", "N", "N"],
            }
        )

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
        row = row_from_block(
            {
                "_chem_comp.id": ["PART"],
                "_chem_comp.type": ["non-polymer"],
                "_chem_comp_atom.atom_id": ["C", "O"],
                "_chem_comp_atom.type_symbol": ["C", "O"],
                "_chem_comp_atom.pdbx_leaving_atom_flag": [flag, "?"],
            }
        )

        assert row["leaving_atom_ids"] == leaving
        assert row["non_leaving_atom_ids"] == non_leaving
