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

"""Behaviour of ofoldx.data.tokenization: the canonical vocabulary, component classes and token layouts."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem, MonomerType
from ofoldx.data.tokenization import (
    TOKEN_INDEX,
    TOKENS,
    Encoding,
    class_sequence,
    encode_sequence,
    select_tokens,
    token_classes,
    tokenize,
)
from ofoldx.errors import UnsupportedError
from ofoldx.provenance import Basis, Resolution, state

KDEF = state(Resolution.KNOWN, Basis.DEFINED)
UN = state(Resolution.UNKNOWN, Basis.NONE)


@pytest.mark.parametrize(
    ("index", "token"),
    [(0, "<pad>"), (3, "<unk>"), (6, "A"), (32, "RNA_A"), (48, "RNA_X"), (52, "DNA_T"), (66, "|"), (71, "OTHER")],
)
def test_vocabulary_layout(index, token):
    assert len(TOKENS) == 72 and TOKENS[index] == token and TOKEN_INDEX[token] == index


def classes(system):
    found = token_classes(system)
    return [(TOKENS[t], Encoding(e)) for t, e in zip(found.token, found.encoding)]


@pytest.fixture
def renamed(system_arrays):
    """The fixture with ALA renamed, its definition's parent and monomer type set and, with ``annotated``, the
    occurrence's parent annotation (the fixture's GLY): (name, parent, monomer, monomer state, polymer)."""

    def make(name, parent=None, monomer=1, monomer_state=KDEF, polymer="polypeptide(L)", annotated=False):
        symbols = dict(system_arrays().symbols)
        symbols["component"] = (name, *symbols["component"][1:])
        symbols["parent"] = ("GLY", parent or "MET")
        symbols["polymer_type"] = (polymer,)
        arrays = system_arrays(symbols=symbols)
        c = arrays.columns
        c["component_monomer_type"][0], c["component_monomer_type_state"][0] = monomer, monomer_state
        if parent is not None:
            c["component_definition_parent_ptr"] = np.array([0, 1, 1, 2, 2])
            c["component_definition_parent"] = np.array([1, 0])
        if annotated:
            c["component_parent_ptr"] = np.array([0, 1, 1, 2, 2])
            c["component_parent"] = np.array([0, 0])
        return MolecularSystem.from_arrays(arrays)

    return make


class TestTokenClasses:
    def test_fixture_classes(self, system):
        assert classes(system) == [
            ("A", Encoding.EXACT),
            ("G", Encoding.EXACT),
            ("OTHER", Encoding.NON_POLYMER),
            ("OTHER", Encoding.NON_POLYMER),
        ]

    @pytest.mark.parametrize(
        ("name", "parent", "monomer", "monomer_state", "polymer", "expected"),
        [
            pytest.param("MSE", "MET", 1, KDEF, "polypeptide(L)", ("M", Encoding.PARENT), id="parent"),
            pytest.param("DU", None, 3, KDEF, "polydeoxyribonucleotide", ("DNA_T", Encoding.CROSS_PYRIMIDINE), id="du"),
            pytest.param("PSU", "U", 2, KDEF, "polyribonucleotide", ("RNA_U", Encoding.PARENT), id="psu"),
            pytest.param("SEC", None, 1, KDEF, "polypeptide(L)", ("U", Encoding.EXACT), id="selenocysteine"),
            pytest.param("ACE", None, 0, UN, "polypeptide(L)", ("X", Encoding.UNMAPPED), id="class-from-chain"),
            pytest.param("XYZ", None, 1, KDEF, "other", ("X", Encoding.UNMAPPED), id="own-class-in-other"),
            pytest.param("XYZ", None, 2, KDEF, "polypeptide(L)", ("RNA_X", Encoding.UNMAPPED), id="own-class-wins"),
            pytest.param("XYZ", None, 0, UN, "other", ("OTHER", Encoding.UNMAPPED), id="undeterminable"),
        ],
    )
    def test_class_of_a_component(self, name, parent, monomer, monomer_state, polymer, expected, renamed):
        assert classes(renamed(name, parent, monomer, monomer_state, polymer))[0] == expected

    @pytest.mark.parametrize(
        ("parent", "expected"), [("MET", ("M", Encoding.PARENT)), (None, ("X", Encoding.UNMAPPED))]
    )
    def test_annotation_never_maps_the_class(self, parent, expected, renamed):
        assert classes(renamed("MSE", parent, annotated=True))[0] == expected


class TestEncodeSequence:
    @pytest.mark.parametrize(
        ("text", "monomer", "tokens"),
        [
            ("ACT", MonomerType.RNA, ["RNA_A", "RNA_C", "RNA_U"]),
            ("ACU", MonomerType.DNA, ["DNA_A", "DNA_C", "DNA_T"]),
            ("MX-?", MonomerType.PROTEIN, ["M", "X", "-", "?"]),
            ("a#", MonomerType.PROTEIN, ["<unk>", "<unk>"]),
        ],
    )
    def test_classes(self, text, monomer, tokens):
        assert [TOKENS[i] for i in encode_sequence(text, monomer)] == tokens

    def test_non_polymer_monomer_invalid(self, invalid):
        invalid(encode_sequence, "A", MonomerType.NONE)


@pytest.fixture
def layout(system):
    residue = np.array([True, True, False, False])
    centre = np.array([1, 7, -1, -1])
    return tokenize(system, residue_token=residue, centre_atom=centre, atom_mask=system.atomic_number != 1)


class TestClassSequence:
    def test_letters(self):
        classes = [TOKEN_INDEX[t] for t in ("M", "X", "RNA_U", "RNA_X", "DNA_T", "DNA_N")]
        assert class_sequence(np.array(classes, dtype=np.uint8)) == "MXUNTN"

    def test_class_outside_vocabulary_invalid(self, invalid):
        invalid(class_sequence, np.array([len(TOKENS)], dtype=np.uint8))

    @pytest.mark.parametrize("token", ["OTHER", "<unk>", "-"])
    def test_unrepresentable(self, token):
        with pytest.raises(UnsupportedError) as caught:
            class_sequence(np.array([TOKEN_INDEX["A"], TOKEN_INDEX[token]], dtype=np.uint8))
        assert caught.value.code == "query_class_unrepresentable"


class TestTokenize:
    def test_residues_and_atoms(self, layout):
        assert layout.token_component.tolist() == [0, 1, 2, 2, 2, 2, 2, 2, 3]
        assert layout.token_centre_atom.tolist()[:3] == [1, 7, 11]
        assert [TOKENS[t] for t in layout.token_class[:3]] == ["A", "G", "OTHER"]

    def test_masked_atoms_have_no_token(self, layout):
        assert layout.atom_token[5] == -1 and layout.atom_token[0] == 0 and layout.atom_token[17] == 8
        assert layout.token_atoms[layout.token_atom_ptr[0] : layout.token_atom_ptr[1]].tolist() == [0, 1, 2, 3, 4]

    def test_centre_must_be_visible(self, system, invalid):
        residue = np.array([True, False, False, False])
        mask = system.atomic_number != 1
        invalid(tokenize, system, residue_token=residue, centre_atom=np.array([5, -1, -1, -1]), atom_mask=mask)

    @pytest.mark.parametrize(
        "change",
        [{"atom_mask": np.arange(18)}, {"residue_token": np.array([1, 1, 0, 0])}],
        ids=["atom_mask", "residue_token"],
    )
    def test_masks_must_be_boolean(self, system, invalid, change):  # never read as atom indices or counts
        arguments = {"residue_token": np.array([True, True, False, False]), "centre_atom": np.array([1, 7, -1, -1])}
        invalid(tokenize, system, **(arguments | change))

    def test_select_needs_token_mask(self, layout, invalid):  # np.arange is no keep mask
        invalid(select_tokens, layout, np.arange(len(layout.token_component)))

    def test_select_tokens(self, layout):
        keep = np.zeros(len(layout.token_component), dtype=bool)
        keep[[1, 8]] = True
        selected = select_tokens(layout, keep)
        assert selected.token_component.tolist() == [1, 3]
        assert selected.atom_token[17] == 1 and selected.atom_token[0] == -1
        assert selected.token_atoms.tolist() == [6, 7, 8, 9, 10, 17]

    def test_select_from_no_tokens(self, system):
        hidden = np.zeros(len(system.atom_symbol), dtype=bool)
        none = tokenize(system, residue_token=np.zeros(4, bool), centre_atom=np.full(4, -1), atom_mask=hidden)
        assert (select_tokens(none, np.zeros(0, dtype=bool)).atom_token == -1).all()
