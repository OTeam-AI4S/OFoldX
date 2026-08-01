from __future__ import annotations

import pytest
import torch

from ofoldx.data import (
    TOKEN_INDEX,
    ChemicalAtom,
    ChemicalComponent,
    MolecularSystem,
    Molecule,
    PolymerSequence,
    select_tokens,
    tokenize_molecular_system,
)


def _system() -> MolecularSystem:
    def component(component_id: str) -> ChemicalComponent:
        atom_ids = ("N", "CA") if component_id == "ALA" else ("C", "O")
        return ChemicalComponent(
            component_id,
            (
                ChemicalAtom(atom_ids[0], "N" if atom_ids[0] == "N" else "C"),
                ChemicalAtom(atom_ids[1], "C" if atom_ids[1] == "CA" else "O"),
                ChemicalAtom("H", "H"),
            ),
            frozenset(),
        )

    return MolecularSystem.from_entities(
        {
            "protein": PolymerSequence(
                "protein",
                (component("ALA"), component("MSE")),
            ),
            "ligand": Molecule((component("LIG"),)),
        },
        {"A": "protein", "L": "ligand"},
    )


class TestTokenizeMolecularSystem:
    def test_default(self) -> None:
        layout = tokenize_molecular_system(_system())

        assert layout.token_component_index.tolist() == [0, 1, 1, 2, 2]
        assert layout.token_atom_index.tolist() == [-1, 3, 4, 6, 7]
        assert layout.token_type_index.tolist() == [
            TOKEN_INDEX["A"],
            TOKEN_INDEX["X"],
            TOKEN_INDEX["X"],
            TOKEN_INDEX["OTHER"],
            TOKEN_INDEX["OTHER"],
        ]
        assert layout.token_instance_index.tolist() == [0, 0, 0, 1, 1]
        assert layout.token_entity_index.tolist() == [0, 0, 0, 1, 1]
        assert layout.token_residue_index.tolist() == [0, 1, 1, 0, 0]
        assert layout.token_symmetry_index.tolist() == [0, 0, 0, 0, 0]
        assert layout.token_center_atom_index.tolist() == [1, 3, 4, 6, 7]
        assert layout.token_atom_ptr.tolist() == [0, 2, 3, 4, 5, 6]
        assert layout.atom_index.tolist() == [0, 1, 3, 4, 6, 7]

    def test_expanded_components(self) -> None:
        layout = tokenize_molecular_system(
            _system(),
            atomize_nonstandard_polymers=False,
            expanded_components=(0,),
        )

        assert layout.token_component_index.tolist() == [0, 0, 1, 2, 2]
        assert layout.token_atom_index.tolist() == [0, 1, -1, 6, 7]
        assert layout.token_atom_ptr.tolist() == [0, 1, 2, 4, 5, 6]
        assert layout.atom_index.tolist() == [0, 1, 3, 4, 6, 7]

    def test_component_mask(self) -> None:
        layout = tokenize_molecular_system(
            _system(),
            component_mask=torch.tensor([True, True, False]),
        )

        assert layout.token_component_index.tolist() == [0, 1, 1]

    def test_atom_mask(self) -> None:
        with pytest.raises(ValueError, match="canonical instance 1 has no visible heavy atoms"):
            tokenize_molecular_system(
                _system(),
                atom_mask=torch.tensor([True] * 6 + [False] * 3),
            )


def test_select_tokens() -> None:
    layout = tokenize_molecular_system(
        _system(),
        atomize_nonstandard_polymers=False,
        expanded_components=(0,),
    )

    selected = select_tokens(layout, torch.tensor([2, 4]))

    assert selected.token_component_index.tolist() == [1, 2]
    assert selected.token_atom_index.tolist() == [-1, 7]
    assert selected.token_type_index.tolist() == [TOKEN_INDEX["X"], TOKEN_INDEX["OTHER"]]
    assert selected.token_instance_index.tolist() == [0, 1]
    assert selected.token_residue_index.tolist() == [1, 0]
    assert selected.token_center_atom_index.tolist() == [3, 7]
    assert selected.token_atom_ptr.tolist() == [0, 2, 3]
    assert selected.atom_index.tolist() == [3, 4, 7]
