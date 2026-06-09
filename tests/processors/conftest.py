from __future__ import annotations

import numpy as np

from ofoldx.data import ReferenceComponent, ResidueKey, Structure
from ofoldx.data.structure import AtomRecord


class _FakeCcd:
    def __init__(self, components: dict[str, ReferenceComponent]) -> None:
        self._components = components

    def get(self, component_name: str) -> ReferenceComponent | None:
        return self._components.get(component_name)


def _residue(name: str, residue: ResidueKey, names_to_element: dict[str, str]) -> tuple[AtomRecord, ...]:
    return tuple(
        AtomRecord(atom_name, element, name, residue, float(index), 0.0, 0.0, mol_type="protein")
        for index, (atom_name, element) in enumerate(names_to_element.items())
    )


def _component(
    atom_map: dict[str, int],
    coords: np.ndarray | None = None,
    *,
    charge: np.ndarray | None = None,
    bonds: tuple[tuple[str, str, str], ...] = (),
    elements: tuple[str, ...] = (),
) -> ReferenceComponent:
    n = len(atom_map)
    return ReferenceComponent(
        atom_map=atom_map,
        coords=(np.arange(n * 3, dtype=np.float32).reshape(n, 3) if coords is None else coords.astype(np.float32)),
        mask=np.ones(n, dtype=bool),
        charge=np.zeros(n, dtype=np.float32) if charge is None else charge.astype(np.float32),
        bonds=bonds,
        elements=elements,
    )


def _two_residue_protein() -> tuple[Structure, _FakeCcd]:
    asp, gly = ResidueKey("A", 10), ResidueKey("A", 11)
    asp_atoms = {"N": "N", "CA": "C", "C": "C", "O": "O", "CB": "C", "CG": "C", "OD1": "O", "OD2": "O"}
    structure = Structure(
        (*_residue("ASP", asp, asp_atoms), *_residue("GLY", gly, {"N": "N", "CA": "C", "C": "C", "O": "O"})),
        bonds=((2, 8),),  # ASP.C -> GLY.N implicit backbone link must be dropped
    )
    ccd = _FakeCcd(
        {
            "ASP": _component({"N": 0, "CA": 1, "C": 2, "O": 3, "CB": 4, "CG": 5, "OD1": 6, "OD2": 7}),
            "GLY": _component({"N": 0, "CA": 1, "C": 2, "O": 3}),
        }
    )
    return structure, ccd
