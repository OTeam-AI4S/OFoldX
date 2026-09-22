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

from collections.abc import Callable

import numpy as np
import pytest

from ofoldx.errors import InvalidRequestError


def _beta_backbone() -> np.ndarray:
    # Residues 1 and 3 have reciprocal C=O...H-N interactions, each across chains.
    # Their O--N, C--H, O--H and C--N distances are 2.8, 2.99, 1.79 and 4 A.
    return np.array(
        [
            [[100, 2, 0], [101, 2, 0], [1.3, 0, 0], [2.5, 0, 0]],
            [[0, 0, 0], [1, 0, 0], [-2.8, 0, 0], [-4, 0, 0]],
            [[-100, 2, 0], [-101, 2, 0], [-8.1, 0, 0], [-9.3, 0, 0]],
            [[-6.8, 0, 0], [-7.8, 0, 0], [-4, 0, 0], [-2.8, 0, 0]],
        ],
        dtype=np.float64,
    )


def _helix_backbone() -> np.ndarray:
    nitrogen = np.stack((np.arange(7) * 10.0, np.zeros(7), np.zeros(7)), axis=-1)
    backbone = np.stack([nitrogen + [offset, 0, 0] for offset in (0.0, 1.0, 11.3, 12.5)], axis=1)
    # Two consecutive i->i+4 bonds define an alpha helix at positions 1..4.
    backbone[0, 2:, 0] = [36.0, 37.2]
    backbone[1, 2:, 0] = [46.0, 47.2]
    return backbone


@pytest.fixture
def secondary_backbones():
    return {"beta": _beta_backbone(), "helix": _helix_backbone()}


_AMINO_ACIDS = frozenset(
    [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    ]
)


@pytest.fixture
def invalid() -> Callable[..., None]:
    """Assert that a call raises ``InvalidRequestError`` with the given code."""

    def check(call, *args, code: str = "value_invalid", **kwargs) -> None:
        with pytest.raises(InvalidRequestError) as caught:
            call(*args, **kwargs)
        assert caught.value.code == code

    return check


@pytest.fixture
def polymers():
    """
    One canonical system from ``(instance, entity, component ids)`` chains; every component has two atoms (``CA``,
    ``CB``) and no bond. A component's monomer type comes from its id: an amino-acid id is protein, ``A C G U N`` RNA,
    ``DA DC DG DT DN`` DNA, any other id none. An entity with a typed component is a polymer, else a non-polymer.
    """

    from ofoldx.data.molecular import MolecularSystem, SystemArrays
    from ofoldx.provenance import Basis, Resolution, state

    unknown, declared, defined = (
        state(Resolution.UNKNOWN, Basis.NONE),
        state(Resolution.KNOWN, Basis.DECLARED),
        state(Resolution.KNOWN, Basis.DEFINED),
    )

    def monomer(name: str) -> int:
        if name in _AMINO_ACIDS:
            return 1
        if name in ("A", "C", "G", "U", "N"):
            return 2
        return 3 if name in ("DA", "DC", "DG", "DT", "DN") else 0

    def build(*chains: tuple[str, str, tuple[str, ...]]) -> MolecularSystem:
        entities = list(dict.fromkeys(entity for _, entity, _ in chains))
        names = [name for _, _, components in chains for name in components]
        family = tuple(dict.fromkeys(names))
        types = np.array([monomer(name) for name in names], dtype=np.uint8)
        sizes = {"system": 1, "entity": len(entities), "instance": len(chains), "component": len(names)}
        sizes |= {"atom": 2 * len(names), "transform": 1}
        columns = {}
        for field in MolecularSystem.SCHEMA.fields:
            rows = sizes.get(field.axis.removesuffix("+1"), 0) + field.axis.endswith("+1")
            fill = unknown if field.state else -1 if field.symbols or field.name == "component_port" else 0
            columns[field.name] = np.full((*field.leading, rows, *field.trailing), fill, dtype=field.dtype)
        typed = dict.fromkeys(entities, False)
        for _, entity, components in chains:
            typed[entity] |= any(monomer(name) for name in components)
        lengths = [len(components) for _, _, components in chains]
        columns |= {
            "system_entity_ptr": np.array([0, len(entities)]),
            "system_instance_ptr": np.array([0, len(chains)]),
            "entity_symbol": np.arange(len(entities)),
            "entity_type": np.array([1 if typed[entity] else 2 for entity in entities]),
            "entity_type_state": np.full(len(entities), declared),
            "instance_symbol": np.arange(len(chains)),
            "instance_entity": np.array([entities.index(entity) for _, entity, _ in chains]),
            "instance_component_ptr": np.concatenate(([0], np.cumsum(lengths))),
            "instance_connections_complete": np.ones(len(chains), dtype=bool),
            "component_symbol": np.array([family.index(name) for name in names]),
            "component_monomer_type": types,
            "component_monomer_type_state": np.where(types > 0, defined, unknown),
            "component_atom_ptr": 2 * np.arange(len(names) + 1),
            "atom_symbol": np.tile([0, 1], len(names)),
            "atom_reference_row": np.full(2 * len(names), -1),
            "transform_rotation": np.eye(3)[None],
            "transform_code": np.array([-1]),
            "transform_state": np.array([defined]),
        }
        symbols = {name: () for name in ("chem_comp_type", "connection_type", "parent", "one_letter", "polymer_type")}
        symbols |= {"entity": tuple(entities), "instance": tuple(name for name, _, _ in chains), "component": family}
        return MolecularSystem.from_arrays(SystemArrays(symbols | {"atom": ("CA", "CB")}, columns))

    return build
