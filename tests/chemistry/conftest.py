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

r"""
Fixtures of the chemistry tests over the builders of ``builders``: ``items`` and ``glycine`` build synthetic items,
``library`` writes definitions into an artifact and loads it, ``ligand`` builds the value of one SMILES ligand and
``edit`` rebuilds a value with columns changed (``ccd`` and ``archive`` live in ``tests/conftest.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from ofoldx.chemistry.construction import Molecule, Recipe, build_system
from ofoldx.chemistry.definitions import (
    ChemicalComponent,
    ComponentItems,
    ComponentLibrary,
    define_component,
    load_library,
    write_artifact,
)
from ofoldx.chemistry.smiles import read_smiles
from ofoldx.data.molecular import DefinitionScope, MolecularSystem

from .builders import glycine_columns, make_items, provenance


@pytest.fixture(name="items")
def items_fixture() -> Callable[..., ComponentItems]:
    return make_items


@pytest.fixture(name="glycine")
def glycine_fixture() -> Callable[[], dict]:
    return glycine_columns


@pytest.fixture(name="define")
def define_fixture() -> Callable[..., ChemicalComponent]:
    def define(items: ComponentItems, scope: DefinitionScope = DefinitionScope.ARCHIVE, **options) -> ChemicalComponent:
        return define_component(items, scope=scope, **options)

    return define


@pytest.fixture
def library(tmp_path) -> Callable[..., ComponentLibrary]:
    r"""``library(components, content_version="test")``: the components written as an artifact and loaded."""

    def build(components: Iterable[ChemicalComponent], content_version: str = "test") -> ComponentLibrary:
        path = tmp_path / f"{content_version}.safetensors"
        write_artifact(components, path, provenance(content_version))
        return load_library(path)

    return build


@pytest.fixture
def ligand(library) -> Callable[[str], MolecularSystem]:
    r"""``ligand(smiles)``: the value of one instance ``L`` of the SMILES ligand ``LIG`` (no library entry needed)."""

    empty = library([], content_version="empty")

    def build(smiles: str) -> MolecularSystem:
        component = read_smiles(smiles, component_id="LIG").component
        system, _, _ = build_system(Recipe({"L": Molecule(("LIG",))}, {"L": "L"}), empty, {"LIG": component})
        return system

    return build


@pytest.fixture
def edit() -> Callable[..., MolecularSystem]:
    r"""``edit(system, column=(index, value), ...)``: the value rebuilt with those cells set."""

    def rebuild(system: MolecularSystem, **cells) -> MolecularSystem:
        arrays = system.build()
        for column, (index, value) in cells.items():
            arrays.columns[column][index] = value
        return MolecularSystem.from_arrays(arrays)

    return rebuild
