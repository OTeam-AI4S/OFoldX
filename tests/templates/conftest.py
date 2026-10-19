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

r"""A structure with two protein copies and a ligand, and a template on its first chain, shared by the template
tests."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.structure import CoordinateEvidence, Structure, StructureArrays
from ofoldx.provenance import Basis, Resolution, state
from ofoldx.templates import Template, chain_coordinates


@pytest.fixture
def make_structure(polymers):
    """Proteins A and B (entity 1: MET LYS GLY) and ligand L; A's first two residues predicted; ``chains`` are the
    author chains of A, B, L (codes into ``X Y``); label asyms are A B C."""

    def make(chains=(0, 1, 0)) -> Structure:
        system = polymers(("A", "1", ("MET", "LYS", "GLY")), ("B", "1", ("MET", "LYS", "GLY")), ("L", "2", ("HEM",)))
        arrays = Structure.unobserved(system).build()
        c = arrays.columns
        c["coordinates"][:4] = np.arange(1, 13).reshape(4, 3)
        c["coordinate_evidence"][:4] = CoordinateEvidence.PREDICTED
        c["instance_author_chain"][:] = chains
        c["instance_author_chain_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        c["instance_source_asym"][:] = (0, 1, 2)
        symbols = {**arrays.symbols, "chain": ("X", "Y"), "asym": ("A", "B", "C")}
        return Structure.from_arrays(system, StructureArrays(symbols, c))

    return make


@pytest.fixture
def structure(make_structure):
    return make_structure()


@pytest.fixture
def template(structure):
    coordinates = chain_coordinates(structure, ["X"])
    return lambda *alignments: Template(coordinates, alignments)
