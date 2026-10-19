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

"""Behaviour of ofoldx.templates.coordinates: the chain projection."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.molecular import MolecularSystem
from ofoldx.data.structure import Structure
from ofoldx.data.tokenization import TOKENS
from ofoldx.errors import InsufficientInformationError
from ofoldx.templates import chain_coordinates


class TestChainCoordinates:
    def test_projects_every_position(self, structure):
        chain = chain_coordinates(structure, ["X"]).chains["X"]
        assert chain.component_ids == ("MET", "LYS", "GLY")
        assert [TOKENS[t] for t in chain.token_class] == ["M", "K", "G"]
        coordinates, present = chain.gather([2, 0], ["CA", None])
        assert present.tolist() == [[False, False], [True, False]]
        assert coordinates[1, 0].tolist() == [1.0, 2.0, 3.0]

    def test_label_namespace(self, structure):
        assert list(chain_coordinates(structure, ["B"], namespace="label").chains) == ["B"]

    def test_unknown_chain(self, structure, invalid):
        invalid(chain_coordinates, structure, ["Z"], code="projection_invalid")

    def test_ambiguous_author_chain(self, make_structure):
        with pytest.raises(InsufficientInformationError) as caught:
            chain_coordinates(make_structure(chains=(0, 0, 1)), ["X"])
        assert caught.value.code == "template_chain_ambiguous"

    def test_parents_are_the_definitions(self, structure):  # never the occurrence's annotation
        arrays = structure.system.build()
        c = arrays.columns
        arrays.symbols["parent"] = ("ALA", "GLY")
        for name, code in (("component_definition_parent", 0), ("component_parent", 1)):
            c[f"{name}_ptr"] = (np.arange(len(c[f"{name}_ptr"])) >= 2).astype(np.int64)
            c[name] = np.array([code], dtype=np.int32)
        annotated = Structure.from_arrays(MolecularSystem.from_arrays(arrays), structure.build())
        assert chain_coordinates(annotated, ["X"]).chains["X"].parent_component_ids == ((), ("ALA",), ())
