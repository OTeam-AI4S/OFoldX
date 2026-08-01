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

"""Behavior of ofoldx.models.alphafold2.relax."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.chemistry.construction import PolymerSequence, Recipe, RecipeBond, build_system
from ofoldx.data.molecular import AtomSite, BondKind
from ofoldx.data.structure import CoordinateEvidence, Structure, replace_coordinates
from ofoldx.errors import UnsupportedError
from ofoldx.models.alphafold2.relax import AlphaFold2AmberRelaxation, alphafold2_structural_violations


class TestViolations:
    def test_gamma_carbon_clash(self, peptide):
        structure = peptide("LYS", "LYS")
        names = structure.system.text("atom", structure.system.atom_symbol)
        gamma = np.flatnonzero(names == "CG")
        # only the two CG atoms keep coordinates, both at the origin: a between-residue clash and nothing else
        clashing, _ = replace_coordinates(
            structure,
            gamma,
            np.zeros((len(gamma), 3)),
            scope=np.ones(len(names), dtype=bool),
            evidence=CoordinateEvidence.OBSERVED,
        )
        assert alphafold2_structural_violations(clashing).tolist() == [1.0, 1.0]


class TestNonlinearGraph:
    @pytest.fixture(params=["nick", "cycle", "crosslink"])
    def structure(self, request, library) -> Structure:
        linked = (False,) if request.param == "nick" else None
        bonds = ()
        if request.param == "cycle":
            bonds = (RecipeBond(AtomSite("A", 1, "C"), AtomSite("A", 0, "N"), 1, BondKind.COVALENT),)
        if request.param == "crosslink":
            first, second = AtomSite("A", 0, "SG"), AtomSite("A", 1, "SG")
            bonds = (RecipeBond(first, second, 1, BondKind.COVALENT, (("HG",), ("HG",))),)
        recipe = Recipe({"P": PolymerSequence("polypeptide(L)", ("CYS", "CYS"), linked)}, {"A": "P"}, bonds)
        system, _, _ = build_system(recipe, library)
        return Structure.unobserved(system)

    @pytest.mark.parametrize(
        "operation", [alphafold2_structural_violations, AlphaFold2AmberRelaxation().relax], ids=["violations", "relax"]
    )
    def test_refused(self, structure, operation):
        with pytest.raises(UnsupportedError, match="model_capacity"):
            operation(structure)
