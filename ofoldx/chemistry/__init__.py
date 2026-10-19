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
Component chemistry: component definitions and their versioned library, reference-graph primitives, attachment
chemistry and the one lowering of its results into graph arrays, construction from recipes, SMILES, and RDKit views
of a determined graph.

The package depends on ``ofoldx.errors``, ``ofoldx.provenance`` and ``ofoldx.data`` only; ``ofoldx.io`` decodes sources
into the input types defined here, never the reverse.
"""

from .attachments import resolve_attachments
from .construction import Molecule, PolymerSequence, Recipe, RecipeBond, build_system, polymer_sequence
from .declarations import AssumptionRule, AssumptionSet
from .definitions import (
    ChemicalComponent,
    ComponentLibrary,
    Definitions,
    configured_library,
    get_component,
    load_library,
)
from .rdkit import constraint_features, embed_conformer, symmetry_candidates, to_rdkit
from .realize import valence_excess
from .smiles import SmilesImport, read_smiles

__all__ = [
    "ChemicalComponent",
    "ComponentLibrary",
    "Definitions",
    "load_library",
    "configured_library",
    "get_component",
    "AssumptionRule",
    "AssumptionSet",
    "resolve_attachments",
    "Recipe",
    "RecipeBond",
    "PolymerSequence",
    "polymer_sequence",
    "Molecule",
    "build_system",
    "read_smiles",
    "SmilesImport",
    "valence_excess",
    "embed_conformer",
    "constraint_features",
    "symmetry_candidates",
    "to_rdkit",
]
