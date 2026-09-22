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

"""One invariant, one function.

Each owned name is bound at module level (a ``def`` or an assignment) exactly once across the core packages and
``models``, in its owner module; the owner-only constructors are called, directly or through a classmethod, only inside
their owners.
"""

from __future__ import annotations

import ast

import pytest

SCOPE = (
    "ofoldx.errors",
    "ofoldx.provenance",
    "ofoldx.data",
    "ofoldx.msa",
    "ofoldx.chemistry",
    "ofoldx.io",
    "ofoldx.templates",
    "ofoldx.models",
)

OWNERS = {
    "residue_table": "ofoldx/io/binding.py",
    "pdb_polymer_types": "ofoldx/io/binding.py",
    "derived_polymer_type": "ofoldx/io/binding.py",
    "chemical_state": "ofoldx/io/binding.py",
    "endpoint_queries": "ofoldx/io/binding.py",
    "resolve_endpoints": "ofoldx/io/binding.py",
    "symmetry": "ofoldx/io/symmetry.py",
    "resolve_transforms": "ofoldx/io/symmetry.py",
    "stabilisers": "ofoldx/io/symmetry.py",
    "edge_geometry": "ofoldx/io/symmetry.py",
    "polymer_junctions": "ofoldx/io/connections.py",
    "glycosylation_order": "ofoldx/io/connections.py",
    "polymer_link_declarations": "ofoldx/chemistry/declarations.py",
    "in_use": "ofoldx/chemistry/declarations.py",
    "merge_edge_fields": "ofoldx/chemistry/declarations.py",
    "resolve_definitions": "ofoldx/chemistry/definitions.py",
    "ports": "ofoldx/chemistry/fragments.py",
    "cap_link": "ofoldx/chemistry/fragments.py",
    "fragment_classes": "ofoldx/chemistry/fragments.py",
    "sigma_lone_pair": "ofoldx/chemistry/fragments.py",
    "rule_l": "ofoldx/chemistry/attachments.py",
    "realize_graph": "ofoldx/chemistry/realize.py",
    "valence_excess": "ofoldx/chemistry/realize.py",
    "embed_conformer": "ofoldx/chemistry/rdkit.py",
    "constraint_features": "ofoldx/chemistry/rdkit.py",
    "symmetry_candidates": "ofoldx/chemistry/rdkit.py",
    "replicate_arrays": "ofoldx/data/assembly.py",
    "token_classes": "ofoldx/data/tokenization.py",
    "remap_parity": "ofoldx/data/stereo.py",
    "select": "ofoldx/data/projection.py",
    "mmcif_capability": "ofoldx/io/mmcif/encode.py",
    "pdb_capability": "ofoldx/io/pdb/encode.py",
}

CONSTRUCTORS = {
    "DeclarationTable": "ofoldx/chemistry/declarations.py",
    "EndpointTable": "ofoldx/chemistry/declarations.py",
    "MergedEdges": "ofoldx/chemistry/declarations.py",
    "ResidueTable": "ofoldx/io/binding.py",
    "EndpointResolution": "ofoldx/io/binding.py",
    "ReplicaMap": "ofoldx/data/assembly.py",
}


def binds(node: ast.stmt, name: str) -> bool:
    """A module-level statement that defines or rebinds ``name`` (an import of it is a use, not a definition)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name == name
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return any(isinstance(n, ast.Name) and n.id == name for target in targets for n in ast.walk(target))


def constructs(call: ast.Call, name: str) -> bool:
    """``name(...)``, ``x.name(...)`` or a classmethod constructor ``name.from_rows(...)``."""
    func = call.func
    named = [func, func.value] if isinstance(func, ast.Attribute) else [func]
    return any(getattr(node, "id", None) == name or getattr(node, "attr", None) == name for node in named)


def definition_violations(tree, name: str, owner: str) -> list[str]:
    sites = [module.relpath for module in tree.governed(*SCOPE) for node in module.tree.body if binds(node, name)]
    found = [f"{path} defines {name}; its owner is {owner}" for path in sites if path != owner]
    if sites.count(owner) != 1:
        found.append(f"{owner} defines {name} {sites.count(owner)} times")
    return found


def constructor_violations(tree, name: str, owner: str) -> list[str]:
    return [
        f"{module.relpath}:{node.lineno} calls {name}; only {owner} may"
        for module in tree.governed(*SCOPE)
        if module.relpath != owner
        for node, _, _ in module.walk()
        if isinstance(node, ast.Call) and constructs(node, name)
    ]


class TestOwners:
    @pytest.mark.parametrize(("name", "owner"), list(OWNERS.items()))
    def test_invariant_is_defined_once_in_its_owner(self, name, owner, source_tree):
        assert definition_violations(source_tree, name, owner) == []

    @pytest.mark.parametrize(("name", "owner"), list(CONSTRUCTORS.items()))
    def test_owner_only_constructor_is_called_only_by_its_owner(self, name, owner, source_tree):
        assert constructor_violations(source_tree, name, owner) == []

    @pytest.mark.parametrize(
        ("check", "name", "owner", "files"),
        [
            pytest.param(
                definition_violations,
                "rule_l",
                OWNERS["rule_l"],
                {
                    "ofoldx/chemistry/attachments.py": "def rule_l(): ...\n",
                    "ofoldx/io/connections.py": "def rule_l(): ...\n",
                },
                id="second-definition-elsewhere",
            ),
            pytest.param(
                definition_violations,
                "rule_l",
                OWNERS["rule_l"],
                {"ofoldx/chemistry/attachments.py": "def rule_l(): ...\n\n\ndef rule_l(): ...\n"},
                id="defined-twice-in-owner",
            ),
            pytest.param(
                definition_violations,
                "rule_l",
                OWNERS["rule_l"],
                {"ofoldx/chemistry/attachments.py": "x = 1\n"},
                id="missing-once-migration-is-complete",
            ),
            pytest.param(
                definition_violations,
                "rule_l",
                OWNERS["rule_l"],
                {
                    "ofoldx/chemistry/attachments.py": "def rule_l(): ...\n",
                    "ofoldx/io/connections.py": "rule_l = print\n",
                },
                id="rebinding-elsewhere",
            ),
            pytest.param(
                constructor_violations,
                "ResidueTable",
                CONSTRUCTORS["ResidueTable"],
                {"ofoldx/io/connections.py": "from ofoldx.io.binding import ResidueTable\ntable = ResidueTable()\n"},
                id="constructor-called-outside-owner",
            ),
            pytest.param(
                constructor_violations,
                "ResidueTable",
                CONSTRUCTORS["ResidueTable"],
                {"ofoldx/io/connections.py": "from ofoldx.io.binding import ResidueTable\nResidueTable.from_rows()\n"},
                id="classmethod-constructor-called-outside-owner",
            ),
        ],
    )
    def test_owner_rule_rejects(self, check, name, owner, files, make_tree):
        assert check(make_tree(files), name, owner)
