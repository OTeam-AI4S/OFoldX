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

"""Gate G1: the dependency rule L1–L9 of design §2.1, read from the source tree.

Each rule is one row: a check returning its violations and a counterexample the check must reject. The rules govern
every module; a package that does not exist yet obeys them trivially.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import NamedTuple

import pytest

CORE = (
    "ofoldx.reference.space_groups",
    "ofoldx.errors",
    "ofoldx.provenance",
    "ofoldx.data",
    "ofoldx.msa",
    "ofoldx.chemistry",
    "ofoldx.io",
    "ofoldx.templates",
)
#: Core packages beside the ones after them rather than below: no other core package imports them.
BESIDE = ("ofoldx.msa",)
#: Above models: the user-facing entry points that read files. ``pipelines``, ``results`` and ``analysis`` are the
#: base classes and values the model families build on, so they sit on the models side and never import io.
UPPER = ("ofoldx.runners", "ofoldx.api", "ofoldx.auto")
SCOPE = (*CORE, "ofoldx.models", "ofoldx.pipelines", "ofoldx.results", "ofoldx.analysis", *UPPER)
OFFLINE = ("tools", "ofoldx.reference.space_groups.build")
COMPUTE = ("torch", "tensordict", "danling")
#: RDKit and the RDKit-based ChEMBL structure pipeline (SMILES standardisation): one confined dependency.
RDKIT = ("rdkit", "chembl_structure_pipeline")
READ_PATH_FORBIDDEN = (*COMPUTE, *RDKIT)
RDKIT_HOMES = ("ofoldx.chemistry.rdkit", "ofoldx.chemistry.smiles")
UNTYPED = ("ValueError", "TypeError", "KeyError")
TYPED = (
    "OFoldXError",
    "ContradictionError",
    "InsufficientInformationError",
    "UnsupportedError",
    "InvalidRequestError",
    "ResourceLimitError",
)
FROZEN_WRITES = (("dataclasses", "replace"), ("copy", "replace"))

#: Interim exemptions from the design scope of L6 (no function-local import, all of OFoldX) and L9 (typed failures, the
#: core packages and ``models``), one reason per module, until the maintainer rules on reading 3 of
#: ``wu0/B-architecture.md``. A module leaves when it is fixed or rewritten; an entry that no longer excuses a violation
#: fails, so the list only shrinks.
EXEMPT: dict[str, dict[str, str]] = {
    "L6-no-local-imports": {
        "ofoldx/artifacts.py": "huggingface_hub and safetensors.torch deferred to save_model",
        "ofoldx/metrics/structure.py": "the torch geometry module (ofoldx.modules.geometry) deferred to the alignment",
        "ofoldx/modules/backend/_apex.py": "optional accelerator backend (apex) imported only when selected",
        "ofoldx/modules/backend/_cueq.py": "optional accelerator backend (cuequivariance) imported only when selected",
        "ofoldx/modules/backend/_deepspeed.py": "optional accelerator backend (deepspeed) imported only when selected",
        "ofoldx/modules/backend/attention.py": "breaks the cycle with modules.attention.multihead, which imports it",
        "ofoldx/modules/backend/knn.py": "sibling modules functional and geometry deferred to the eager reference path",
        "ofoldx/pipelines/composite.py": "ofoldx.hub (huggingface_hub) deferred to save",
        "ofoldx/pipelines/paired.py": "ofoldx.hub (huggingface_hub) deferred to save",
    },
    "L9-typed-failures": {
        "ofoldx/models/registry.py": "family registration raises ValueError for an empty or duplicate id",
    },
}


def layer(name: str) -> str:
    """The layer of a module. Modules outside the named packages (ofoldx.modules, ofoldx.pipelines, ofoldx.results,
    ofoldx.analysis, model.py, hub.py, …) are the model infrastructure and sit on the models side of the io/models
    split."""
    if any(name == package or name.startswith(package + ".") for package in OFFLINE):
        return "tools"
    for package in CORE:
        if name == package or name.startswith(package + "."):
            return package
    if name == "ofoldx.models" or name.startswith("ofoldx.models."):
        return "models"
    if name == "ofoldx" or any(name == p or name.startswith(p + ".") for p in UPPER):
        return "upper"
    return "infrastructure"


def may_import(source: str, target: str) -> bool:
    if target == "tools":
        return source == "tools"
    if source in CORE:
        return target in CORE[: CORE.index(source) + 1] and (target == source or target not in BESIDE)
    if source in ("models", "infrastructure"):
        return target not in ("ofoldx.io", "upper")
    return True


def l4_permits(module: str, imp) -> bool:
    if imp.under(*RDKIT):
        return module in RDKIT_HOMES and imp.function is not None
    return imp.under("multimolecule.graph")


def target_layer(tree, path: str) -> str:
    """The layer an import lands in; a name the top package re-exports is the top package's."""
    return "upper" if tree.owner(path) == "ofoldx" and layer(path) == "infrastructure" else layer(path)


def exempted(rule: str, modules, exempt: dict[str, str] | None):
    """``modules`` without those the rule's interim exemption list names (``exempt`` overrides the list)."""
    skip = EXEMPT[rule] if exempt is None else exempt
    return [module for module in modules if module.relpath not in skip]


def module_aliases(tree, module, wanted) -> dict[str, str]:
    """Local names the module binds to a module path accepted by ``wanted`` (an ofoldx module, ``copy`` …)."""
    return {i.bound: i.refers for i in tree.imports(module) if i.bound and i.refers and wanted(i.refers)}


def layer_direction(tree):
    return [
        f"{i.module}:{i.line} ({layer(i.module)}) imports {i.path} ({target_layer(tree, i.path)})"
        for module in tree.governed()
        for i in tree.imports(module)
        if i.under("ofoldx", "tools") and not may_import(layer(module.name), target_layer(tree, i.path))
    ]


def private_names(tree):
    """A private segment may be imported or reached as an attribute only by a sibling module, and only when it names a
    module."""
    found = []
    for module in tree.governed(*SCOPE):
        aliases = module_aliases(tree, module, lambda path: path.startswith("ofoldx") and tree.owner(path) == path)
        for node, _, _ in module.walk():
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr.startswith("_")
                and not node.attr.startswith("__")
            ):
                target = f"{aliases[node.value.id]}.{node.attr}"
                if not (aliases[node.value.id] == module.base and target in tree):
                    found.append(f"{module.name}:{node.lineno} reaches private {target}")
        for i in tree.imports(module):
            parts = i.path.split(".") if i.under("ofoldx") else ()
            for index, part in enumerate(parts):
                container, prefix = ".".join(parts[:index]), ".".join(parts[: index + 1])
                if (
                    part.startswith("_")
                    and not part.startswith("__")
                    and not (container == module.base and prefix in tree)
                ):
                    found.append(f"{module.name}:{i.line} imports private {prefix}")
    return found


def compute_imports(tree):
    return [
        f"{i.module}:{i.line} imports {i.path}"
        for module in tree.governed(*CORE)
        if layer(module.name) != "tools"
        for i in tree.imports(module)
        if i.under(*COMPUTE)
    ]


def csgraph_imports(tree):
    return [
        f"{i.module}:{i.line} imports {i.path}"
        for module in tree.governed("ofoldx")
        for i in tree.imports(module)
        if i.under("scipy.sparse.csgraph")
    ]


def read_path_imports(tree):
    """No module a core package (``ofoldx.io`` included) loads imports a compute dependency or RDKit itself;
    ``multimolecule.graph``, the one toolkit allowed, brings torch."""
    return [
        f"import {module.name} loads {i.path} via {i.module}:{i.line}"
        for module in tree.governed(*CORE)
        if layer(module.name) != "tools"
        for loaded in sorted(tree.runtime_closure(module.name))
        for i in tree.imports(tree.modules[loaded])
        if i.module_level and i.under(*READ_PATH_FORBIDDEN)
    ]


def confined_imports(tree):
    found = []
    for module in tree.governed("ofoldx"):
        for i in tree.imports(module):
            if i.under(*RDKIT) and not l4_permits(module.name, i):
                found.append(f"{module.name}:{i.line} imports {i.path} outside a function of {RDKIT_HOMES}")
            if i.under("gemmi") and module.name not in {"ofoldx.io.cif", "ofoldx.reference.space_groups.build"}:
                found.append(f"{module.name}:{i.line} imports gemmi outside ofoldx.io.cif")
            if i.under("multimolecule") and layer(module.name) in CORE and not l4_permits(module.name, i):
                found.append(f"{module.name}:{i.line} imports {i.path} beyond multimolecule.graph")
    return found


def offline_imports(tree):
    return [
        f"{i.module}:{i.line} imports offline {i.path}"
        for module in tree.modules.values()
        if layer(module.name) != "tools" and module.name not in OFFLINE
        for i in tree.imports(module)
        if i.under(*OFFLINE)
    ]


def import_cycles(tree):
    graph = {
        name: {tree.owner(i.path) for i in tree.imports(module) if i.module_level and i.under("ofoldx")} - {name}
        for name, module in tree.modules.items()
    }
    governed = {module.name for module in tree.governed(*SCOPE)}
    return [" -> ".join(sorted(c)) for c in strongly_connected(graph) if len(c) > 1 and c & governed]


def local_imports(tree, exempt=None):
    return [
        f"{i.module}:{i.line} imports {i.path} inside {i.function}"
        for module in exempted("L6-no-local-imports", tree.governed("ofoldx"), exempt)
        for i in tree.imports(module)
        if i.function is not None and i.runtime and not l4_permits(module.name, i)
    ]


def frozen_value_writes(tree):
    found = []
    for module in tree.governed(*SCOPE):
        if layer(module.name) == "ofoldx.data":
            continue
        found += [
            f"{i.module}:{i.line} imports {i.path}"
            for i in tree.imports(module)
            if i.under(*(".".join(write) for write in FROZEN_WRITES))
        ]
        aliases = module_aliases(tree, module, lambda path: path in dict(FROZEN_WRITES))
        found += [
            f"{module.name}:{node.lineno} uses {node.value.id}.{node.attr}"
            for node, _, _ in module.walk()
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (
                (aliases.get(node.value.id), node.attr) in FROZEN_WRITES
                or (node.value.id, node.attr) == ("object", "__setattr__")
            )
        ]
    return found


def untyped_failures(tree, exempt=None):
    found = []
    for module in exempted("L9-typed-failures", tree.governed(*CORE, "ofoldx.models"), exempt):
        if layer(module.name) == "tools":
            continue
        for node, _, _ in module.walk():
            if isinstance(node, ast.Raise) and node.exc is not None:
                raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                name = raised.id if isinstance(raised, ast.Name) else getattr(raised, "attr", None)
                if name in UNTYPED or (name in TYPED and module.name != "ofoldx.errors"):
                    found.append(f"{module.name}:{node.lineno} raises {name} (use ofoldx.errors.fail)")
            if isinstance(node, ast.ExceptHandler) and all(silent(statement) for statement in node.body):
                found.append(f"{module.name}:{node.lineno} swallows an exception")
    return found


def silent(statement: ast.stmt) -> bool:
    return isinstance(statement, (ast.Pass, ast.Continue)) or (
        isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and statement.value.value is ...
    )


def strongly_connected(graph: dict[str, set[str]]) -> list[set[str]]:
    """Tarjan's algorithm, iterative."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[set[str]] = []
    for root in graph:
        if root in index:
            continue
        work = [(root, iter(sorted(graph[root])))]
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        while work:
            node, successors = work[-1]
            for successor in successors:
                if successor not in graph:
                    continue
                if successor not in index:
                    index[successor] = low[successor] = len(index)
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(sorted(graph[successor]))))
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[node])
                if low[node] == index[node]:
                    component = set()
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.add(member)
                        if member == node:
                            break
                    components.append(component)
    return components


class Rule(NamedTuple):
    check: Callable[..., list[str]]
    counterexamples: dict[str, dict[str, str]]


RULES = {
    "L1-layer-direction": Rule(
        layer_direction,
        {
            "data-imports-chemistry": {"ofoldx/data/schema.py": "from ofoldx.chemistry import x\n"},
            "io-imports-msa": {"ofoldx/io/read.py": "from ofoldx.msa import read_msa\n"},
            "models-imports-io": {"ofoldx/models/boltz1/processor.py": "from ofoldx.io.read import read_structure\n"},
            "pipelines-imports-io": {"ofoldx/pipelines/input.py": "from ofoldx.io.read import read_structure\n"},
        },
    ),
    "L2-private-names": Rule(
        private_names,
        {
            "private-name": {"ofoldx/io/read.py": "from ofoldx.io.binding import _join\n"},
            "private-module-of-another-package": {"ofoldx/chemistry/realize.py": "from ofoldx.io._cif import parse\n"},
            "private-attribute": {
                "ofoldx/io/binding.py": "def _join(): ...\n",
                "ofoldx/io/read.py": "from ofoldx.io import binding\nbinding._join()\n",
            },
        },
    ),
    "L3-no-compute-in-core": Rule(compute_imports, {"torch": {"ofoldx/data/molecular.py": "import torch\n"}}),
    "L3-no-csgraph": Rule(
        csgraph_imports,
        {"csgraph": {"ofoldx/models/layout.py": "from scipy.sparse.csgraph import connected_components\n"}},
    ),
    "L3-read-path": Rule(
        read_path_imports,
        {"parent-package-loads-torch": {"ofoldx/__init__.py": "import torch\n", "ofoldx/io/read.py": "x = 1\n"}},
    ),
    "L4-confined-imports": Rule(
        confined_imports,
        {
            "rdkit-at-module-level": {"ofoldx/chemistry/rdkit.py": "from rdkit import Chem\n"},
            "gemmi-outside-cif": {"ofoldx/io/source.py": "import gemmi\n"},
            "multimolecule-beyond-graph": {
                "ofoldx/data/assembly.py": "def union():\n    from multimolecule.models import components\n"
            },
        },
    ),
    "L5-no-offline-imports": Rule(
        offline_imports,
        {
            "tools": {"ofoldx/chemistry/definitions.py": "from tools.build_ccd_artifact import build\n"},
            "reference-builder": {"ofoldx/io/symmetry.py": "from ofoldx.reference.space_groups.build import main\n"},
        },
    ),
    "L6-no-cycles": Rule(
        import_cycles,
        {
            "cycle": {
                "ofoldx/io/read.py": "from ofoldx.io import report\n",
                "ofoldx/io/report.py": "from ofoldx.io import read\n",
            }
        },
    ),
    "L6-no-local-imports": Rule(
        local_imports, {"local": {"ofoldx/io/read.py": "def read_structure():\n    from ofoldx.io import report\n"}}
    ),
    "L8-no-frozen-writes": Rule(
        frozen_value_writes,
        {
            "replace": {"ofoldx/io/read.py": "from dataclasses import replace\n"},
            "setattr": {"ofoldx/chemistry/realize.py": "object.__setattr__(value, 'x', 1)\n"},
            "copy-replace": {"ofoldx/io/read.py": "import copy\ncopy.replace(value, x=1)\n"},
            "aliased-replace": {"ofoldx/io/read.py": "import dataclasses as dc\ndc.replace(value, x=1)\n"},
        },
    ),
    "L9-typed-failures": Rule(
        untyped_failures,
        {
            "untyped-raise": {"ofoldx/io/binding.py": "def f():\n    raise ValueError('bad row')\n"},
            "typed-raise-bypasses-fail": {"ofoldx/io/binding.py": "def f():\n    raise ContradictionError('x')\n"},
            "swallowed": {"ofoldx/models/layout.py": "try:\n    x = 1\nexcept KeyError:\n    pass\n"},
        },
    ),
}

COUNTEREXAMPLES = [
    pytest.param(rule.check, files, id=f"{name}-{label}")
    for name, rule in RULES.items()
    for label, files in rule.counterexamples.items()
]


EXEMPTIONS = [pytest.param(RULES[rule].check, path, id=f"{rule}-{path}") for rule in EXEMPT for path in EXEMPT[rule]]


class TestLayering:
    @pytest.mark.parametrize("check", [rule.check for rule in RULES.values()], ids=list(RULES))
    def test_source_tree_obeys_rule(self, check, source_tree):
        assert check(source_tree) == []

    @pytest.mark.parametrize(("check", "files"), COUNTEREXAMPLES)
    def test_rule_rejects_counterexample(self, check, files, make_tree):
        assert check(make_tree(files))

    @pytest.mark.parametrize(("check", "path"), EXEMPTIONS)
    def test_exemption_still_excuses_a_violation(self, check, path, source_tree):
        (name,) = (module.name for module in source_tree.modules.values() if module.relpath == path)
        assert any(found.startswith(f"{name}:") for found in check(source_tree, exempt={}))
