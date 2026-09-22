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

"""Shared source-tree fixtures for the architecture checks.

Every check here reads source text and ASTs; nothing under ``ofoldx`` is imported, so the gate runs without torch or
any compiled dependency and states what *would* be imported, not what one interpreter happened to load.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGES = ("ofoldx", "tools")
EXTENSION_SOURCES = (".so", ".pyd", ".cpp", ".c", ".pyx")  # a compiled module counts as a module of its package


@dataclass(frozen=True)
class Import:
    """One imported name as written: ``path`` is the absolute dotted path (``from a import b`` gives ``a.b``);
    ``bound`` is the local name the statement binds and ``refers`` the dotted path that name stands for (``import a.b``
    binds ``a``; ``import a.b as c`` binds ``c`` to ``a.b``; a star import binds nothing)."""

    module: str
    path: str
    line: int
    function: str | None
    type_checking: bool
    bound: str | None = None
    refers: str | None = None

    @property
    def runtime(self) -> bool:
        return not self.type_checking

    @property
    def module_level(self) -> bool:
        return self.function is None and not self.type_checking

    def under(self, *prefixes: str) -> bool:
        return any(self.path == prefix or self.path.startswith(prefix + ".") for prefix in prefixes)


@dataclass(frozen=True)
class Module:
    name: str
    file: Path
    relpath: str
    package: bool

    @cached_property
    def text(self) -> str:
        return self.file.read_text()

    @cached_property
    def tree(self) -> ast.Module:
        return ast.parse(self.text, filename=self.relpath)

    @property
    def lines(self) -> int:
        return len(self.text.splitlines())

    @property
    def base(self) -> str:
        return self.name if self.package else self.name.rpartition(".")[0]

    def walk(self) -> Iterator[tuple[ast.AST, str | None, bool]]:
        """Yield ``(node, enclosing function qualname, inside if TYPE_CHECKING)`` for every node."""

        def visit(node: ast.AST, scope: tuple[str, ...], function: str | None, checking: bool):
            for child in ast.iter_child_nodes(node):
                yield child, function, checking
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = ".".join((*scope, child.name))
                    yield from visit(child, (*scope, child.name), qualname, checking)
                elif isinstance(child, ast.ClassDef):
                    yield from visit(child, (*scope, child.name), function, checking)
                elif isinstance(child, ast.If) and _is_type_checking(child.test):
                    for statement in child.body:
                        yield statement, function, True
                        yield from visit(statement, scope, function, True)
                    for statement in child.orelse:
                        yield statement, function, checking
                        yield from visit(statement, scope, function, checking)
                else:
                    yield from visit(child, scope, function, checking)

        yield from visit(self.tree, (), None, False)

    @cached_property
    def top_level_names(self) -> frozenset[str]:
        """Names bound at module scope; ``*`` stands for "anything" (star import or module ``__getattr__``)."""
        names: set[str] = set()

        def bind(statements):
            for node in statements:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        names.add("*" if alias.name == "*" else (alias.asname or alias.name.split(".")[0]))
                elif isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                    for field in ("body", "orelse", "finalbody"):
                        bind(getattr(node, field, ()))
                    for handler in getattr(node, "handlers", ()):
                        bind(handler.body)

        bind(self.tree.body)
        if "__getattr__" in names:
            names.add("*")
        return frozenset(names)

    def defines(self, name: str) -> bool:
        return name in self.top_level_names or "*" in self.top_level_names


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


class SourceTree:
    """The modules under ``ofoldx/`` and ``tools/`` of one checkout."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules: dict[str, Module] = {}
        for package in PACKAGES:
            for file in sorted((root / package).rglob("*.py")):
                relpath = file.relative_to(root).as_posix()
                parts = relpath[: -len(".py")].split("/")
                package_init = parts[-1] == "__init__"
                name = ".".join(parts[:-1] if package_init else parts)
                self.modules[name] = Module(name, file, relpath, package_init)
        self.extensions = frozenset(
            ".".join((*file.relative_to(root).parent.parts, file.name.split(".")[0]))
            for package in PACKAGES
            for file in (root / package).rglob("*")
            if file.suffix in EXTENSION_SOURCES
        )

    def __contains__(self, name: str) -> bool:
        return name in self.modules or name in self.extensions

    def governed(self, *prefixes: str) -> list[Module]:
        """Modules the layering rules apply to, optionally only those under ``prefixes``."""
        return [
            module
            for name, module in self.modules.items()
            if (not prefixes or any(name == p or name.startswith(p + ".") for p in prefixes))
        ]

    def owner(self, path: str) -> str:
        """The module a dotted import path lands in (longest existing prefix), or the path itself if none exists."""
        candidate = path
        while candidate and candidate not in self:
            candidate = candidate.rpartition(".")[0]
        return candidate or path

    def imports(self, module: Module) -> list[Import]:
        return self._imports[module.name]

    @cached_property
    def _imports(self) -> dict[str, list[Import]]:
        return {name: list(self._read_imports(module)) for name, module in self.modules.items()}

    def _read_imports(self, module: Module) -> Iterator[Import]:
        for node, function, checking in module.walk():
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refers = alias.name if alias.asname else alias.name.split(".")[0]
                    bound = alias.asname or refers
                    yield Import(module.name, alias.name, node.lineno, function, checking, bound, refers)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module.base.split(".")
                    base = base[: len(base) - node.level + 1]
                    origin = ".".join([*base, *([node.module] if node.module else [])])
                else:
                    origin = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        yield Import(module.name, origin, node.lineno, function, checking)
                    else:
                        path = f"{origin}.{alias.name}"
                        yield Import(
                            module.name, path, node.lineno, function, checking, alias.asname or alias.name, path
                        )

    def runtime_closure(self, start: str, exempt: frozenset[str] = frozenset()) -> set[str]:
        """The ofoldx modules ``import start`` executes (module-level runtime imports plus parent packages), not
        descending into ``exempt`` ones."""
        seen: set[str] = set()
        stack = [start]
        while stack:
            name = stack.pop()
            if name in seen or name not in self.modules or name in exempt:
                continue
            seen.add(name)
            parts = name.split(".")
            stack.extend(".".join(parts[:i]) for i in range(1, len(parts)))
            stack.extend(self.owner(i.path) for i in self.imports(self.modules[name]) if i.module_level)
        return seen


def parse_list(text: str) -> tuple[str, ...]:
    """Lines of a bookkeeping file without blanks and ``#`` comments."""
    lines = (line.split("#", 1)[0].strip() for line in text.splitlines())
    return tuple(line for line in lines if line)


def read_list(path: Path) -> tuple[str, ...]:
    return parse_list(path.read_text()) if path.exists() else ()


@pytest.fixture(scope="session")
def source_tree() -> SourceTree:
    return SourceTree(ROOT)


@pytest.fixture
def make_tree(tmp_path) -> Callable[..., SourceTree]:
    """Build a throwaway checkout from ``{relative path: source}`` to show a rule rejects what it forbids."""

    def make(files: Mapping[str, str]) -> SourceTree:
        for relpath, text in files.items():
            file = tmp_path / relpath
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text)
            for parent in file.relative_to(tmp_path).parents:
                init = tmp_path / parent / "__init__.py"
                if parent.parts and not init.exists():
                    init.write_text("")
        return SourceTree(tmp_path)

    return make
