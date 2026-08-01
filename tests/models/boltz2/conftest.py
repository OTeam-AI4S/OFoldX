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

"""Builder of the Boltz-2 tests: templates of a structure's chains."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ofoldx.data.structure import Structure, StructureArrays
from ofoldx.provenance import Basis, Resolution, state
from ofoldx.templates import Template, TemplateAlignment, chain_coordinates


@pytest.fixture
def template() -> Callable[..., Template]:
    """``template(structure, "A", (target_instance, ((0, 0), (1, 1))), ...)``: a template of chain ``A`` of
    ``structure`` (author chains named as the instances), aligned to entity ``P``."""

    def make(structure: Structure, chain: str, *alignments: tuple[str | None, tuple]) -> Template:
        arrays = structure.build()
        arrays.columns["instance_author_chain"][:] = structure.system.instance_symbol
        arrays.columns["instance_author_chain_state"][:] = state(Resolution.KNOWN, Basis.DECLARED)
        symbols = {**arrays.symbols, "chain": tuple(structure.system.symbols()["instance"])}
        source = Structure.from_arrays(structure.system, StructureArrays(symbols, arrays.columns))
        return Template(
            chain_coordinates(source, [chain]),
            tuple(TemplateAlignment("P", target, chain, pairs) for target, pairs in alignments),
        )

    return make
