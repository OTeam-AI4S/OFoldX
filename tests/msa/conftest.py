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

r"""Pools and token layouts shared by the MSA tests."""

from __future__ import annotations

import numpy as np
import pytest

from ofoldx.data.tokenization import TOKEN_INDEX, token_classes, tokenize
from ofoldx.msa import MSA


@pytest.fixture
def pool():
    """A pool with zero deletions: unpaired ``(query, *rows)``, paired ``(query, *paired rows)`` with their headers."""

    def build(entity, query, *rows, paired=()):
        unpaired = (query, *rows)
        paired_rows = (query, *(row for _, row in paired))
        return MSA(
            entity,
            unpaired,
            np.zeros((len(unpaired), len(query)), dtype=np.uint32),
            paired_rows,
            np.zeros((len(paired_rows), len(query)), dtype=np.uint32),
            ("query", *(header for header, _ in paired)),
        )

    return build


@pytest.fixture
def layout():
    """Polymer components one token each (those in ``per_atom`` one per atom), other components one per atom."""

    def build(system, per_atom=()):
        residue = token_classes(system).token != TOKEN_INDEX["OTHER"]
        residue[list(per_atom)] = False
        return tokenize(system, residue_token=residue, centre_atom=system.component_atom_ptr[:-1])

    return build
