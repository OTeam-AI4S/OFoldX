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
Sources to canonical values and back: mmCIF and PDB reading through one read path (``read_structure``), writing,
native artifacts and CIF syntax. Multiple sequence alignments live in ``ofoldx.msa``.
"""

from __future__ import annotations

from .cif import parse_cif, read_cif
from .native import load_definitions, load_structure, save_structure
from .read import read_structure
from .report import ReadReport
from .source import Source, parse_source, read_source
from .tables import DocumentMetadata
from .write import WriteReport, write_structure

__all__ = [
    "read_source",
    "parse_source",
    "Source",
    "read_structure",
    "ReadReport",
    "DocumentMetadata",
    "write_structure",
    "WriteReport",
    "save_structure",
    "load_structure",
    "load_definitions",
    "read_cif",
    "parse_cif",
]
