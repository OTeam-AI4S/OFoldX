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
Multiple sequence alignments: reading A3M and aligned-FASTA files, indexed storage, pairing aligned sequences, and
projecting alignment rows onto tokens.
"""

from .alignment import MSA, MsaCollection, MsaRows, msa_rows, unique_rows
from .files import IndexedMsaLoader, load_msa_collection, read_msa, save_msa_collection
from .pairing import pair_msas

__all__ = [
    "MSA",
    "IndexedMsaLoader",
    "MsaCollection",
    "MsaRows",
    "load_msa_collection",
    "msa_rows",
    "pair_msas",
    "read_msa",
    "save_msa_collection",
    "unique_rows",
]
