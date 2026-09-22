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
The ``ReadReport`` of one read and the records it holds.

What was read (source, the selection that took effect, the definitions the value was built from and the CCD artifact
they were resolved against), every Conflict, Unknown, unrealised declaration and finding as ``Records``, the
counted-only standard derivations and the document metadata. Identity facts, states and masks live in the
``Structure``; valence excess and exclusion policy belong to the consumer.

Subjects index the ``Structure`` whose digest ``structure_digest`` names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ..chemistry.declarations import AssumptionSet
from ..chemistry.definitions import Definitions
from ..errors import RecordKind
from ..provenance import Records, SourceRefs
from .tables import DocumentMetadata


@dataclass(frozen=True)
class SourceInfo:
    r"""The source document: its path, the digest of its decompressed content, format, block and the reader version
    (with the stabiliser rule version and the assumption rule version)."""

    path: str | None
    sha256: str
    format: Literal["mmcif", "pdb"]
    block: str | None
    reader_version: str


@dataclass(frozen=True)
class AppliedSelection:
    r"""The selection that took effect: the model read, the assembly, the altloc policy and the consumer assumption
    rules enabled (the facts they decided carry the KNOWN×ASSUMED state)."""

    model_number: int | None
    assembly: str | None
    altloc: str
    assumptions: AssumptionSet


@dataclass(frozen=True, eq=False)
class ReadReport:
    r"""
    The report of one read. ``records`` hold every Conflict, Unknown, unrealised declaration and finding (kind
    from the code); ``bulk`` the counted-only standard derivations (``standard_junction_endpoint``: endpoints);
    ``refs`` the read's references; the CCD artifact consulted is ``definitions.library``.
    """

    source: SourceInfo
    selection: AppliedSelection
    metadata: DocumentMetadata
    definitions: Definitions
    structure_digest: str
    records: Records
    bulk: Mapping[str, int]
    refs: SourceRefs

    def conflicts(self) -> Records:
        return self.records.where(kind=RecordKind.CONFLICT)

    def unknowns(self) -> Records:
        return self.records.where(kind=RecordKind.UNKNOWN)

    def unrealised(self) -> Records:
        return self.records.where(kind=RecordKind.UNREALISED)

    def findings(self) -> Records:
        return self.records.where(kind=RecordKind.FINDING)

    def summary(self) -> Mapping[str, int]:
        r"""Records per code name and ``bulk.<key>``."""
        return dict(self.records.counts()) | {f"bulk.{key}": int(value) for key, value in self.bulk.items()}

    def __reduce__(self):  # a MappingProxyType does not pickle (e.g. across processes)
        return _report, ({**vars(self), "bulk": dict(self.bulk)},)


def _report(values: dict) -> ReadReport:
    return ReadReport(**(values | {"bulk": MappingProxyType(values["bulk"])}))
