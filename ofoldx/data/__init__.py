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

r"""Molecular identities, coordinates, and evidence for structure workflows.

Immutable NumPy values support molecular selection, assembly, chemical edits, and
coordinate replacement. This layer is independent of PyTorch, RDKit, and source readers.
"""

from .assembly import batch, replicate, unbatch, union
from .editing import ChemicalEdit, apply_edit
from .molecular import AtomSite, BondKind, ComponentRef, EndpointState, MolecularSystem, MonomerType
from .projection import fold_hydrogens, project_atoms, project_instances, select
from .structure import CoordinateEvidence, CopyId, CopyKind, Structure, replace_coordinates
from .tokenization import TOKENS, TokenLayout, token_classes, tokenize

__all__ = [
    "TOKENS",
    "AtomSite",
    "BondKind",
    "ChemicalEdit",
    "ComponentRef",
    "CoordinateEvidence",
    "CopyId",
    "CopyKind",
    "EndpointState",
    "MolecularSystem",
    "MonomerType",
    "Structure",
    "TokenLayout",
    "apply_edit",
    "batch",
    "fold_hydrogens",
    "project_atoms",
    "project_instances",
    "replace_coordinates",
    "replicate",
    "select",
    "token_classes",
    "tokenize",
    "unbatch",
    "union",
]
