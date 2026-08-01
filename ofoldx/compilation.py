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
Explicit ownership contracts for regional compilation.
"""

from __future__ import annotations

from torch import nn

CompilationRegionGroup = tuple[nn.Module, ...]
CompilationRegionGroups = tuple[CompilationRegionGroup, ...]


class CompilationRegionOwner:
    r"""Mixin for modules that explicitly own local compilation regions."""

    def local_compilation_region_groups(self) -> CompilationRegionGroups:
        r"""Return this owner's local groups of interchangeable regions."""

        raise NotImplementedError


class HomogeneousBlockRegions(CompilationRegionOwner):
    r"""Expose a homogeneous ``blocks`` collection as one compilation group."""

    blocks: nn.ModuleList

    def local_compilation_region_groups(self) -> CompilationRegionGroups:
        blocks = tuple(self.blocks)
        return (blocks,) if blocks else ()
